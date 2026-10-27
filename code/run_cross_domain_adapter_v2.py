"""
run_cross_domain_adapter_v2.py - Cross-Domain Adapter (L_recon + L_MMD + L_consist)

구조:
  1단계: Source(A)로 전체 모델 학습
  2단계: 전체 모델 freeze + Adapter 삽입
  3단계: Target(B) 데이터로 Adapter 학습
         Loss = L_recon + α × L_MMD + β × L_consist
  4단계: 적응된 모델로 Target 테스트

비교:
  - Baseline: A 학습 → B 그냥 테스트
  - Adapter (recon only): L_recon만 사용
  - Adapter (recon + MMD): L_recon + L_MMD
  - Adapter (full): L_recon + L_MMD + L_consist  ← 제안 방법
  - Full FT: 전체 fine-tuning

실행:
  python run_cross_domain_adapter_v2.py --experiment xjtu --epochs_pretrain 200 --epochs_adapt 50
  python run_cross_domain_adapter_v2.py --experiment phm --epochs_pretrain 200 --epochs_adapt 50
  python run_cross_domain_adapter_v2.py --experiment pipe --epochs_pretrain 100 --epochs_adapt 50
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '.')

from config import model_cfg, train_cfg
from models import create_model, select_best_head, BaseAnomalyModel
from trainer import train_model


FEATURE_COLS = ['rms_h', 'rms_v', 'peak_h', 'peak_v',
                'kurtosis_h', 'kurtosis_v', 'std_h', 'std_v']


# =============================================================
# MMD 계산
# =============================================================
def compute_mmd(x, y, kernel='rbf', bandwidth=1.0):
    """
    Maximum Mean Discrepancy between two sets of samples.
    x: (N, D) - source latent samples
    y: (M, D) - adapted target latent samples
    
    MMD가 0에 가까우면 두 분포가 비슷한 것.
    """
    def rbf_kernel(a, b, bw):
        # (N, M)
        dist = torch.cdist(a, b, p=2)
        return torch.exp(-dist ** 2 / (2 * bw ** 2))
    
    xx = rbf_kernel(x, x, bandwidth)
    yy = rbf_kernel(y, y, bandwidth)
    xy = rbf_kernel(x, y, bandwidth)
    
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


# =============================================================
# Head Consistency Loss (L_consist)
# =============================================================
def compute_head_consistency_loss(source_head_probs, target_head_probs):
    """
    Source에서의 head 선택 패턴과 Target에서의 head 선택 패턴이
    일관되도록 KL divergence로 정규화.
    
    source_head_probs: (B_src, W, N) - source의 head별 선택 확률
    target_head_probs: (B_tgt, W, N) - target의 head별 선택 확률
    
    시점별(W) 평균 분포를 비교.
    """
    # 시점별 평균 head 선택 분포
    src_dist = source_head_probs.mean(dim=0)  # (W, N)
    tgt_dist = target_head_probs.mean(dim=0)  # (W, N)
    
    # KL(target || source) - target이 source 패턴을 따르도록
    kl = F.kl_div(tgt_dist.log(), src_dist, reduction='batchmean')
    return kl


# =============================================================
# Adapter 모듈 (이전과 동일)
# =============================================================
class LatentAdapter(nn.Module):
    def __init__(self, latent_dim, bottleneck=None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(latent_dim // 4, 16)
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, latent_dim),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x):
        return x + self.adapter(x)


# =============================================================
# Adapted Model (encoder → adapter → decoder)
# =============================================================
class AdaptedModel(BaseAnomalyModel):
    def __init__(self, base_model, adapter, model_type):
        super().__init__()
        self.base = base_model
        self.adapter = adapter
        self.model_type = model_type

    def forward(self, x):
        if self.model_type == 'lstm_ae':
            return self._forward_lstm(x)
        elif self.model_type == 'transformer_ae':
            return self._forward_transformer(x)
        elif self.model_type == 'tcn_ae':
            return self._forward_tcn(x)

    def encode_to_latent(self, x):
        """encoder만 실행해서 latent 반환 (MMD 계산용)"""
        if self.model_type == 'lstm_ae':
            _, (h_n, _) = self.base.encoder(x)
            return self.base.enc_to_latent(h_n[-1])
        elif self.model_type == 'transformer_ae':
            enc = self.base.enc_proj(x)
            enc = self.base.enc_pos(enc)
            enc = self.base.transformer_enc(enc)
            return enc.mean(dim=1)
        elif self.model_type == 'tcn_ae':
            x_conv = x.permute(0, 2, 1)
            enc = self.base.encoder(x_conv)
            return enc.mean(dim=-1)

    def _forward_lstm(self, x):
        B, W, F = x.shape
        m = self.base
        _, (h_n, c_n) = m.encoder(x)
        latent = m.enc_to_latent(h_n[-1])
        latent = self.adapter(latent)
        dec_input = m.latent_to_dec(latent)
        dec_input = dec_input.unsqueeze(1).repeat(1, W, 1)
        dec_out, _ = m.decoder(dec_input)
        heads = torch.stack([head(dec_out) for head in m.output_heads], dim=0)
        return heads

    def _forward_transformer(self, x):
        B, W, F = x.shape
        m = self.base
        enc = m.enc_proj(x)
        enc = m.enc_pos(enc)
        enc = m.transformer_enc(enc)
        latent = enc.mean(dim=1)
        latent = self.adapter(latent)
        dec = m.dec_proj(latent)
        dec = dec.unsqueeze(1).repeat(1, W, 1)
        dec = m.dec_pos(dec)
        dec = m.transformer_dec(dec)
        heads = torch.stack([head(dec) for head in m.output_heads], dim=0)
        return heads

    def _forward_tcn(self, x):
        B, W, F = x.shape
        m = self.base
        x_conv = x.permute(0, 2, 1)
        enc = m.encoder(x_conv)
        latent = enc.mean(dim=-1)
        latent = self.adapter(latent)
        dec = latent.unsqueeze(-1).repeat(1, 1, W)
        dec = m.decoder(dec)
        dec = dec.permute(0, 2, 1)
        heads = torch.stack([head(dec) for head in m.output_heads], dim=0)
        return heads


# =============================================================
# 유틸리티
# =============================================================
def load_and_normalize(train_path, test_path, feature_cols, scaler_type='minmax'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_data = train_df[feature_cols].values.astype(np.float32)
    test_data = test_df[feature_cols].values.astype(np.float32)
    if scaler_type == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data)
    test_norm = scaler.transform(test_data)
    return train_norm, test_norm


def make_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)


def make_dataloader(data_norm, window_size, batch_size, shuffle=True):
    from torch.utils.data import DataLoader, TensorDataset
    windows = make_windows(data_norm, window_size)
    tensor = torch.FloatTensor(windows)
    dataset = TensorDataset(tensor)
    def collate_fn(batch):
        return torch.stack([b[0] for b in batch])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


@torch.no_grad()
def compute_test_mae(model, data_norm, window_size, device):
    model.eval()
    data_tensor = torch.FloatTensor(data_norm).to(device)
    errors = []
    for i in range(len(data_norm) - window_size + 1):
        window = data_tensor[i:i+window_size].unsqueeze(0)
        heads = model(window)
        best, _, _ = select_best_head(heads, window)
        err = torch.mean(torch.abs(best - window)).item()
        errors.append(err)
    return round(float(np.mean(errors)), 5)


def get_latent_dim(model_name):
    if model_name == 'lstm_ae':
        return model_cfg.lstm_latent
    elif model_name == 'transformer_ae':
        return model_cfg.d_model
    elif model_name == 'tcn_ae':
        return model_cfg.tcn_channels[-1]


# =============================================================
# Source latent & head prob 수집
# =============================================================
@torch.no_grad()
def collect_source_stats(adapted_model, source_loader, device):
    adapted_model.eval()
    all_latents = []
    all_head_probs = []

    for batch in source_loader:
        x = batch.to(device)
        # latent 수집
        latent = adapted_model.encode_to_latent(x)
        all_latents.append(latent)

        # head별 error 계산
        heads = adapted_model(x)  # (N, B, W, F)
        # x를 heads와 같은 shape으로 맞추기
        x_expanded = x.unsqueeze(0).repeat(heads.shape[0], 1, 1, 1)  # (N, B, W, F)
        errors = torch.mean(torch.abs(heads - x_expanded), dim=-1)  # (N, B, W)
        # (N, B, W) → (B, W, N) → softmax
        errors_bwn = errors.permute(1, 2, 0)  # (B, W, N)
        head_probs = torch.softmax(-errors_bwn, dim=-1)  # (B, W, N)
        all_head_probs.append(head_probs)

    source_latents = torch.cat(all_latents, dim=0)
    source_head_probs = torch.cat(all_head_probs, dim=0)

    return source_latents, source_head_probs


# =============================================================
# Adapter 학습 (3개 Loss)
# =============================================================
def train_adapter_v2(adapted_model, tgt_train_loader, tgt_val_loader,
                     source_latents, source_head_probs,
                     epochs, lr, device, alpha=1.0, beta=0.1, loss_mode='full'):
    """
    loss_mode:
      'recon_only': L_recon만
      'recon_mmd':  L_recon + α × L_MMD
      'full':       L_recon + α × L_MMD + β × L_consist
    """
    optimizer = torch.optim.Adam(adapted_model.adapter.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    # source latent에서 랜덤 샘플링을 위한 인덱스
    n_source = source_latents.size(0)

    for epoch in range(1, epochs + 1):
        adapted_model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_mmd = 0.0
        total_consist = 0.0

        for batch in tgt_train_loader:
            x = batch.to(device)
            B = x.size(0)
            optimizer.zero_grad()

            # ── L_recon ──
            loss_recon, best_output, best_indices = adapted_model.compute_loss(x)

            # ── L_MMD ──
            loss_mmd = torch.tensor(0.0, device=device)
            if loss_mode in ['recon_mmd', 'full']:
                # target의 adapted latent
                target_latent = adapted_model.encode_to_latent(x)  # (B, D)
                adapted_latent = adapted_model.adapter(target_latent)  # (B, D)

                # source에서 같은 수만큼 랜덤 샘플
                idx = torch.randint(0, n_source, (B,))
                src_sample = source_latents[idx].to(device)

                loss_mmd = compute_mmd(src_sample, adapted_latent)

            # ── L_consist ──
            loss_consist = torch.tensor(0.0, device=device)
            if loss_mode == 'full':
                # target의 head 선택 확률
                heads = adapted_model(x)  # (N, B, W, F)
                N_heads, _, W, F_dim = heads.shape
                errors = torch.mean(torch.abs(heads - x.unsqueeze(0)), dim=-1)  # (N, B, W)
                target_probs = F.softmax(-errors.permute(1, 2, 0), dim=-1)  # (B, W, N)

                # source에서 같은 수만큼 랜덤 샘플
                idx_h = torch.randint(0, source_head_probs.size(0), (B,))
                src_probs = source_head_probs[idx_h].to(device)  # (B, W, N)

                loss_consist = compute_head_consistency_loss(src_probs, target_probs)

            # ── Total ──
            loss = loss_recon + alpha * loss_mmd + beta * loss_consist
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted_model.adapter.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_mmd += loss_mmd.item()
            total_consist += loss_consist.item()

        n_batches = len(tgt_train_loader)
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_mmd = total_mmd / n_batches
        avg_consist = total_consist / n_batches

        # validation
        adapted_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tgt_val_loader:
                x = batch.to(device)
                l, _, _ = adapted_model.compute_loss(x)
                val_loss += l.item()
        val_loss /= len(tgt_val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(adapted_model.adapter.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | total={avg_loss:.5f} "
                  f"(recon={avg_recon:.5f} mmd={avg_mmd:.5f} consist={avg_consist:.5f}) "
                  f"| val={val_loss:.5f}")

        if patience_counter >= 20:
            print(f"    Early Stop at epoch {epoch}")
            break

    if best_state is not None:
        adapted_model.adapter.load_state_dict(best_state)

    return best_val_loss


# =============================================================
# Full Fine-tuning (비교용)
# =============================================================
def full_finetune(model, train_loader, val_loader, epochs, lr, device):
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = model.compute_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                l, _, _ = model.compute_loss(x)
                val_loss += l.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    FullFT Epoch {epoch:3d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        if patience_counter >= 20:
            print(f"    FullFT Early Stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_loss


# =============================================================
# 한 모델에 대해 5가지 방법 비교
# =============================================================
def run_one(model_name, source_norm, target_norm, window_size,
            epochs_pretrain, epochs_adapt, device, adapt_lr, alpha, beta):

    results = {}

    # source split
    n_src_train = int(len(source_norm) * 0.8)
    src_train = source_norm[:n_src_train]
    src_val = source_norm[n_src_train:]
    src_train_loader = make_dataloader(src_train, window_size, train_cfg.batch_size, shuffle=True)
    src_val_loader = make_dataloader(src_val, window_size, train_cfg.batch_size, shuffle=False)

    # target split (60/20/20)
    n_tgt = len(target_norm)
    n_adapt_train = int(n_tgt * 0.6)
    n_adapt_val = int(n_tgt * 0.8)
    tgt_adapt_train = target_norm[:n_adapt_train]
    tgt_adapt_val = target_norm[n_adapt_train:n_adapt_val]
    tgt_test = target_norm[n_adapt_val:]
    tgt_train_loader = make_dataloader(tgt_adapt_train, window_size, train_cfg.batch_size, shuffle=True)
    tgt_val_loader = make_dataloader(tgt_adapt_val, window_size, train_cfg.batch_size, shuffle=False)

    # ── 1단계: Source 학습 ──
    print(f"    [Source 학습]")
    train_cfg.epochs = epochs_pretrain
    model, trainer, history = train_model(
        model_name, model_cfg, train_cfg, src_train_loader, src_val_loader, verbose=False
    )
    src_best_val = round(float(history['best_val_loss']), 5)

    # ── Source stats 수집 ──
    latent_dim = get_latent_dim(model_name)
    dummy_adapter = LatentAdapter(latent_dim).to(device)
    nn.init.zeros_(dummy_adapter.adapter[0].weight)
    nn.init.zeros_(dummy_adapter.adapter[0].bias)
    nn.init.zeros_(dummy_adapter.adapter[2].weight)
    nn.init.zeros_(dummy_adapter.adapter[2].bias)

    source_model = AdaptedModel(model, dummy_adapter, model_name).to(device)
    source_latents, source_head_probs = collect_source_stats(source_model, src_train_loader, device)
    print(f"    Source stats: latents={source_latents.shape}, head_probs={source_head_probs.shape}")

    # ── Baseline ──
    print(f"    [Baseline]")
    baseline_mae = compute_test_mae(model, tgt_test, window_size, device)
    results['Baseline'] = {'method': 'Baseline', 'test_mae': baseline_mae}
    print(f"      Test MAE = {baseline_mae}")

    # ── Adapter (recon only) - ablation용 ──
    print(f"    [Adapter: recon only]")
    adapter_r = LatentAdapter(latent_dim).to(device)
    base_r = copy.deepcopy(model)
    for p in base_r.parameters():
        p.requires_grad = False
    adapted_r = AdaptedModel(base_r, adapter_r, model_name).to(device)
    train_adapter_v2(adapted_r, tgt_train_loader, tgt_val_loader,
                     source_latents, source_head_probs,
                     epochs_adapt, adapt_lr, device, alpha=0, beta=0, loss_mode='recon_only')
    mae_r = compute_test_mae(adapted_r, tgt_test, window_size, device)
    results['Adapter(recon)'] = {'method': 'Adapter(recon)', 'test_mae': mae_r}
    print(f"      Test MAE = {mae_r}")

    # ── Adapter (recon + MMD) - ablation용 ──
    print(f"    [Adapter: recon + MMD]")
    adapter_rm = LatentAdapter(latent_dim).to(device)
    base_rm = copy.deepcopy(model)
    for p in base_rm.parameters():
        p.requires_grad = False
    adapted_rm = AdaptedModel(base_rm, adapter_rm, model_name).to(device)
    train_adapter_v2(adapted_rm, tgt_train_loader, tgt_val_loader,
                     source_latents, source_head_probs,
                     epochs_adapt, adapt_lr, device, alpha=alpha, beta=0, loss_mode='recon_mmd')
    mae_rm = compute_test_mae(adapted_rm, tgt_test, window_size, device)
    results['Adapter(+MMD)'] = {'method': 'Adapter(+MMD)', 'test_mae': mae_rm}
    print(f"      Test MAE = {mae_rm}")

    # ── Adapter (full: recon + MMD + consist) - 제안 방법 ──
    print(f"    [Adapter: FULL (recon + MMD + consist)]")
    adapter_f = LatentAdapter(latent_dim).to(device)
    base_f = copy.deepcopy(model)
    for p in base_f.parameters():
        p.requires_grad = False
    adapted_f = AdaptedModel(base_f, adapter_f, model_name).to(device)
    train_adapter_v2(adapted_f, tgt_train_loader, tgt_val_loader,
                     source_latents, source_head_probs,
                     epochs_adapt, adapt_lr, device, alpha=alpha, beta=beta, loss_mode='full')
    mae_f = compute_test_mae(adapted_f, tgt_test, window_size, device)
    results['Adapter(FULL)'] = {'method': 'Adapter(FULL)', 'test_mae': mae_f}
    print(f"      Test MAE = {mae_f}")

    # ── Full FT ──
    print(f"    [Full Fine-tuning]")
    ft_model = copy.deepcopy(model)
    full_finetune(ft_model, tgt_train_loader, tgt_val_loader, epochs_adapt, adapt_lr, device)
    mae_ft = compute_test_mae(ft_model, tgt_test, window_size, device)
    results['Full FT'] = {'method': 'Full FT', 'test_mae': mae_ft}
    print(f"      Test MAE = {mae_ft}")

    return results, src_best_val


# =============================================================
# 실험 설정
# =============================================================
EXPERIMENTS = {
    'xjtu': {
        'name': 'XJTU-SY (XB1 ↔ XB2)',
        'pairs': [
            ('XB1→XB2', 'data/xjtu_sy/XB1_combined.csv', 'data/xjtu_sy/XB2_combined.csv'),
            ('XB2→XB1', 'data/xjtu_sy/XB2_combined.csv', 'data/xjtu_sy/XB1_combined.csv'),
        ],
        'features': FEATURE_COLS, 'n_features': 8,
    },
    'phm': {
        'name': 'PHM2012 (PB1 ↔ PB2)',
        'pairs': [
            ('PB1→PB2', 'data/phm2012/PB1_combined.csv', 'data/phm2012/PB2_combined.csv'),
            ('PB2→PB1', 'data/phm2012/PB2_combined.csv', 'data/phm2012/PB1_combined.csv'),
        ],
        'features': FEATURE_COLS, 'n_features': 8,
    },
    'pipe': {
        'name': '배관 (Sim → Real)',
        'pairs': [
            ('Sim→Real', 'data/pipe/simCase1_pfei.csv', 'data/pipe/realCase1_pfei.csv'),
        ],
        'features': None, 'n_features': 5,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, choices=['xjtu', 'phm', 'pipe'])
    parser.add_argument('--models', nargs='*', default=['lstm_ae', 'transformer_ae', 'tcn_ae'])
    parser.add_argument('--epochs_pretrain', type=int, default=200)
    parser.add_argument('--epochs_adapt', type=int, default=50)
    parser.add_argument('--adapt_lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='MMD loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='Head consistency loss weight')
    parser.add_argument('--n_heads', type=int, default=5)
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    model_cfg.n_heads = args.n_heads
    model_cfg.input_size = exp['n_features']
    model_cfg.window_size = 20
    model_cfg.d_model = 128
    model_cfg.dim_ff = 512
    model_cfg.lstm_hidden = 128
    model_cfg.lstm_latent = 64
    model_cfg.tcn_channels = [128, 128, 128]
    device = train_cfg.device

    save_dir = os.path.join('results', f'adapter_v2_{args.experiment}')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"  {exp['name']} + Adapter v2 (L_recon + L_MMD + L_consist)")
    print("=" * 70)
    print(f"  alpha={args.alpha}, beta={args.beta}")
    print("=" * 70)

    all_rows = []

    for pair_name, src_path, tgt_path in exp['pairs']:
        print(f"\n{'━'*60}")
        print(f"  {pair_name}")
        print(f"{'━'*60}")

        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            print(f"  [에러] 파일 없음")
            continue

        if args.experiment == 'pipe':
            from pipe_columns import SIM_COLS as sim_cols, REAL_COLS as real_cols  # NDA
            src_df = pd.read_csv(src_path)
            tgt_df = pd.read_csv(tgt_path)
            from sklearn.preprocessing import MinMaxScaler
            s1, s2 = MinMaxScaler(), MinMaxScaler()
            source_norm = s1.fit_transform(src_df[sim_cols].values.astype(np.float32))
            target_norm = s2.fit_transform(tgt_df[real_cols].values.astype(np.float32))
        else:
            source_norm, target_norm = load_and_normalize(src_path, tgt_path, exp['features'])

        print(f"  Source: {source_norm.shape} | Target: {target_norm.shape}")

        for model_name in args.models:
            print(f"\n  [{model_name}]")

            results, src_val = run_one(
                model_name, source_norm, target_norm,
                model_cfg.window_size, args.epochs_pretrain, args.epochs_adapt,
                device, args.adapt_lr, args.alpha, args.beta
            )

            for method, res in results.items():
                all_rows.append({
                    'direction': pair_name,
                    'model': model_name,
                    'method': res['method'],
                    'test_mae': res['test_mae'],
                    'src_best_val': src_val,
                })

    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    csv_path = os.path.join(save_dir, f'adapter_v2_{args.experiment}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[저장] {csv_path}")

    # 개선율 분석
    print("\n[개선율 분석]")
    for pair_name in df['direction'].unique():
        print(f"\n  {pair_name}:")
        for model_name in df['model'].unique():
            sub = df[(df['direction'] == pair_name) & (df['model'] == model_name)]
            if len(sub) == 0:
                continue
            baseline = sub[sub['method'] == 'Baseline']['test_mae'].values[0]
            for _, row in sub.iterrows():
                if row['method'] == 'Baseline':
                    continue
                improve = (baseline - row['test_mae']) / baseline * 100
                print(f"    {model_name} {row['method']}: {baseline} → {row['test_mae']} ({improve:+.1f}%)")


if __name__ == '__main__':
    main()

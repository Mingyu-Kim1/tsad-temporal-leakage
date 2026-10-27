"""
=============================================================
models.py  ─  전체 모델 아키텍처 정의
=============================================================
구현된 모델 (총 6가지):

[구조 A] Encoder-Decoder AE (병목 구조 + N 디코더 헤드)
  1. LSTMAEModel        ← 논문 제안 방법과 동일 (백본: LSTM)
  2. TransformerAEModel ← 논문 백본을 Transformer로 교체
  3. TCNAEModel         ← 논문 백본을 TCN으로 교체

[구조 B] Predictor (병목 없이 N output heads로 직접 예측)
  4. LSTMPredictorModel
  5. TransformerPredictorModel
  6. TCNPredictorModel

핵심 메커니즘 (논문 3.1절):
  - 모든 모델 공통: N개 출력 헤드 중 재구성 오류가 최소인 헤드 선택
  - 학습 및 추론 모두 동일한 헤드 선택 방식 적용
  - 선택된 헤드의 파라미터만 gradient 업데이트 (torch.gather 활용)
=============================================================
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List

from config import ModelConfig


# =============================================================
# 핵심 유틸리티: N heads 중 최적 헤드 선택 (논문 식 3)
# =============================================================
def select_best_head(
    heads_output: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    N개 헤드 출력 중 재구성 오류가 가장 작은 헤드를 선택.
    torch.gather를 사용하므로 선택된 헤드에만 gradient 흐름.

    논문 식 (3):  y_i = y_i^{n*},  n* = argmin_n ||x_i - y_i^n||

    Args:
        heads_output: (N, B, W, F) - N개 헤드의 재구성 값
        x:            (B, W, F)   - 입력값 (정답)

    Returns:
        best_output:  (B, W, F)  - 선택된 헤드의 재구성 값
        best_indices: (B, W)     - 각 (배치, 시점)에서 선택된 헤드 인덱스
        loss:         scalar     - MAE (논문 식 6)
    """
    N, B, W, F = heads_output.shape

    # 각 헤드별 MAE: (N, B, W)
    errors = torch.mean(torch.abs(heads_output - x.unsqueeze(0)), dim=-1)

    # 최소 오류 헤드 인덱스: (B, W)
    best_indices = torch.argmin(errors, dim=0)

    # 선택된 헤드 출력 gathering: (B, W, N, F) → gather → (B, W, 1, F) → squeeze
    heads_perm = heads_output.permute(1, 2, 0, 3)          # (B, W, N, F)
    idx = best_indices.unsqueeze(-1).unsqueeze(-1)          # (B, W, 1, 1)
    idx = idx.expand(B, W, 1, F)                           # (B, W, 1, F)
    best_output = torch.gather(heads_perm, dim=2, index=idx).squeeze(2)  # (B, W, F)

    # MAE loss (논문 식 6)
    loss = torch.mean(torch.abs(best_output - x))

    return best_output, best_indices, loss


# =============================================================
# Positional Encoding (Transformer용)
# =============================================================
class PositionalEncoding(nn.Module):
    """sin/cos 기반 Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================
# TCN Building Block
# =============================================================
class TCNBlock(nn.Module):
    """
    Dilated Conv1d 기반 TCN 블록.
    Same-padding으로 시퀀스 길이 유지.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # same padding: output_len = input_len 보장
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 채널 수가 다를 때 residual 연결용 1×1 conv
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


class TCNBackbone(nn.Module):
    """TCN 블록을 쌓은 백본 (dilations=[1,2,4] 기본)"""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dilations: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2 ** i for i in range(len(channels))]

        layers = []
        prev_ch = in_channels
        for ch, dil in zip(channels, dilations):
            layers.append(TCNBlock(prev_ch, ch, kernel_size, dil, dropout))
            prev_ch = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, out_channels, L)
        return self.net(x)


# =============================================================
# 공통 Base 클래스
# =============================================================
class BaseAnomalyModel(nn.Module):
    """
    모든 모델의 공통 forward 인터페이스.
    forward()는 heads_output (N, B, W, F)를 반환.
    compute_loss()는 헤드 선택 + MAE를 계산.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns: heads_output (N, B, W, F)"""
        raise NotImplementedError

    def compute_loss(self, x: torch.Tensor):
        """
        Args:
            x: (B, W, F)
        Returns:
            loss, best_output, best_indices
        """
        heads_output = self.forward(x)
        best_output, best_indices, loss = select_best_head(heads_output, x)
        return loss, best_output, best_indices


# =============================================================
# ── 구조 A: AE (Encoder-Decoder, 병목 구조) ──────────────────
# =============================================================

# ─────────────────────────
# 1. LSTM-AE (논문 제안 방법)
# ─────────────────────────
class LSTMAEModel(BaseAnomalyModel):
    """
    논문 제안 방법과 동일한 LSTM 기반 Autoencoder.
    - Encoder: LSTM → 마지막 hidden state (latent)
    - Decoder: latent를 W번 반복 → LSTM → N개 output heads
    - 학습/추론 모두 min reconstruction error head 선택
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H  = cfg.lstm_hidden
        Z  = cfg.lstm_latent
        NL = cfg.lstm_layers
        F  = cfg.input_size
        W  = cfg.window_size
        N  = cfg.n_heads
        drop = cfg.lstm_dropout if NL > 1 else 0.0

        # Encoder
        self.encoder = nn.LSTM(F, H, NL, batch_first=True, dropout=drop)

        # Latent bottleneck
        self.enc_to_latent = nn.Linear(H, Z)
        self.latent_to_dec = nn.Linear(Z, H)

        # Shared Decoder backbone
        self.decoder = nn.LSTM(H, H, NL, batch_first=True, dropout=drop)

        # N 출력 헤드 (논문 Figure 4 구조)
        self.output_heads = nn.ModuleList([nn.Linear(H, F) for _ in range(N)])

        self.window_size = W
        self.n_heads = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape

        # ── Encode ──
        _, (h_n, c_n) = self.encoder(x)          # h_n: (NL, B, H)
        latent = self.enc_to_latent(h_n[-1])      # (B, Z)

        # ── Decode ──
        dec_input = self.latent_to_dec(latent)    # (B, H)
        dec_input = dec_input.unsqueeze(1).repeat(1, W, 1)  # (B, W, H)
        dec_out, _ = self.decoder(dec_input)      # (B, W, H)

        # ── N output heads ──
        heads = torch.stack([head(dec_out) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# ─────────────────────────────
# 2. Transformer-AE
# ─────────────────────────────
class TransformerAEModel(BaseAnomalyModel):
    """
    Transformer 기반 Autoencoder.
    - Encoder: Linear projection + PositionalEncoding + TransformerEncoder → mean pool (latent)
    - Decoder: latent expand + PositionalEncoding + TransformerEncoder → N heads
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d  = cfg.d_model
        F  = cfg.input_size
        W  = cfg.window_size
        N  = cfg.n_heads

        # ── Encoder ──
        self.enc_proj = nn.Linear(F, d)
        self.enc_pos  = PositionalEncoding(d, W + 10, cfg.tf_dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff, dropout=cfg.tf_dropout,
            batch_first=True
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, cfg.n_enc_layers)

        # ── Decoder ──
        self.dec_proj = nn.Linear(d, d)  # latent → decoder input dimension
        self.dec_pos  = PositionalEncoding(d, W + 10, cfg.tf_dropout)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff, dropout=cfg.tf_dropout,
            batch_first=True
        )
        self.transformer_dec = nn.TransformerEncoder(dec_layer, cfg.n_dec_layers)

        # ── N output heads ──
        self.output_heads = nn.ModuleList([nn.Linear(d, F) for _ in range(N)])

        self.window_size = W
        self.n_heads = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape

        # ── Encode ──
        enc = self.enc_proj(x)                  # (B, W, d)
        enc = self.enc_pos(enc)
        enc = self.transformer_enc(enc)         # (B, W, d)
        latent = enc.mean(dim=1)                # (B, d) ← global average pooling

        # ── Decode ──
        dec = self.dec_proj(latent)             # (B, d)
        dec = dec.unsqueeze(1).repeat(1, W, 1) # (B, W, d)
        dec = self.dec_pos(dec)
        dec = self.transformer_dec(dec)         # (B, W, d)

        # ── N output heads ──
        heads = torch.stack([head(dec) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# ─────────────────────────────
# 3. TCN-AE
# ─────────────────────────────
class TCNAEModel(BaseAnomalyModel):
    """
    TCN 기반 Autoencoder.
    - Conv1d는 (B, C, L) 형식 사용
    - Encoder: TCN blocks → mean pool (latent)
    - Decoder: latent expand → TCN blocks → N heads
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        channels = cfg.tcn_channels
        d = channels[-1]
        F = cfg.input_size
        W = cfg.window_size
        N = cfg.n_heads

        # ── Encoder ──
        self.encoder = TCNBackbone(F, channels, cfg.tcn_kernel,
                                   cfg.tcn_dilations, cfg.tcn_dropout)

        # ── Decoder ──
        self.decoder = TCNBackbone(d, channels, cfg.tcn_kernel,
                                   cfg.tcn_dilations, cfg.tcn_dropout)

        # ── N output heads ──
        self.output_heads = nn.ModuleList([nn.Linear(d, F) for _ in range(N)])

        self.window_size = W
        self.n_heads = N
        self.latent_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape

        # ── Encode ──
        x_conv = x.permute(0, 2, 1)            # (B, F, W)
        enc = self.encoder(x_conv)              # (B, d, W)
        latent = enc.mean(dim=-1)               # (B, d) ← global average pooling

        # ── Decode ──
        dec = latent.unsqueeze(-1).repeat(1, 1, W)  # (B, d, W)
        dec = self.decoder(dec)                 # (B, d, W)
        dec = dec.permute(0, 2, 1)             # (B, W, d)

        # ── N output heads ──
        heads = torch.stack([head(dec) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# =============================================================
# ── 구조 B: Predictor (병목 없음, N output heads) ──────────────
# =============================================================

# ─────────────────────────────────────
# 4. LSTM Predictor
# ─────────────────────────────────────
class LSTMPredictorModel(BaseAnomalyModel):
    """
    LSTM 백본 + N output heads (병목 없이 직접 예측).
    입력 시퀀스를 그대로 LSTM에 통과시켜 hidden sequence를 얻고,
    N개 head가 각각 재구성 값을 예측.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H  = cfg.lstm_hidden
        NL = cfg.lstm_layers
        F  = cfg.input_size
        N  = cfg.n_heads
        drop = cfg.lstm_dropout if NL > 1 else 0.0

        self.lstm = nn.LSTM(F, H, NL, batch_first=True, dropout=drop)

        # N output heads (각 head: hidden_size → input_size)
        self.output_heads = nn.ModuleList([nn.Linear(H, F) for _ in range(N)])
        self.n_heads = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape
        out, _ = self.lstm(x)                   # (B, W, H)
        heads = torch.stack([head(out) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# ─────────────────────────────────────
# 5. Transformer Predictor
# ─────────────────────────────────────
class TransformerPredictorModel(BaseAnomalyModel):
    """
    Transformer 백본 + N output heads (병목 없이 직접 예측).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        F = cfg.input_size
        W = cfg.window_size
        N = cfg.n_heads

        self.proj = nn.Linear(F, d)
        self.pos  = PositionalEncoding(d, W + 10, cfg.tf_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff, dropout=cfg.tf_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, cfg.n_enc_layers + cfg.n_dec_layers   # 더 깊게 (AE와 파라미터 수 맞춤)
        )

        self.output_heads = nn.ModuleList([nn.Linear(d, F) for _ in range(N)])
        self.n_heads = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape
        out = self.proj(x)                      # (B, W, d)
        out = self.pos(out)
        out = self.transformer(out)             # (B, W, d)
        heads = torch.stack([head(out) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# ─────────────────────────────────────
# 6. TCN Predictor
# ─────────────────────────────────────
class TCNPredictorModel(BaseAnomalyModel):
    """
    TCN 백본 + N output heads (병목 없이 직접 예측).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        channels = cfg.tcn_channels
        d = channels[-1]
        F = cfg.input_size
        N = cfg.n_heads

        # AE보다 더 깊게 (encoder + decoder 합친 깊이)
        deep_channels = channels + channels
        deep_dilations = cfg.tcn_dilations + cfg.tcn_dilations

        self.backbone = TCNBackbone(F, deep_channels, cfg.tcn_kernel,
                                    deep_dilations, cfg.tcn_dropout)

        self.output_heads = nn.ModuleList([nn.Linear(d, F) for _ in range(N)])
        self.n_heads = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, F = x.shape
        x_conv = x.permute(0, 2, 1)            # (B, F, W)
        out = self.backbone(x_conv)             # (B, d, W)
        out = out.permute(0, 2, 1)             # (B, W, d)
        heads = torch.stack([head(out) for head in self.output_heads], dim=0)
        # (N, B, W, F)
        return heads


# =============================================================
# Model Factory
# =============================================================
MODEL_REGISTRY = {
    # AE 계열
    'lstm_ae':          LSTMAEModel,
    'transformer_ae':   TransformerAEModel,
    'tcn_ae':           TCNAEModel,
    # Predictor 계열
    'lstm_pred':        LSTMPredictorModel,
    'transformer_pred': TransformerPredictorModel,
    'tcn_pred':         TCNPredictorModel,
}


def create_model(model_type: str, cfg: ModelConfig) -> BaseAnomalyModel:
    """
    모델 타입 문자열로 모델 인스턴스 생성.

    Args:
        model_type: 'lstm_ae' | 'transformer_ae' | 'tcn_ae' |
                    'lstm_pred' | 'transformer_pred' | 'tcn_pred'
        cfg: ModelConfig
    Returns:
        모델 인스턴스
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_type](cfg)

    # 파라미터 수 출력
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_type} | 파라미터 수: {n_params:,}")
    return model


# =============================================================
# 빠른 확인용 테스트
# =============================================================
if __name__ == '__main__':
    from config import model_cfg

    B, W, F = 8, 15, 5  # batch=8, window=15, features=5
    x = torch.randn(B, W, F)

    print("=" * 55)
    for name in MODEL_REGISTRY:
        model = create_model(name, model_cfg)
        heads = model(x)                    # (N, B, W, F)
        loss, best, idx = model.compute_loss(x)
        assert heads.shape == (model_cfg.n_heads, B, W, F), f"heads shape error: {heads.shape}"
        assert best.shape  == (B, W, F),                    f"best shape error: {best.shape}"
        print(f"  {name:<22} heads={heads.shape} | loss={loss.item():.4f} ✓")
    print("=" * 55)
    print("모든 모델 forward 테스트 통과!")

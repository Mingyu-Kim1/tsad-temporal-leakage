# -*- coding: utf-8 -*-
"""
tranad_model.py — TranAD 자립형 구현
=====================================
출처: Tuli, Casale, Jennings, "TranAD: Deep Transformer Networks for Anomaly
      Detection in Multivariate Time Series Data", VLDB 2022.
      공식 저장소 github.com/imperial-qore/TranAD (BSD-3) 의 src/models.py ·
      src/dlutils.py 에서 TranAD 관련 클래스를 **그대로 옮겨** 구성.

왜 저장소를 직접 import 하지 않는가 (실측 확인된 문제 3가지)
  1) src/models.py 최상단의 `import dgl` — TranAD는 dgl을 쓰지 않으나 같은 파일의
     다른 모델(MTAD_GAT 등)이 사용. dgl 설치 실패 시 전체 import 불가.
  2) src/constants.py → src/parser.py 가 argparse를 모듈 로드시 실행하여
     본 실험 스크립트의 CLI 인자와 충돌하고, args.dataset 키가 없으면 KeyError.
  3) PositionalEncoding의 max_len이 n_window=10으로 고정 → window 20/100을 넣으면
     self.pe[0:W] 길이가 모자라 broadcasting 오류.
  → 아래 구현은 (1)(2)를 제거하고, (3)은 max_len을 실제 window로 받도록 함.
     ★단, 논문 충실성을 위해 기본 사용은 TranAD 원 설정인 window=10 하위분할 방식.

원본과 동일한 부분: 구조·활성함수·2단계 self-conditioning·adversarial 학습식
                  L = (1/n)·MSE(x1, w) + (1−1/n)·MSE(x2, w),  n = epoch
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


# ── 원본 src/dlutils.py 에서 그대로 ────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float()
                             * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        return self.dropout(x + self.pe[pos:pos + x.size(0), :])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)


# ── 원본 src/models.py 의 TranAD (lr 전역참조만 제거) ──────────────────────
class TranAD(nn.Module):
    def __init__(self, feats, n_window=10):
        super().__init__()
        self.name = "TranAD"
        self.n_feats = feats
        self.n_window = n_window
        self.n = feats * n_window
        # nhead=feats: d_model=2*feats 이므로 2*feats % feats == 0 항상 성립
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, n_window)
        # 원본은 nn.TransformerEncoder(layer, num_layers=1)를 사용하나,
        # 최신 PyTorch가 커스텀 layer에 is_causal 인자를 전달해 호환 문제가 생김.
        # num_layers=1·norm=None이면 layer를 1회 적용하는 것과 완전히 동일하므로 직접 사용.
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)                                  # Phase 1
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2                                        # Phase 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# ── 학습·추론 래퍼 ─────────────────────────────────────────────────────────
def tranad_subwindows(w_arr, sub_w):
    """
    [N, W, F] → [N*k, sub_w, F].  TranAD 원 설정(window 10)을 유지하기 위해
    동일 데이터·동일 오염을 보존한 채 하위 window로 분할. USAD와 같은 방식.
    출력은 [0,1] clip (TranAD 출력이 Sigmoid이므로 필수).
    """
    N, W, F = w_arr.shape
    k = max(W // sub_w, 1)
    x = w_arr[:, :k * sub_w, :].reshape(N * k, sub_w, F)
    return np.clip(x, 0.0, 1.0).astype(np.float32), k


def tranad_train(model, tr, va, epochs, lr, device, bs=128, patience=8, sub_w=10):
    """tr, va: [N, W, F] 원 window. 내부에서 하위 window로 분할."""
    trs, _ = tranad_subwindows(tr, sub_w)
    vas, _ = tranad_subwindows(va, sub_w) if len(va) else (trs[-5:], 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss(reduction="none")
    best, state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        n = float(ep)
        perm = np.random.permutation(len(trs))
        for i in range(0, len(trs), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            w = torch.from_numpy(trs[idx]).to(device)          # [B,sub_w,F]
            win = w.permute(1, 0, 2)                            # [sub_w,B,F]
            elem = win[-1, :, :].view(1, len(idx), -1)          # [1,B,F]
            x1, x2 = model(win, elem)
            loss = (1 / n) * mse(x1, elem).mean() + (1 - 1 / n) * mse(x2, elem).mean()
            opt.zero_grad(); loss.backward(retain_graph=True); opt.step()
        model.eval()
        with torch.no_grad():
            v = []
            for i in range(0, len(vas), bs):
                w = torch.from_numpy(vas[i:i + bs]).to(device)
                if len(w) < 1:
                    continue
                win = w.permute(1, 0, 2); elem = win[-1].view(1, len(w), -1)
                _, x2 = model(win, elem)
                v.append(mse(x2, elem).mean().item())
            v = float(np.mean(v)) if v else float("inf")
        if v < best - 1e-9:
            best, wait = v, 0
            state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if state:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def tranad_score(model, w_arr, device, bs=256, sub_w=10):
    """[N,W,F] → [N] 이상점수. 하위 window 점수를 max로 집계('any' 라벨과 정합)."""
    subs, k = tranad_subwindows(w_arr, sub_w)
    model.eval()
    mse = nn.MSELoss(reduction="none")
    out = []
    for i in range(0, len(subs), bs):
        w = torch.from_numpy(subs[i:i + bs]).to(device)
        win = w.permute(1, 0, 2); elem = win[-1].view(1, len(w), -1)
        _, x2 = model(win, elem)
        out.append(mse(x2, elem).mean(dim=2).squeeze(0).cpu().numpy())
    s = np.concatenate(out)
    return s.reshape(len(w_arr), k).max(axis=1)

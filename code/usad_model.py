# -*- coding: utf-8 -*-
"""
usad_model.py — USAD 자립형 구현
==================================
출처: Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate
      Time Series", KDD 2020.

왜 자립형인가
  이전 실행에서 run_all.py가 exp6_baselines.py의 usad_flat()을 호출했는데,
  폴더의 exp6가 구버전(sub_w 인자 없음)이라 60건 전부 실패했습니다.
  → 버전 불일치를 원천 차단하기 위해 TranAD와 동일하게 자립 모듈로 분리.

원 논문 구성
  encoder 1개 + decoder 2개, 2단계 적대적 학습
    L_AE1 = (1/n)·‖W − AE1(W)‖² + (1 − 1/n)·‖W − AE2(AE1(W))‖²
    L_AE2 = (1/n)·‖W − AE2(W)‖² − (1 − 1/n)·‖W − AE2(AE1(W))‖²
    score = α·‖W − AE1(W)‖² + β·‖W − AE2(AE1(W))‖²   (α = β = 0.5)
  n = epoch 번호 (원 논문의 스케줄 그대로)

공정성
  USAD는 원 논문에서 짧은 window(5~12)를 사용하므로, 동일 데이터·동일 오염을
  유지한 채 하위 window로 분할하고 점수는 max로 집계 (TranAD와 같은 방식).
  출력이 Sigmoid이므로 입력을 [0,1]로 clip.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn


class USAD(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        h1, h2 = max(w_size // 2, 8), max(w_size // 4, 4)
        self.enc = nn.Sequential(nn.Linear(w_size, h1), nn.ReLU(True),
                                 nn.Linear(h1, h2), nn.ReLU(True),
                                 nn.Linear(h2, z_size), nn.ReLU(True))

        def dec():
            return nn.Sequential(nn.Linear(z_size, h2), nn.ReLU(True),
                                 nn.Linear(h2, h1), nn.ReLU(True),
                                 nn.Linear(h1, w_size), nn.Sigmoid())
        self.dec1, self.dec2 = dec(), dec()

    def forward(self, w):
        z = self.enc(w)
        return self.dec1(z), self.dec2(z), self.dec2(self.enc(self.dec1(z)))


def usad_flat(w_arr, sub_w=None):
    """[N,W,F] → [N*k, sub_w*F] (sub_w 지정 시). Sigmoid 출력이므로 [0,1] clip."""
    if sub_w:
        N, W, F = w_arr.shape
        k = max(W // sub_w, 1)
        x = w_arr[:, :k * sub_w, :].reshape(N * k, sub_w, F)
    else:
        x = w_arr
    return np.clip(x.reshape(len(x), -1), 0.0, 1.0).astype(np.float32)


def usad_agg(sub_scores, n_win, sub_w, W):
    """하위 window 점수 → 원 window 점수 (max 집계, 'any' 라벨 규칙과 정합)."""
    k = max(W // sub_w, 1)
    return np.asarray(sub_scores).reshape(n_win, k).max(axis=1)


def usad_train(model, tr, va, epochs, lr, device, bs=128, patience=8):
    """tr, va: 이미 flatten된 [N, sub_w*F] 배열."""
    o1 = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec1.parameters()), lr=lr)
    o2 = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec2.parameters()), lr=lr)
    best, state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        n = float(ep)
        perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            w = torch.from_numpy(tr[idx]).to(device)
            w1, w2, w3 = model(w)
            l1 = (1 / n) * torch.mean((w - w1) ** 2) + (1 - 1 / n) * torch.mean((w - w3) ** 2)
            o1.zero_grad(); l1.backward(); o1.step()
            w1, w2, w3 = model(w)
            l2 = (1 / n) * torch.mean((w - w2) ** 2) - (1 - 1 / n) * torch.mean((w - w3) ** 2)
            o2.zero_grad(); l2.backward(); o2.step()
        model.eval()
        with torch.no_grad():
            v = []
            for i in range(0, len(va), bs):
                w = torch.from_numpy(va[i:i + bs]).to(device)
                if len(w) < 1:
                    continue
                w1, _, w3 = model(w)
                v.append((0.5 * torch.mean((w - w1) ** 2)
                          + 0.5 * torch.mean((w - w3) ** 2)).item())
            v = float(np.mean(v)) if v else float("inf")
        if v < best - 1e-9:
            best, wait = v, 0
            state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if state:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def usad_score(model, arr, device, bs=256, alpha=0.5, beta=0.5):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        w = torch.from_numpy(arr[i:i + bs]).to(device)
        w1, _, w3 = model(w)
        s = alpha * torch.mean((w - w1) ** 2, dim=1) + beta * torch.mean((w - w3) ** 2, dim=1)
        out.append(s.cpu().numpy())
    return np.concatenate(out)

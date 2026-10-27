# -*- coding: utf-8 -*-
"""
exp2_robust.py — Mode-wise Robust Adaptation (MoRA) + Label-free MMD 검증
==========================================================================
두 실험을 한 스크립트에 통합.

[A] 방법론: Mode-wise Trimmed Adaptation
    오염된 target 데이터에서도 안전하게 적응하도록, 적응 손실 자체를 강건화.
      - per-window best-head 재구성 오차를 구하고
      - '지배 head(=운전 모드)'별로 상위 q% 오차 샘플을 손실에서 제외
      - 모드별로 자르므로 소수 모드가 통째로 제거되는 것을 방지
    → 사전 필터링(프로토콜)이 아니라 학습 손실(방법론)로 편입

[B] 라벨 없는 MMD·Domain Gap 검증
    적응 전/후 latent 거리 측정: MMD(다중 bandwidth·다중 kernel), CORAL, mean-L2
    + featnorm 대조군(단순 통계 정규화로도 같은 개선이 나오는가)
    + ΔGap 과 ΔAUROC 상관분석(post-hoc 분석 전용, 선택에 사용 금지)

■ 라벨 누수 방지 규칙 (전부 코드로 강제)
  1) 하이퍼파라미터(alpha, trim_q, FAR, window)는 CLI 고정값. target 라벨로 고르지 않음.
  2) trim_q '자동' 모드도 target train 오차 분포의 robust 통계만 사용(라벨 무관).
  3) threshold는 target 정상 validation의 FAR 고정(사전 지정)으로만 산출.
  4) early stopping은 target validation 재구성 손실(라벨 무관)로만.
  5) adapt_pool='gt'는 ORACLE 상한 참조용이며 제안 방법이 아님(표기 강제).
  6) 상관분석은 post-hoc 리포트일 뿐 어떤 선택에도 사용하지 않음.

실행 예:
  python exp2_robust.py --family bench --pairs Boiler_1to2 --seeds 0 1 2
  python exp2_robust.py --family bench --datasets boiler --adapt_pool all gt --seeds 0 1 2
  python exp2_robust.py --family field --seeds 0 1 2
  python exp2_robust.py --family bench --trim_q 0.05 0.1 0.2 --methods adapter_modetrim  # 민감도
  python exp2_robust.py --analyze_only --out exp2_out
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import json
import time
import random
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sps
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)

sys.path.insert(0, ".")

from config import model_cfg, train_cfg
from models import create_model, select_best_head
from trainer import train_model
from run_cross_domain_adapter_v2 import (
    LatentAdapter, AdaptedModel, full_finetune, get_latent_dim,
)

MODEL_NAME = "transformer_ae"

# ---------------------------------------------------------------------------
# 사전 고정 하이퍼파라미터 (튜닝 금지 — 논문에 '고정' 명시)
# ---------------------------------------------------------------------------
DEFAULT_TRIM_Q = 0.10       # 강건통계 관례(10% trimming)
DEFAULT_ALPHA = 0.001       # MMD 가중치 고정
DEFAULT_FAR = (0.05, 0.01, 0.001)   # 운영 오경보율 후보(전부 보고, 선택 아님)
PRIMARY_FAR = 0.01
BW_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)   # median heuristic 배수
MMD_SUBSAMPLE = 1500

METHODS_ALL = ["baseline", "featnorm", "adapter_recon", "adapter_trim",
               "adapter_modetrim", "adapter_mmd", "adapter_modetrim_mmd",
               "full_ft", "full_ft_trim"]


def set_all_seeds(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ===========================================================================
# 데이터
# ===========================================================================

def get_bench_data(pair, window, stride, k):
    import exp1_benchmark as B
    cfg = B.BENCH_PAIRS[pair]
    S, T = B.load_pair(cfg)
    sc = B.MinMax().fit(S["train"])
    s_tr, t_tr, t_te = sc(S["train"]), sc(T["train"]), sc(T["test"])

    s_ok = S["train_label"] == 0
    s_all = B.make_win(s_tr, window, stride)
    s_keep = B.win_healthy(s_ok, window, stride)
    src_win = s_all[s_keep] if s_keep.any() else s_all

    pool = {}
    t_all = B.make_win(t_tr, window, stride)
    t_lab_w = B.win_labels(T["train_label"], window, stride)
    pool["all"] = t_all
    m_stat = B.stat_healthy_mask(t_tr, k)
    keep = B.win_healthy(m_stat, window, stride)
    pool["healthy"] = t_all[keep] if keep.any() else t_all
    keep_gt = B.win_healthy(T["train_label"] == 0, window, stride)
    pool["gt"] = t_all[keep_gt] if keep_gt.any() else t_all      # ORACLE
    masks = {"all": np.ones(len(t_all), bool), "healthy": keep, "gt": keep_gt}
    contam = {kk: (float(t_lab_w[mk].mean()) if mk.any() else float("nan"))
              for kk, mk in masks.items()}

    ev_win = B.make_win(t_te, window, 1)
    ev_lab = B.win_labels(T["test_label"], window, 1)
    return dict(src=src_win, pool=pool, contam=contam,
                eval_x=ev_win, eval_y=ev_lab, n_feat=s_tr.shape[1])


def get_field_data(pair, window, stride, k, persistence, out):
    import run_experiment1 as R
    import exp1_labeling as L
    cfg = R.DOMAIN_PAIRS[pair]
    src, tgt, n_feat = R.load_pair(cfg)
    if cfg["kind"] == "pipe":
        res = L.pipe_labels(len(tgt), L.PIPE_FAULT_IDX_DEFAULT)
        healthy = [(0, res["onset_index"])]
    else:
        res = L.bearing_pseudo_labels(cfg["tgt"], k=k, persistence=persistence)
        healthy = [(m["start"],
                    m["onset_global"] if m["onset_global"] is not None
                    else m["start"] + m["length"]) for m in res["segment_meta"]]
    pl = res["labels"]
    wl = L.point_to_window_labels(pl, window, rule="any")

    def win(x, s=stride):
        n = (len(x) - window) // s + 1
        idx = np.arange(window)[None, :] + s * np.arange(n)[:, None]
        return x[idx]

    src_win = win(src)
    pool = {"all": win(tgt)}
    hs = [win(tgt[a:b]) for a, b in healthy if b - a >= window + 5]
    pool["healthy"] = np.concatenate(hs) if hs else pool["all"]
    lab_all = np.array([pl[i:i + window].max() for i in
                        range(0, len(tgt) - window + 1, stride)])
    contam = {"all": float(lab_all.mean()), "healthy": 0.0}

    ev_x = win(tgt, 1)
    ev_y = wl[:len(ev_x)]
    return dict(src=src_win, pool=pool, contam=contam,
                eval_x=ev_x, eval_y=ev_y, n_feat=n_feat)


# ===========================================================================
# 손실: mode-wise trimmed reconstruction
# ===========================================================================

def sample_errors(heads, x):
    """(N,B,W,F),(B,W,F) → per-window 오차(B,), 지배 head(B,)"""
    err = (heads - x.unsqueeze(0)).abs().mean(-1)        # (N,B,W)
    best = err.min(dim=0).values                         # (B,W)  gradient→선택 head
    idx = err.argmin(dim=0)                              # (B,W)
    per = best.mean(dim=1)                               # (B,)
    onehot = torch.nn.functional.one_hot(idx, heads.shape[0]).sum(1)  # (B,N)
    return per, onehot.argmax(1)


def trimmed_mean(per, mode, q, mode_wise):
    if q <= 0:
        return per.mean()
    if not mode_wise:
        kk = max(int(len(per) * (1 - q)), 1)
        return torch.topk(per, kk, largest=False).values.mean()
    tot, cnt = per.new_zeros(()), 0
    for m in torch.unique(mode):
        v = per[mode == m]
        kk = max(int(len(v) * (1 - q)), 1)
        tot = tot + torch.topk(v, kk, largest=False).values.sum()
        cnt += kk
    return tot / max(cnt, 1)


@torch.no_grad()
def auto_trim_q(model, w_arr, bs, device, k=3.0, cap=0.30):
    """label-free 오염률 추정: source 모델 오차 분포의 robust 상한 초과 비율."""
    e = []
    model.eval()
    for i in range(0, len(w_arr), bs):
        x = torch.FloatTensor(w_arr[i:i + bs]).to(device)
        per, _ = sample_errors(model(x), x)
        e.append(per.cpu().numpy())
    e = np.concatenate(e)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) * 1.4826 + 1e-12
    return float(min((e > med + k * mad).mean(), cap))


def train_adapter_robust(adapted, tr_w, va_w, bs, epochs, lr, device,
                         trim_q=0.0, mode_wise=False, alpha=0.0,
                         src_lat=None, patience=8):
    params = [p for p in adapted.adapter.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state, wait = float("inf"), None, 0
    n = len(tr_w)
    for ep in range(1, epochs + 1):
        adapted.train()
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            x = torch.FloatTensor(tr_w[perm[i:i + bs]]).to(device)
            if len(x) < 2:
                continue
            per, mode = sample_errors(adapted(x), x)
            loss = trimmed_mean(per, mode, trim_q, mode_wise)
            if alpha > 0 and src_lat is not None:
                z = adapted.adapter(adapted.encode_to_latent(x))
                j = torch.randint(0, len(src_lat), (min(len(z), len(src_lat)),))
                loss = loss + alpha * mmd_rbf_torch(src_lat[j].to(device), z)
            opt.zero_grad(); loss.backward(); opt.step()
        # early stopping: target val 재구성 손실 (라벨 무관, 동일 trim 기준)
        adapted.eval()
        with torch.no_grad():
            vs = []
            for i in range(0, len(va_w), bs):
                x = torch.FloatTensor(va_w[i:i + bs]).to(device)
                per, mode = sample_errors(adapted(x), x)
                vs.append(trimmed_mean(per, mode, trim_q, mode_wise).item())
            v = float(np.mean(vs)) if vs else float("inf")
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = copy.deepcopy(adapted.adapter.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        adapted.adapter.load_state_dict(best_state)
    return adapted


# ===========================================================================
# Domain gap 측정 (라벨 불필요)
# ===========================================================================

def mmd_rbf_torch(X, Y, bw=None):
    Z = torch.cat([X, Y], 0)
    if bw is None:
        with torch.no_grad():
            d = torch.cdist(Z[:512], Z[:512])
            bw = torch.median(d[d > 0]).clamp(min=1e-6)
    k = lambda a, b: torch.exp(-torch.cdist(a, b) ** 2 / (2 * bw ** 2))
    return k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean()


def gap_metrics(S: np.ndarray, T: np.ndarray) -> dict:
    """MMD(다중 bandwidth·선형kernel) + CORAL + mean-L2. 전부 라벨 불필요."""
    rng = np.random.default_rng(0)
    S = S[rng.choice(len(S), min(len(S), MMD_SUBSAMPLE), replace=False)]
    T = T[rng.choice(len(T), min(len(T), MMD_SUBSAMPLE), replace=False)]
    s, t = torch.tensor(S), torch.tensor(T)
    Z = torch.cat([s, t], 0)
    d = torch.cdist(Z, Z)
    med = torch.median(d[d > 0]).clamp(min=1e-6)
    out = {}
    for m in BW_MULTS:
        out[f"mmd_rbf_x{m}"] = float(mmd_rbf_torch(s, t, med * m))
    out["mmd_rbf_median"] = out["mmd_rbf_x1.0"]
    out["mmd_linear"] = float(((s.mean(0) - t.mean(0)) ** 2).sum())
    out["mean_l2"] = float(torch.norm(s.mean(0) - t.mean(0)))
    cs = torch.cov(s.T); ct = torch.cov(t.T)
    out["coral"] = float(((cs - ct) ** 2).sum() / (4 * s.shape[1] ** 2))
    return out


@torch.no_grad()
def latents(model, w_arr, bs, device, apply_adapter=True):
    model.eval()
    out = []
    for i in range(0, len(w_arr), bs):
        x = torch.FloatTensor(w_arr[i:i + bs]).to(device)
        if hasattr(model, "encode_to_latent"):
            z = model.encode_to_latent(x)                # frozen encoder 출력
            if apply_adapter and hasattr(model, "adapter"):
                z = model.adapter(z)                     # ★ adapter 적용 필수
        else:                                            # full_ft / baseline
            m = model
            enc = m.enc_proj(x); enc = m.enc_pos(enc)
            z = m.transformer_enc(enc).mean(1)
        out.append(z.cpu().numpy())
    return np.concatenate(out)


class FeatNorm(nn.Module):
    """대조군: 학습 없는 통계 정규화 (target→source 평균·분산 정합)."""
    def __init__(self, mu_s, sd_s, mu_t, sd_t):
        super().__init__()
        self.register_buffer("mu_s", mu_s); self.register_buffer("sd_s", sd_s)
        self.register_buffer("mu_t", mu_t); self.register_buffer("sd_t", sd_t)

    def forward(self, z):
        return (z - self.mu_t) / (self.sd_t + 1e-6) * (self.sd_s + 1e-6) + self.mu_s


# ===========================================================================
# 평가 (threshold는 전부 label-free)
# ===========================================================================

@torch.no_grad()
def scores(model, w_arr, bs, device):
    model.eval()
    o = []
    for i in range(0, len(w_arr), bs):
        x = torch.FloatTensor(w_arr[i:i + bs]).to(device)
        b, _, _ = select_best_head(model(x), x)
        o.append((b - x).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(o)


def evaluate_all(te_s, te_y, va_s):
    r = {"anomaly_ratio": float(np.mean(te_y))}
    if len(np.unique(te_y)) > 1:
        r["auroc"] = roc_auc_score(te_y, te_s)
        r["aupr"] = average_precision_score(te_y, te_s)
        r["aupr_lift"] = r["aupr"] / max(r["anomaly_ratio"], 1e-9)
    else:
        r["auroc"] = r["aupr"] = r["aupr_lift"] = np.nan
    for far in DEFAULT_FAR:
        thr = float(np.quantile(va_s, 1 - far))          # label-free
        p = (te_s >= thr).astype(int)
        tag = f"far{far:g}"
        r[f"f1_{tag}"] = f1_score(te_y, p, zero_division=0)
        r[f"prec_{tag}"] = precision_score(te_y, p, zero_division=0)
        r[f"rec_{tag}"] = recall_score(te_y, p, zero_division=0)
        r[f"thr_{tag}"] = thr
    r["f1_primary"] = r[f"f1_far{PRIMARY_FAR:g}"]
    return r


# ===========================================================================
# 1회 실행
# ===========================================================================

def run_once(pair, family, seed, pool_name, method, trim_q, args, device, out, D, cache):
    set_all_seeds(seed)
    bs = train_cfg.batch_size
    pool = D["pool"][pool_name]
    cut = max(int(len(pool) * 0.8), 1)
    tr_w, va_w = pool[:cut], (pool[cut:] if len(pool) - cut >= 5 else pool[-5:])

    model = cache["model"]
    latent_dim = get_latent_dim(MODEL_NAME)
    src_lat_t = torch.tensor(cache["src_lat"])

    q = trim_q
    if args.trim_mode == "auto" and "trim" in method:
        q = auto_trim_q(model, tr_w, bs, device, k=args.k)

    t0 = time.perf_counter()
    if method == "baseline":
        adapted = model
    elif method == "featnorm":
        base = copy.deepcopy(model)
        for p in base.parameters():
            p.requires_grad = False
        zt = latents(AdaptedModel(base, nn.Identity(), MODEL_NAME).to(device),
                     tr_w, bs, device, apply_adapter=False)
        fn = FeatNorm(torch.tensor(cache["src_lat"].mean(0)),
                      torch.tensor(cache["src_lat"].std(0)),
                      torch.tensor(zt.mean(0)), torch.tensor(zt.std(0))).to(device)
        adapted = AdaptedModel(base, fn, MODEL_NAME).to(device)
    elif method.startswith("adapter"):
        base = copy.deepcopy(model)
        for p in base.parameters():
            p.requires_grad = False
        adapted = AdaptedModel(base, LatentAdapter(latent_dim).to(device),
                               MODEL_NAME).to(device)
        train_adapter_robust(
            adapted, tr_w, va_w, bs, args.epochs_adapt, args.adapt_lr, device,
            trim_q=(q if "trim" in method else 0.0),
            mode_wise=("modetrim" in method),
            alpha=(args.alpha if method.endswith("mmd") else 0.0),
            src_lat=src_lat_t)
    elif method.startswith("full_ft"):
        adapted = copy.deepcopy(model)
        if method == "full_ft":
            full_finetune(adapted,
                          torch.utils.data.DataLoader(torch.FloatTensor(tr_w),
                                                      batch_size=bs, shuffle=True),
                          torch.utils.data.DataLoader(torch.FloatTensor(va_w),
                                                      batch_size=bs, shuffle=False),
                          args.epochs_adapt, args.ft_lr, device)
        else:   # full_ft_trim — 강건화가 adapter 고유 효과인지 분리하는 ablation
            opt = torch.optim.Adam(adapted.parameters(), lr=args.ft_lr)
            for _ in range(args.epochs_adapt):
                adapted.train()
                perm = np.random.permutation(len(tr_w))
                for i in range(0, len(tr_w), bs):
                    x = torch.FloatTensor(tr_w[perm[i:i + bs]]).to(device)
                    if len(x) < 2:
                        continue
                    per, mode = sample_errors(adapted(x), x)
                    loss = trimmed_mean(per, mode, q, True)
                    opt.zero_grad(); loss.backward(); opt.step()
    else:
        raise ValueError(method)
    adapt_time = 0.0 if method == "baseline" else time.perf_counter() - t0

    # ── 성능
    te_s = scores(adapted, D["eval_x"], bs, device)
    va_s = scores(adapted, va_w, bs, device)
    met = evaluate_all(te_s, D["eval_y"], va_s)

    # ── domain gap (라벨 불필요): source 기준 latent와의 거리
    tgt_lat = latents(adapted, tr_w, bs, device, apply_adapter=True)
    gaps = gap_metrics(cache["src_lat"], tgt_lat)
    gaps_before = cache["gap_before"]

    tot = sum(p.numel() for p in adapted.parameters())
    tra = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    if method == "baseline" or method == "featnorm":
        tra = 0
    rec = {"family": family, "pair": pair, "adapt_pool": pool_name,
           "method": method, "seed": seed, "trim_q": q,
           "contam": D["contam"].get(pool_name, np.nan),
           "is_oracle": pool_name == "gt",
           "trainable_params": tra, "trainable_pct": 100.0 * tra / max(tot, 1),
           "adapt_time_s": adapt_time, **met,
           **{f"gap_{k}": v for k, v in gaps.items()},
           **{f"gap0_{k}": v for k, v in gaps_before.items()},
           "d_mmd": gaps_before["mmd_rbf_median"] - gaps["mmd_rbf_median"],
           "d_coral": gaps_before["coral"] - gaps["coral"],
           "d_mean_l2": gaps_before["mean_l2"] - gaps["mean_l2"]}
    np.save(out / "scores" / f"{pair}__{pool_name}__{method}__q{q:.2f}__s{seed}.npy", te_s)
    print(f"      [{method:20s} pool={pool_name:7s} q={q:.2f}] "
          f"AUROC={met['auroc']:.4f} AUPR={met['aupr']:.4f} "
          f"F1@1%={met['f1_far0.01']:.4f} ΔMMD={rec['d_mmd']:+.4f}")
    return rec


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["bench", "field"], default="bench")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=["boiler"],
                    choices=["smd", "boiler"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="*", default=[
        "baseline", "featnorm", "adapter_recon", "adapter_modetrim",
        "adapter_mmd", "full_ft"])
    ap.add_argument("--adapt_pool", nargs="*", default=["all"],
                    choices=["all", "healthy", "gt"])
    ap.add_argument("--trim_q", nargs="*", type=float, default=[DEFAULT_TRIM_Q])
    ap.add_argument("--trim_mode", choices=["fixed", "auto"], default="fixed")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--stride_train", type=int, default=5)
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--epochs_pretrain", type=int, default=60)
    ap.add_argument("--epochs_adapt", type=int, default=30)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp2_out")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    device = train_cfg.device

    if args.pairs:
        pairs = args.pairs
    elif args.family == "bench":
        import exp1_benchmark as B
        pairs = [k for k, v in B.BENCH_PAIRS.items() if v["kind"] in args.datasets]
    else:
        import run_experiment1 as R
        pairs = list(R.DOMAIN_PAIRS)

    print(f"[exp2] family={args.family} pairs={pairs} methods={args.methods} "
          f"pool={args.adapt_pool} trim_q={args.trim_q}({args.trim_mode}) "
          f"alpha={args.alpha}(고정) window={args.window} FAR={DEFAULT_FAR}")

    if not args.analyze_only:
        rows = []
        p_prev = out / "raw_results.csv"
        if p_prev.exists():
            prev = pd.read_csv(p_prev)
            m = ~(prev["pair"].isin(pairs) & prev["method"].isin(args.methods)
                  & prev["seed"].isin(args.seeds)
                  & prev["adapt_pool"].isin(args.adapt_pool))
            rows = prev[m].to_dict("records")

        for pair in pairs:
            print(f"\n{'━'*66}\n  {pair}\n{'━'*66}")
            D = (get_bench_data(pair, args.window, args.stride_train, args.k)
                 if args.family == "bench" else
                 get_field_data(pair, args.window, args.stride_train,
                                args.k, args.persistence, out))
            print(f"  src={len(D['src'])}w  pools=" +
                  ", ".join(f"{k}:{len(v)}w(오염 {D['contam'].get(k, float('nan')):.3f})"
                            for k, v in D["pool"].items() if k in args.adapt_pool) +
                  f"  eval={len(D['eval_x'])}w(이상 {D['eval_y'].mean():.3f})")

            for seed in args.seeds:
                set_all_seeds(seed)
                model_cfg.n_heads = 5
                model_cfg.input_size = D["n_feat"]
                model_cfg.window_size = args.window
                model_cfg.d_model = 128
                model_cfg.dim_ff = 512
                ck = out / "ckpt" / f"{pair}__s{seed}__w{args.window}.pt"
                if ck.exists() and not args.force_pretrain:
                    model = create_model(MODEL_NAME, model_cfg)
                    model.load_state_dict(torch.load(ck, map_location=device))
                    model = model.to(device)
                else:
                    n = int(len(D["src"]) * 0.8)
                    train_cfg.seed = seed; train_cfg.epochs = args.epochs_pretrain
                    model, _, _ = train_model(
                        MODEL_NAME, model_cfg, train_cfg,
                        torch.utils.data.DataLoader(torch.FloatTensor(D["src"][:n]),
                                                    batch_size=train_cfg.batch_size,
                                                    shuffle=True),
                        torch.utils.data.DataLoader(torch.FloatTensor(D["src"][n:]),
                                                    batch_size=train_cfg.batch_size,
                                                    shuffle=False), verbose=False)
                    torch.save(model.state_dict(), ck)
                    model = model.to(device)

                src_lat = latents(model, D["src"], train_cfg.batch_size, device)
                cache = {"model": model, "src_lat": src_lat}
                print(f"  ── seed {seed}")
                for pool_name in args.adapt_pool:
                    pl = D["pool"][pool_name]
                    cut = max(int(len(pl) * 0.8), 1)
                    tgt0 = latents(model, pl[:cut], train_cfg.batch_size, device)
                    cache["gap_before"] = gap_metrics(src_lat, tgt0)
                    for method in args.methods:
                        qs = args.trim_q if "trim" in method else [0.0]
                        for q in qs:
                            rows.append(run_once(pair, args.family, seed, pool_name,
                                                 method, q, args, device, out, D, cache))
                            pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)

    analyze(out)


# ===========================================================================
def analyze(out: Path):
    out = Path(out)
    df = pd.read_csv(out / "raw_results.csv")
    key = ["pair", "adapt_pool", "method", "trim_q"]
    show = ["auroc", "aupr", "aupr_lift", "f1_far0.01", "d_mmd", "d_coral"]

    rows = []
    for k, g in df.groupby(key):
        r = dict(zip(key, k)); r["n"] = len(g); r["oracle"] = bool(g["is_oracle"].iloc[0])
        for c in show + ["trainable_pct", "adapt_time_s", "contam"]:
            if c in g:
                v = g[c].astype(float).dropna()
                r[c] = (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1
                        else (f"{v.mean():.4f}" if len(v) else "nan"))
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(key)
    summ.to_csv(out / "summary.csv", index=False)

    # 통계: t-test + Wilcoxon + Holm 보정
    st = []
    for (pair, pool), g in df.groupby(["pair", "adapt_pool"]):
        piv = g.pivot_table(index="seed", columns="method", values="auroc")
        if "baseline" not in piv:
            continue
        for m in [c for c in piv.columns if c != "baseline"]:
            a, b = piv[m].dropna(), piv["baseline"].dropna()
            i = a.index.intersection(b.index)
            if len(i) < 2:
                continue
            t, pt = sps.ttest_rel(a[i], b[i])
            try:
                _, pw = sps.wilcoxon(a[i], b[i])
            except Exception:
                pw = np.nan
            st.append({"pair": pair, "adapt_pool": pool, "method": m,
                       "diff": float((a[i] - b[i]).mean()), "t": float(t),
                       "p_ttest": float(pt), "p_wilcoxon": float(pw), "n": len(i)})
    if st:
        s = pd.DataFrame(st).sort_values("p_ttest")
        m = len(s)
        s["p_holm"] = np.minimum.accumulate(
            (s["p_ttest"].values * (m - np.arange(m)))[::-1])[::-1].clip(max=1.0)
        s["sig_holm"] = s["p_holm"] < 0.05
        s.to_csv(out / "stats_vs_baseline.csv", index=False)

    # ΔGap ↔ ΔAUROC 상관 (post-hoc 분석 전용)
    base = df[df.method == "baseline"].groupby(["pair", "adapt_pool", "seed"])["auroc"].mean()
    cor = []
    for gapc in ["d_mmd", "d_coral", "d_mean_l2"]:
        sub = df[df.method != "baseline"].copy()
        sub["d_auroc"] = sub.apply(
            lambda r: r["auroc"] - base.get((r["pair"], r["adapt_pool"], r["seed"]), np.nan),
            axis=1)
        s2 = sub[[gapc, "d_auroc"]].dropna()
        if len(s2) > 3:
            pr, pp = sps.pearsonr(s2[gapc], s2["d_auroc"])
            sr, sp = sps.spearmanr(s2[gapc], s2["d_auroc"])
            cor.append({"gap_metric": gapc, "n": len(s2), "pearson_r": pr,
                        "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})
        sub.to_csv(out / "with_delta_auroc.csv", index=False)
    if cor:
        pd.DataFrame(cor).to_csv(out / "gap_auroc_correlation.csv", index=False)

    # bandwidth 민감도
    bw = [c for c in df.columns if c.startswith("gap_mmd_rbf_x")]
    if bw:
        b0 = [c.replace("gap_", "gap0_") for c in bw]
        r = []
        for m, g in df.groupby("method"):
            for c, c0 in zip(bw, b0):
                if c0 in g:
                    r.append({"method": m, "bandwidth": c.split("_x")[-1],
                              "mmd_before": g[c0].mean(), "mmd_after": g[c].mean(),
                              "reduction": g[c0].mean() - g[c].mean()})
        pd.DataFrame(r).to_csv(out / "mmd_bandwidth_sensitivity.csv", index=False)

    print("\n=== summary ===")
    cols = [c for c in ["pair", "adapt_pool", "method", "trim_q", "oracle",
                        "auroc", "aupr_lift", "f1_far0.01", "d_mmd"] if c in summ]
    print(summ[cols].to_string(index=False))
    if (out / "gap_auroc_correlation.csv").exists():
        print("\n=== Δgap ↔ ΔAUROC 상관 (post-hoc) ===")
        print(pd.read_csv(out / "gap_auroc_correlation.csv").to_string(index=False))
    if (out / "stats_vs_baseline.csv").exists():
        print("\n=== vs Baseline (Holm 보정) ===")
        print(pd.read_csv(out / "stats_vs_baseline.csv")[
            ["pair", "adapt_pool", "method", "diff", "p_ttest",
             "p_wilcoxon", "p_holm", "sig_holm"]].to_string(index=False))
    print(f"\n[exp2] 완료 → {out.resolve()}")


if __name__ == "__main__":
    main()

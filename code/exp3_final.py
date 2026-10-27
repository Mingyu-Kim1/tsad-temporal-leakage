# -*- coding: utf-8 -*-
"""
exp3_final.py — 최종 확정 실험 (Boiler + SMD, seed 3회)
=========================================================
4개 축을 한 스크립트에서. 결과가 어느 쪽으로 나와도 논문이 되도록 설계.

  [Stage 1] inject   — 오염 통제 주입 dose–response (0/5/10/20/40%)
                       "적응 이득이 오염에 어떻게 무너지는가"를 인과적으로 특성화
  [Stage 2] isca     — 반복 자가정제 적응 vs 1회 트리밍 vs oracle
                       breakdown point(50%)를 넘을 수 있는가
  [Stage 3] align    — 정렬 4종: none / global MMD / whitened MMD / mode-conditional MMD
                       "latent 정렬을 어떻게 해야 하는가"에 대한 답
  [Stage 4] capacity — 재보정 용량 사다리: shift → diag affine → bottleneck → linear → full FT
                       "adapter가 단순 scale/shift 보정 아닌가"에 대한 답

■ 라벨 누수 방지 (코드로 강제)
  - 방법(적응/정제/threshold/early stopping)은 target 라벨을 일절 사용하지 않음
  - GT 라벨은 (a) 실험 조건 구성(오염률 주입), (b) 평가, (c) 진단 기록에만 사용.
    → 결과 CSV의 uses_gt_for_condition 열로 명시
  - ★ SMD는 train에 이상이 없어 실제 이상을 test에서 가져오면 평가셋 누수 →
    오염 주입은 '합성(synthetic)'으로 통일. Boiler는 real/synth 둘 다 지원해
    "합성 결론이 실제 이상에서도 성립하는가"를 교차 검증.
  - 하이퍼파라미터 전부 사전 고정: alpha=0.001, trim q, ISCA keep 스케줄,
    window(bench=100), FAR=1%. 성능 보고 고르지 않음.

실행 (anomaly_detection/ 에서):
  python exp3_final.py --stage capacity --seeds 0            # 파일럿 (빠름)
  python exp3_final.py --stage align    --seeds 0 1 2
  python exp3_final.py --stage isca     --seeds 0 1 2
  python exp3_final.py --stage inject   --seeds 0 1 2        # 가장 오래 걸림
  python exp3_final.py --analyze_only
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sps

sys.path.insert(0, ".")

from config import model_cfg, train_cfg
from models import create_model, select_best_head
from trainer import train_model
from run_cross_domain_adapter_v2 import (
    LatentAdapter, AdaptedModel, full_finetune, get_latent_dim,
)
import exp1_benchmark as B
from exp2_robust import (sample_errors, trimmed_mean, scores, evaluate_all,
                         gap_metrics, latents, FeatNorm, mmd_rbf_torch)

MODEL_NAME = "transformer_ae"

# ── 사전 고정 하이퍼파라미터 (튜닝 금지) ───────────────────────────────────
ALPHA = 0.001                       # 정렬 항 가중치
TRIM_Q = 0.50                       # 1회 트리밍 (breakdown point 상한)
ISCA_KEEP = (1.0, 0.7, 0.5)         # 반복 자가정제 유지 비율 스케줄
CONTAM_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.40)
MIN_MODE_N = 32                     # mode-conditional MMD 최소 표본
WINDOW = 100                        # 벤치마크 프로토콜(DACAD) 고정
STRIDE = 5

PAIRS = {"boiler": ["Boiler_1to2", "Boiler_2to3", "Boiler_3to1"],
         "smd": ["SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2"]}


def set_all_seeds(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ===========================================================================
# 용량 사다리 (전부 zero-init residual → 시작 시 항등)
# ===========================================================================

class ShiftAdapter(nn.Module):                       # d params
    def __init__(self, d):
        super().__init__(); self.b = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        return z + self.b


class DiagAffineAdapter(nn.Module):                  # 2d params
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(d)); self.b = nn.Parameter(torch.zeros(d))

    def forward(self, z):
        return z + self.a * z + self.b


class LinearAdapter(nn.Module):                      # d^2 + d params
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, z):
        return z + self.lin(z)


CAPACITY = {"shift": ShiftAdapter, "diag": DiagAffineAdapter,
            "bottleneck": LatentAdapter, "linear": LinearAdapter}


# ===========================================================================
# 정렬 손실 4종
# ===========================================================================

def _whiten(z, eps=1e-5):
    return (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + eps)


def align_loss(kind, z_t, mode_t, src_lat, src_mode, device):
    """kind: none | mmd | mmd_white | mmd_mode"""
    if kind == "none":
        return z_t.new_zeros(())
    n = min(len(z_t), 256)
    j = torch.randint(0, len(src_lat), (n,))
    zs = src_lat[j].to(device)
    if kind == "mmd":
        return mmd_rbf_torch(zs, z_t[:n])
    if kind == "mmd_white":
        return mmd_rbf_torch(_whiten(zs), _whiten(z_t[:n]))
    if kind == "mmd_mode":
        # 모드별로 자기 도메인 통계로 표준화 후 정렬 (head는 frozen·공유 → 대응 성립)
        tot, w = z_t.new_zeros(()), 0
        for k in torch.unique(mode_t):
            mt = mode_t == k
            ms = (src_mode == int(k))
            if int(mt.sum()) < MIN_MODE_N or int(ms.sum()) < MIN_MODE_N:
                continue
            a = _whiten(z_t[mt])
            idx = torch.nonzero(torch.as_tensor(ms)).squeeze(-1)
            idx = idx[torch.randperm(len(idx))[:min(len(idx), 256)]]
            b = _whiten(src_lat[idx].to(device))
            tot = tot + mmd_rbf_torch(b, a) * int(mt.sum()); w += int(mt.sum())
        return tot / max(w, 1)
    raise ValueError(kind)


@torch.no_grad()
def source_latents_modes(model, w_arr, bs, device):
    """frozen source 모델의 latent와 지배 head(모드) — 정렬 대응용."""
    model.eval()
    L, M = [], []
    for i in range(0, len(w_arr), bs):
        x = torch.FloatTensor(w_arr[i:i + bs]).to(device)
        enc = model.enc_proj(x); enc = model.enc_pos(enc)
        L.append(model.transformer_enc(enc).mean(1).cpu())
        _, m = sample_errors(model(x), x)
        M.append(m.cpu())
    return torch.cat(L), torch.cat(M).numpy()


# ===========================================================================
# 적응 학습 (트리밍 + 정렬 통합)
# ===========================================================================

def train_adapter(adapted, tr_w, va_w, bs, epochs, lr, device,
                  trim_q=0.0, align="none", alpha=ALPHA,
                  src_lat=None, src_mode=None, patience=8):
    opt = torch.optim.Adam([p for p in adapted.adapter.parameters()
                            if p.requires_grad], lr=lr)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        adapted.train()
        perm = np.random.permutation(len(tr_w))
        for i in range(0, len(tr_w), bs):
            x = torch.FloatTensor(tr_w[perm[i:i + bs]]).to(device)
            if len(x) < 2:
                continue
            per, mode = sample_errors(adapted(x), x)
            loss = trimmed_mean(per, mode, trim_q, True)
            if align != "none":
                z = adapted.adapter(adapted.encode_to_latent(x))
                loss = loss + alpha * align_loss(align, z, mode, src_lat,
                                                 src_mode, device)
            opt.zero_grad(); loss.backward(); opt.step()
        adapted.eval()
        with torch.no_grad():
            v = []
            for i in range(0, len(va_w), bs):
                x = torch.FloatTensor(va_w[i:i + bs]).to(device)
                per, mode = sample_errors(adapted(x), x)
                v.append(trimmed_mean(per, mode, trim_q, True).item())
            v = float(np.mean(v)) if v else float("inf")
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


def fresh_adapted(model, cap, latent_dim, device):
    base = copy.deepcopy(model)
    for p in base.parameters():
        p.requires_grad = False
    return AdaptedModel(base, CAPACITY[cap](latent_dim).to(device),
                        MODEL_NAME).to(device)


# ===========================================================================
# ISCA — 반복 자가정제 적응 (라벨 미사용)
# ===========================================================================

def run_isca(model, pool, pool_gt, bs, args, device, latent_dim, cap="bottleneck",
             align="none", src_lat=None, src_mode=None):
    """
    pool     : 적응 후보 window (오염 포함)
    pool_gt  : 진단 전용 GT 라벨(방법에 사용하지 않음)
    keep 스케줄에 따라 자기 점수로 정제 → frozen source에서 매번 새로 적응
    """
    cur = pool
    diag = []
    adapted = None
    for it, keep in enumerate(ISCA_KEEP):
        if keep < 1.0 and adapted is not None:
            s = scores(adapted, pool, bs, device)          # 원본 pool 전체 채점
            k = max(int(len(pool) * keep), 10)
            cur = pool[np.argsort(s)[:k]]
            sel = np.argsort(s)[:k]
            diag.append({"iter": it, "keep": keep, "n": k,
                         "contam_selected": float(pool_gt[sel].mean())})
        cut = max(int(len(cur) * 0.8), 1)
        adapted = fresh_adapted(model, cap, latent_dim, device)
        train_adapter(adapted, cur[:cut], cur[cut:] if len(cur) - cut >= 5 else cur[-5:],
                      bs, args.epochs_adapt, args.adapt_lr, device,
                      trim_q=0.0, align=align, src_lat=src_lat, src_mode=src_mode)
    return adapted, diag


# ===========================================================================
# 데이터 + 오염 주입
# ===========================================================================

def synth_contaminate(w, rng):
    """합성 오염: window 단위로 spike / level shift / scale 변화 중 하나 적용."""
    out = w.copy()
    for i in range(len(out)):
        W, F = out[i].shape
        ch = rng.choice(F, max(1, F // 4), replace=False)
        t = rng.integers(0, 3)
        if t == 0:                                    # spike
            s = rng.integers(0, max(W - 5, 1))
            out[i][s:s + 5, ch] += rng.normal(0.6, 0.1, (min(5, W - s), len(ch)))
        elif t == 1:                                  # level shift
            s = rng.integers(0, max(W // 2, 1))
            out[i][s:, ch] += rng.normal(0.35, 0.05)
        else:                                         # scale/variance 변화
            out[i][:, ch] *= rng.uniform(1.5, 2.2)
    return out


def prepare(pair, args, rng):
    cfg = B.BENCH_PAIRS[pair]
    S, T = B.load_pair(cfg)
    sc = B.MinMax().fit(S["train"])
    s_tr, t_tr, t_te = sc(S["train"]), sc(T["train"]), sc(T["test"])

    src_win = B.make_win(s_tr, WINDOW, STRIDE)
    s_keep = B.win_healthy(S["train_label"] == 0, WINDOW, STRIDE)
    if s_keep.any():
        src_win = src_win[s_keep]

    t_all = B.make_win(t_tr, WINDOW, STRIDE)
    t_lab = B.win_labels(T["train_label"], WINDOW, STRIDE)
    clean = t_all[t_lab == 0]
    anom = t_all[t_lab == 1]

    ev_x = B.make_win(t_te, WINDOW, 1)
    ev_y = B.win_labels(T["test_label"], WINDOW, 1)
    return dict(src=src_win, clean=clean, anom=anom, nat_pool=t_all,
                nat_gt=t_lab, eval_x=ev_x, eval_y=ev_y, n_feat=s_tr.shape[1])


def make_pool(D, frac, mode, rng):
    """오염률 frac의 적응 pool 구성. mode: synth | real"""
    clean = D["clean"]
    N = len(clean)
    n_a = int(N * frac)
    if n_a == 0:
        pool = clean.copy()
        gt = np.zeros(len(pool), int)
    else:
        n_c = N - n_a
        idx = rng.permutation(N)
        base = clean[idx[:n_c]]
        if mode == "real" and len(D["anom"]) > 0:
            j = rng.choice(len(D["anom"]), n_a, replace=len(D["anom"]) < n_a)
            add = D["anom"][j]
        else:
            j = rng.choice(N, n_a, replace=False)
            add = synth_contaminate(clean[j], rng)
        pool = np.concatenate([base, add])
        gt = np.concatenate([np.zeros(len(base), int), np.ones(len(add), int)])
        p = rng.permutation(len(pool))
        pool, gt = pool[p], gt[p]
    return pool, gt


# ===========================================================================
# 한 설정 실행
# ===========================================================================

def run_cfg(pair, seed, stage, cfgname, spec, pool, pool_gt, D, model,
            src_lat, src_mode, args, device, out):
    bs = train_cfg.batch_size
    latent_dim = get_latent_dim(MODEL_NAME)
    cut = max(int(len(pool) * 0.8), 1)
    tr_w = pool[:cut]
    va_w = pool[cut:] if len(pool) - cut >= 5 else pool[-5:]
    diag = []

    t0 = time.perf_counter()
    if spec["kind"] == "baseline":
        adapted = model
    elif spec["kind"] == "featnorm":
        base = copy.deepcopy(model)
        for p in base.parameters():
            p.requires_grad = False
        zt = latents(AdaptedModel(base, nn.Identity(), MODEL_NAME).to(device),
                     tr_w, bs, device, apply_adapter=False)
        sl = src_lat.numpy()
        adapted = AdaptedModel(base, FeatNorm(
            torch.tensor(sl.mean(0)), torch.tensor(sl.std(0)),
            torch.tensor(zt.mean(0)), torch.tensor(zt.std(0))).to(device),
            MODEL_NAME).to(device)
    elif spec["kind"] == "full_ft":
        adapted = copy.deepcopy(model)
        full_finetune(adapted,
                      torch.utils.data.DataLoader(torch.FloatTensor(tr_w),
                                                  batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(va_w),
                                                  batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
    elif spec["kind"] == "isca":
        adapted, diag = run_isca(model, pool, pool_gt, bs, args, device, latent_dim,
                                 cap=spec.get("cap", "bottleneck"),
                                 align=spec.get("align", "none"),
                                 src_lat=src_lat, src_mode=src_mode)
    else:  # adapter
        adapted = fresh_adapted(model, spec.get("cap", "bottleneck"), latent_dim, device)
        train_adapter(adapted, tr_w, va_w, bs, args.epochs_adapt, args.adapt_lr,
                      device, trim_q=spec.get("trim", 0.0),
                      align=spec.get("align", "none"),
                      src_lat=src_lat, src_mode=src_mode)
    el = 0.0 if spec["kind"] == "baseline" else time.perf_counter() - t0

    te_s = scores(adapted, D["eval_x"], bs, device)
    va_s = scores(adapted, va_w, bs, device)
    met = evaluate_all(te_s, D["eval_y"], va_s)
    tgt_lat = latents(adapted, tr_w, bs, device, apply_adapter=True)
    gaps = gap_metrics(src_lat.numpy(), tgt_lat)

    tot = sum(p.numel() for p in adapted.parameters())
    tra = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    if spec["kind"] in ("baseline", "featnorm"):
        tra = 0
    rec = {"stage": stage, "pair": pair, "seed": seed, "config": cfgname,
           "kind": spec["kind"], "capacity": spec.get("cap", "-"),
           "align": spec.get("align", "none"), "trim_q": spec.get("trim", 0.0),
           "contam_target": spec.get("contam", np.nan),
           "contam_actual": float(pool_gt.mean()),
           "inject": spec.get("inject", "-"),
           "uses_gt_for_condition": spec.get("contam", None) is not None,
           "trainable_params": tra, "trainable_pct": 100 * tra / max(tot, 1),
           "adapt_time_s": el, **met,
           "gap_mmd": gaps["mmd_rbf_median"], "gap_mean_l2": gaps["mean_l2"]}
    for d in diag:
        rec[f"isca_it{d['iter']}_contam"] = d["contam_selected"]
    np.save(out / "scores" / f"{pair}__{stage}__{cfgname}__s{seed}.npy", te_s)
    print(f"      [{cfgname:28s}] AUROC={met['auroc']:.4f} "
          f"AUPR-lift={met['aupr_lift']:.2f} F1@1%={met['f1_far0.01']:.4f} "
          f"pool오염={pool_gt.mean():.3f}"
          + (f" → 정제후 {diag[-1]['contam_selected']:.3f}" if diag else ""))
    return rec


# ===========================================================================
def build_specs(stage):
    if stage == "capacity":
        s = {"baseline": {"kind": "baseline"}}
        for c in ["shift", "diag", "bottleneck", "linear"]:
            s[f"cap_{c}"] = {"kind": "adapter", "cap": c}
        s["full_ft"] = {"kind": "full_ft"}
        return s
    if stage == "align":
        s = {"baseline": {"kind": "baseline"}, "featnorm": {"kind": "featnorm"}}
        for a in ["none", "mmd", "mmd_white", "mmd_mode"]:
            s[f"align_{a}"] = {"kind": "adapter", "cap": "bottleneck", "align": a}
        return s
    if stage == "isca":
        return {"baseline": {"kind": "baseline"},
                "adapter_plain": {"kind": "adapter", "cap": "bottleneck"},
                "trim0.5": {"kind": "adapter", "cap": "bottleneck", "trim": TRIM_Q},
                "isca": {"kind": "isca", "cap": "bottleneck"},
                "full_ft": {"kind": "full_ft"}}
    if stage == "inject":
        return {"baseline": {"kind": "baseline"},
                "adapter_plain": {"kind": "adapter", "cap": "bottleneck"},
                "trim0.5": {"kind": "adapter", "cap": "bottleneck", "trim": TRIM_Q},
                "isca": {"kind": "isca", "cap": "bottleneck"},
                "full_ft": {"kind": "full_ft"}}
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["capacity", "align", "isca", "inject"],
                    default="capacity")
    ap.add_argument("--datasets", nargs="*", default=["boiler", "smd"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--contam", nargs="*", type=float, default=list(CONTAM_LEVELS))
    ap.add_argument("--inject", choices=["synth", "real", "auto"], default="auto",
                    help="auto: Boiler=real(train 이상 사용), SMD=synth(누수 방지)")
    ap.add_argument("--epochs_pretrain", type=int, default=60)
    ap.add_argument("--epochs_adapt", type=int, default=30)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp3_out")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    device = train_cfg.device
    pairs = args.pairs or [p for d in args.datasets for p in PAIRS[d]]
    specs = build_specs(args.stage)
    print(f"[exp3] stage={args.stage} pairs={pairs} seeds={args.seeds} "
          f"configs={list(specs)} window={WINDOW} alpha={ALPHA}(고정) "
          f"trim={TRIM_Q}(고정) isca_keep={ISCA_KEEP}(고정)")

    if not args.analyze_only:
        rows = []
        f = out / "raw_results.csv"
        if f.exists():
            prev = pd.read_csv(f)
            m = ~((prev["stage"] == args.stage) & prev["pair"].isin(pairs)
                  & prev["seed"].isin(args.seeds))
            rows = prev[m].to_dict("records")

        for pair in pairs:
            rng = np.random.default_rng(1234)
            D = prepare(pair, args, rng)
            inj = ("real" if (args.inject == "auto" and pair.startswith("Boiler"))
                   else ("synth" if args.inject == "auto" else args.inject))
            print(f"\n{'━'*66}\n  {pair}  (src={len(D['src'])}w, clean={len(D['clean'])}w, "
                  f"anom={len(D['anom'])}w, eval={len(D['eval_x'])}w "
                  f"이상 {D['eval_y'].mean():.3f}) inject={inj}\n{'━'*66}")
            if len(D["clean"]) < 200:
                print("  ⚠ clean window 부족 → 건너뜀"); continue

            for seed in args.seeds:
                set_all_seeds(seed)
                model_cfg.n_heads = 5; model_cfg.input_size = D["n_feat"]
                model_cfg.window_size = WINDOW
                model_cfg.d_model = 128; model_cfg.dim_ff = 512
                ck = out / "ckpt" / f"{pair}__s{seed}__w{WINDOW}.pt"
                if ck.exists() and not args.force_pretrain:
                    model = create_model(MODEL_NAME, model_cfg)
                    model.load_state_dict(torch.load(ck, map_location=device))
                else:
                    n = int(len(D["src"]) * 0.8)
                    train_cfg.seed = seed; train_cfg.epochs = args.epochs_pretrain
                    model, _, _ = train_model(
                        MODEL_NAME, model_cfg, train_cfg,
                        torch.utils.data.DataLoader(torch.FloatTensor(D["src"][:n]),
                                                    batch_size=train_cfg.batch_size, shuffle=True),
                        torch.utils.data.DataLoader(torch.FloatTensor(D["src"][n:]),
                                                    batch_size=train_cfg.batch_size, shuffle=False),
                        verbose=False)
                    torch.save(model.state_dict(), ck)
                model = model.to(device)
                src_lat, src_mode = source_latents_modes(
                    model, D["src"][:4000], train_cfg.batch_size, device)

                levels = args.contam if args.stage == "inject" else [0.0]
                for c in levels:
                    rng2 = np.random.default_rng(seed * 1000 + int(c * 100))
                    pool, pgt = make_pool(D, c, inj, rng2)
                    tag = f"c{c:.2f}" if args.stage == "inject" else ""
                    print(f"  ── seed {seed}" + (f" | 오염 {c:.0%}" if tag else ""))
                    for name, spec in specs.items():
                        sp = dict(spec)
                        if args.stage == "inject":
                            sp["contam"] = c; sp["inject"] = inj
                        rows.append(run_cfg(pair, seed, args.stage,
                                            f"{name}{('__'+tag) if tag else ''}",
                                            sp, pool, pgt, D, model, src_lat,
                                            src_mode, args, device, out))
                        pd.DataFrame(rows).to_csv(f, index=False)
    analyze(out)


# ===========================================================================
def analyze(out: Path):
    f = out / "raw_results.csv"
    if not f.exists():
        print("결과 없음"); return
    df = pd.read_csv(f)
    for stage, g in df.groupby("stage"):
        print(f"\n{'='*70}\n[{stage}]")
        idx = ["pair", "config"] if stage != "inject" else ["pair", "contam_target", "config"]
        t = g.groupby(idx).agg(auroc=("auroc", "mean"), sd=("auroc", "std"),
                               aupr_lift=("aupr_lift", "mean"),
                               f1=("f1_far0.01", "mean"),
                               params=("trainable_params", "mean")).round(4)
        print(t.to_string())
        # pair 단위 집계 검정 (주 분석)
        base = g[g["kind"] == "baseline"].groupby(["pair", "seed"])["auroc"].mean()
        gg = g[g["kind"] != "baseline"].copy()
        gg["d"] = gg.apply(lambda r: r["auroc"] - base.get((r["pair"], r["seed"]), np.nan), axis=1)
        piv = gg.pivot_table(index="pair", columns="config", values="d")
        print("\n  -- baseline 대비 개선 (pair 단위, n={}) --".format(len(piv)))
        for c in piv.columns:
            v = piv[c].dropna()
            if len(v) >= 3:
                try:
                    _, p = sps.wilcoxon(v)
                except Exception:
                    p = np.nan
                print(f"    {c:30s} mean={v.mean():+.4f} win={int((v>0).sum())}/{len(v)} p={p:.4f}")
        if stage == "isca" or stage == "inject":
            cols = [c for c in g.columns if c.startswith("isca_it")]
            if cols:
                print("\n  -- ISCA 정제 진행 (선택된 pool의 실제 오염률) --")
                print(g.groupby("pair")[["contam_actual"] + cols].mean().round(4).to_string())
    df.to_csv(out / "raw_results.csv", index=False)
    print(f"\n[exp3] → {out.resolve()}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
exp1_benchmark.py — GT 라벨 벤치마크 실험 (SMD · Boiler)
==========================================================
목적 (교수님 피드백 반영):
  1) GT 라벨 벤치마크에서도 결론(Adapter > Baseline, ≈ Full FT)이 유지되는가
     → pseudo label 의존성 해소
  2) Healthy-Only adaptation의 필요성을 GT로 정량 증명
     → Boiler는 target train에 이상이 ~9% 섞여 있어 오염도를 정확히 알 수 있음
     → adapt_data ∈ {all, stat(label-free), gt(oracle)} 3종 비교
  3) MMD 기여를 domain gap 큰 전이에서 재확인 (α는 '고정', 튜닝 금지)

★ α는 전 데이터셋 고정(--alpha, 기본 0.001). target 라벨로 고르지 않음.
  (기존 α 스윕은 post-hoc 민감도 분석으로만 보고할 것)

실행 (anomaly_detection/ 에서):
    python exp1_benchmark.py --pairs SMD_1-1to1-2 --seeds 0          # 파일럿
    python exp1_benchmark.py --datasets smd boiler --seeds 0 1 2     # 본실험
    python exp1_benchmark.py --adapt_data all stat gt --pairs Boiler_1to2  # 프로토콜 실험
    python exp1_benchmark.py --analyze_only
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, ".")

# ── 기존 코드 (수정 없이 사용) ───────────────────────────────────────────────
from config import model_cfg, train_cfg
from models import create_model, select_best_head
from trainer import train_model
from run_cross_domain_adapter_v2 import (
    LatentAdapter, AdaptedModel, train_adapter_v2, full_finetune,
    collect_source_stats, get_latent_dim,
)
from exp1_metrics import evaluate, VAL_QUANTILE
from exp1_efficiency import count_params, AdaptTimer, measure_inference_latency
from exp1_stats import aggregate_mean_std, paired_t_vs_baseline, all_pairwise_t

# ---------------------------------------------------------------------------
# 경로 · 상수
# ---------------------------------------------------------------------------
SMD_DIR = Path("data/benchmark/OmniAnomaly/ServerMachineDataset")
BOILER_DIR = Path("data/benchmark/SASA-pytorch/datasets/Boiler")

BOILER_LABEL = "Abnormal Blow Down"
BOILER_DROP = ["boiler_no", "TIME", BOILER_LABEL]  # 나머지 20개가 feature

MODEL_NAME = "transformer_ae"
METHODS = ["baseline", "adapter_recon", "adapter_mmd", "full_ft"]
METRIC_COLS = ["auroc", "aupr", "f1_val_thr", "precision_val_thr",
               "recall_val_thr", "best_f1", "anomaly_ratio"]

# DACAD 프로토콜: 같은 데이터셋 내 machine/unit 간 전이
BENCH_PAIRS = {
    # SMD — DACAD 논문의 대표 설정(1-1 → 1-2) 및 역방향
    "SMD_1-1to1-2": dict(kind="smd", src="machine-1-1", tgt="machine-1-2"),
    "SMD_1-2to1-1": dict(kind="smd", src="machine-1-2", tgt="machine-1-1"),
    "SMD_2-1to2-2": dict(kind="smd", src="machine-2-1", tgt="machine-2-2"),
    "SMD_3-1to3-2": dict(kind="smd", src="machine-3-1", tgt="machine-3-2"),
    # Boiler — 폴더(=도메인) 간 전이
    "Boiler_1to2": dict(kind="boiler", src="1", tgt="2"),
    "Boiler_2to3": dict(kind="boiler", src="2", tgt="3"),
    "Boiler_3to1": dict(kind="boiler", src="3", tgt="1"),
}


def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 데이터 로더
# ---------------------------------------------------------------------------

def load_smd(entity: str):
    """SMD: train=정상만(라벨 없음), test+test_label=평가용."""
    tr = np.genfromtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",", dtype=np.float32)
    te = np.genfromtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",", dtype=np.float32)
    lb = np.genfromtxt(SMD_DIR / "test_label" / f"{entity}.txt", delimiter=",", dtype=np.float32)
    tr = np.nan_to_num(tr); te = np.nan_to_num(te)
    return dict(train=tr, train_label=np.zeros(len(tr), int),   # 구성상 전부 정상
                test=te, test_label=(lb > 0.5).astype(int))


def load_boiler(folder: str):
    """Boiler: train.csv에도 이상이 ~9% 포함 → train_label 함께 반환."""
    d = BOILER_DIR / folder
    tr_df = pd.read_csv(d / "train.csv")
    te_df = pd.read_csv(d / "test.csv")
    feats = [c for c in tr_df.columns if c not in BOILER_DROP]
    f = lambda df: np.nan_to_num(df[feats].to_numpy(np.float32))
    g = lambda df: (df[BOILER_LABEL].to_numpy() > 0.5).astype(int)
    return dict(train=f(tr_df), train_label=g(tr_df),
                test=f(te_df), test_label=g(te_df), features=feats)


def load_pair(cfg):
    if cfg["kind"] == "smd":
        return load_smd(cfg["src"]), load_smd(cfg["tgt"])
    return load_boiler(cfg["src"]), load_boiler(cfg["tgt"])


class MinMax:
    """source train 통계로만 fit (target 정보 미사용)."""
    def fit(self, x):
        self.lo = x.min(axis=0, keepdims=True)
        self.hi = x.max(axis=0, keepdims=True)
        self.rng = np.where((self.hi - self.lo) < 1e-8, 1.0, self.hi - self.lo)
        return self

    def __call__(self, x):
        return ((x - self.lo) / self.rng).astype(np.float32)


# ---------------------------------------------------------------------------
# Healthy 구간 선택 (교수님 지적 대응의 핵심)
# ---------------------------------------------------------------------------

def stat_healthy_mask(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    label-free 건전 구간 판정.
    전체 target train에 대해 robust 통계(median/MAD)로 feature별 z-score를 구하고,
    포인트별 최대 z를 상태지표(HI)로 사용. HI가 median+k·MAD(HI)를 넘으면 '의심'.
    → 오염이 소수(<50%)면 robust 통계가 정상 분포를 대표하므로
      초기 구간이 정상이라는 가정 없이도 동작 (배관·베어링 규칙의 일반화 버전).
    """
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826 + 1e-6
    hi = (np.abs(x - med) / mad).max(axis=1)
    m, s = np.median(hi), np.median(np.abs(hi - np.median(hi))) * 1.4826 + 1e-6
    return hi <= (m + k * s)          # True = healthy


def select_adapt_points(train_x, train_label, mode: str, k: float = 3.0):
    """mode: all(필터 없음) / stat(label-free) / gt(oracle, 상한 비교용)"""
    if mode == "all":
        mask = np.ones(len(train_x), bool)
    elif mode == "stat":
        mask = stat_healthy_mask(train_x, k)
    elif mode == "gt":
        mask = train_label == 0
    else:
        raise ValueError(mode)
    return mask


# ---------------------------------------------------------------------------
# window 유틸 (stride 지원 — 벤치마크는 시퀀스가 길어 학습은 stride>1 권장)
# ---------------------------------------------------------------------------

def make_win(x, w, stride=1):
    n = (len(x) - w) // stride + 1
    idx = np.arange(w)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def win_labels(point_labels, w, stride=1, rule="any"):
    p = np.asarray(point_labels, int)
    n = (len(p) - w) // stride + 1
    idx = np.arange(w)[None, :] + stride * np.arange(n)[:, None]
    return (p[idx].max(axis=1) if rule == "any" else p[idx][:, -1]).astype(int)


def win_healthy(mask, w, stride=1, tol=0.0):
    """window 내 비건전 비율이 tol 이하일 때만 healthy window로 채택."""
    m = (~np.asarray(mask, bool)).astype(float)
    n = (len(m) - w) // stride + 1
    idx = np.arange(w)[None, :] + stride * np.arange(n)[:, None]
    return m[idx].mean(axis=1) <= tol


def loader(w_arr, batch_size, shuffle):
    t = torch.FloatTensor(np.ascontiguousarray(w_arr))
    return torch.utils.data.DataLoader(t, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# score (window별 best-head MAE — 기존과 동일 정의)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_windows(model, w_arr, batch_size, device):
    model.eval()
    out = []
    for i in range(0, len(w_arr), batch_size):
        x = torch.FloatTensor(w_arr[i:i + batch_size]).to(device)
        heads = model(x)
        best, _, _ = select_best_head(heads, x)
        out.append(torch.mean(torch.abs(best - x), dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def make_score_fn(model):
    @torch.no_grad()
    def fn(x):
        heads = model(x)
        best, _, _ = select_best_head(heads, x)
        return torch.mean(torch.abs(best - x), dim=(1, 2))
    return fn


# ---------------------------------------------------------------------------
# pretrain (pair×seed 캐시)
# ---------------------------------------------------------------------------

def get_pretrained(pair, seed, src_win, n_feat, args, device, ckpt_dir: Path):
    model_cfg.n_heads = 5
    model_cfg.input_size = n_feat
    model_cfg.window_size = args.window
    model_cfg.d_model = 128
    model_cfg.dim_ff = 512

    path = ckpt_dir / f"{pair}__seed{seed}__pretrain.pt"
    if path.exists() and not args.force_pretrain:
        m = create_model(MODEL_NAME, model_cfg)
        m.load_state_dict(torch.load(path, map_location=device))
        print(f"    [pretrain] 캐시 로드")
        return m.to(device)

    n_tr = int(len(src_win) * 0.8)
    tr_l = loader(src_win[:n_tr], train_cfg.batch_size, True)
    va_l = loader(src_win[n_tr:], train_cfg.batch_size, False)
    train_cfg.seed = seed
    train_cfg.epochs = args.epochs_pretrain
    m, _, _ = train_model(MODEL_NAME, model_cfg, train_cfg, tr_l, va_l, verbose=False)
    torch.save(m.state_dict(), path)
    return m.to(device)


# ---------------------------------------------------------------------------
# 한 (pair, seed, adapt_data) 실행
# ---------------------------------------------------------------------------

def run_one(pair, cfg, seed, adapt_mode, args, device, out: Path):
    set_all_seeds(seed)
    S, T = load_pair(cfg)
    sc = MinMax().fit(S["train"])                      # source-fit (target 정보 미사용)
    s_tr = sc(S["train"])
    t_tr, t_te = sc(T["train"]), sc(T["test"])
    n_feat = s_tr.shape[1]
    W = args.window

    # source: 정상만 (SMD는 train 전체가 정상, Boiler는 GT 정상으로 한정)
    s_mask = S["train_label"] == 0
    s_win = make_win(s_tr, W, args.stride_train)
    s_keep = win_healthy(s_mask, W, args.stride_train)
    s_win = s_win[s_keep] if s_keep.any() else s_win

    # target 적응 데이터 (핵심 변수)
    t_mask = select_adapt_points(t_tr, T["train_label"], adapt_mode, args.k)
    t_all = make_win(t_tr, W, args.stride_train)
    t_keep = win_healthy(t_mask, W, args.stride_train)
    t_win = t_all[t_keep] if t_keep.any() else t_all
    contam_pt = float((T["train_label"] == 1).mean())
    contam_used = float(win_labels(T["train_label"], W, args.stride_train)[t_keep].mean()) \
        if t_keep.any() else float("nan")
    n_cut = int(len(t_win) * 0.8)
    at_l = loader(t_win[:n_cut], train_cfg.batch_size, True)
    av_w = t_win[n_cut:] if len(t_win) - n_cut >= 5 else t_win[-5:]
    av_l = loader(av_w, train_cfg.batch_size, False)

    # 평가: target test (stride 1, GT 라벨)
    te_win = make_win(t_te, W, 1)
    te_lab = win_labels(T["test_label"], W, 1, "any")
    print(f"    [data] src_win={len(s_win)} tgt_adapt={len(t_win)}(val {len(av_w)}) "
          f"test_win={len(te_win)} | target train 오염 {contam_pt:.3f}→적응데이터 {contam_used:.3f} "
          f"| test 이상 {te_lab.mean():.3f}")

    model = get_pretrained(pair, seed, s_win, n_feat, args, device, out / "ckpt")
    latent_dim = get_latent_dim(MODEL_NAME)
    dummy = LatentAdapter(latent_dim).to(device)
    for lin in (dummy.adapter[0], dummy.adapter[2]):
        nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    src_lat, src_hp = collect_source_stats(
        AdaptedModel(model, dummy, MODEL_NAME).to(device),
        loader(s_win[:min(len(s_win), 4000)], train_cfg.batch_size, False), device)

    recs, effs = [], []
    for method in args.methods:
        set_all_seeds(seed)
        with AdaptTimer() as t:
            if method == "baseline":
                adapted = model
            elif method in ("adapter_recon", "adapter_mmd"):
                base = copy.deepcopy(model)
                for p in base.parameters():
                    p.requires_grad = False
                adapted = AdaptedModel(base, LatentAdapter(latent_dim).to(device),
                                       MODEL_NAME).to(device)
                train_adapter_v2(adapted, at_l, av_l, src_lat, src_hp,
                                 args.epochs_adapt, args.adapt_lr, device,
                                 alpha=(0.0 if method == "adapter_recon" else args.alpha),
                                 beta=0.0,
                                 loss_mode=("recon_only" if method == "adapter_recon"
                                            else "recon_mmd"))
            elif method == "full_ft":
                adapted = copy.deepcopy(model)
                full_finetune(adapted, at_l, av_l, args.epochs_adapt, args.ft_lr, device)
            else:
                raise ValueError(method)

        eff = count_params(adapted)
        if method == "baseline":
            eff["trainable_params"], eff["trainable_pct"] = 0, 0.0
        eff["adapt_time_s"] = 0.0 if method == "baseline" else t.elapsed
        eff["peak_mem_mb"] = t.peak_mem_mb

        te_s = score_windows(adapted, te_win, train_cfg.batch_size, device)
        va_s = score_windows(adapted, av_w, train_cfg.batch_size, device)
        eff["infer_ms_per_window"] = measure_inference_latency(
            make_score_fn(adapted), loader(te_win, train_cfg.batch_size, False), device)

        tag = f"{pair}__{adapt_mode}__{method}__seed{seed}"
        np.save(out / "scores" / f"{tag}__test.npy", te_s)
        np.save(out / "scores" / f"{tag}__val.npy", va_s)

        r = evaluate(te_s, te_lab, va_s, quantile=VAL_QUANTILE)
        recs.append({"pair": pair, "adapt_data": adapt_mode, "method": method,
                     "seed": seed, "contam_train": contam_pt,
                     "contam_adapt": contam_used, **r.as_dict()})
        effs.append({"pair": pair, "adapt_data": adapt_mode, "method": method,
                     "seed": seed, **eff})
        print(f"      [{method:13s}] AUROC={r.auroc:.4f} AUPR={r.aupr:.4f} "
              f"F1={r.f1_val_thr:.4f} | {eff['adapt_time_s']:.1f}s "
              f"trainable={eff['trainable_pct']:.2f}%")
    return recs, effs


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------

def analyze(out: Path):
    df = pd.read_csv(out / "raw_results.csv")
    rows = []
    for (p, a, m), g in df.groupby(["pair", "adapt_data", "method"]):
        r = {"pair": p, "adapt_data": a, "method": m, "n_seeds": len(g)}
        for c in METRIC_COLS:
            v = g[c].astype(float).dropna()
            r[c] = (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1
                    else (f"{v.mean():.4f}" if len(v) else "nan"))
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values(["pair", "adapt_data", "method"])
    summ.to_csv(out / "summary_mean_std.csv", index=False)

    # Baseline 대비 t-test (adapt_data별로 분리)
    tt = []
    for a, g in df.groupby("adapt_data"):
        for met in ("auroc", "aupr", "f1_val_thr"):
            t = paired_t_vs_baseline(g, met)
            if len(t):
                t.insert(1, "adapt_data", a)
                tt.append(t)
    if tt:
        pd.concat(tt, ignore_index=True).to_csv(out / "ttest_vs_baseline.csv", index=False)

    # 프로토콜 효과: adapt_data 변형 간 비교 (healthy-only 필요성 근거)
    if df["adapt_data"].nunique() > 1:
        piv = df.pivot_table(index=["pair", "method", "seed"],
                             columns="adapt_data", values="auroc").reset_index()
        piv.to_csv(out / "protocol_effect_auroc.csv", index=False)
        print("\n=== 적응 데이터 프로토콜별 AUROC (pair×method 평균) ===")
        print(df.groupby(["pair", "adapt_data"])["auroc"].mean().round(4).to_string())

    print("\n=== summary (mean ± std) ===")
    print(summ[["pair", "adapt_data", "method", "auroc", "aupr", "f1_val_thr"]]
          .to_string(index=False))
    tp = out / "ttest_vs_baseline.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        print("\n=== paired t-test vs Baseline (AUROC) ===")
        print(t[t.metric == "auroc"][["pair", "adapt_data", "method",
                                      "diff(method-baseline)", "p_value",
                                      "significant(p<0.05)"]].to_string(index=False))
    print(f"\n[analyze] 완료 → {out.resolve()}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=["smd"], choices=["smd", "boiler"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--adapt_data", nargs="*", default=["stat"],
                    choices=["all", "stat", "gt"],
                    help="적응 데이터 선택: all=필터없음 / stat=label-free / gt=oracle")
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="MMD 가중치 — 전 데이터셋 고정(튜닝 금지)")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride_train", type=int, default=5,
                    help="학습 window stride (평가는 항상 1)")
    ap.add_argument("--epochs_pretrain", type=int, default=30)
    ap.add_argument("--epochs_adapt", type=int, default=20)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp1_bench")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    pairs = args.pairs or [k for k, v in BENCH_PAIRS.items() if v["kind"] in args.datasets]
    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    device = train_cfg.device
    print(f"[bench] device={device} pairs={pairs} seeds={args.seeds} "
          f"adapt_data={args.adapt_data} alpha={args.alpha}(고정) window={args.window}")

    if not args.analyze_only:
        def prev(path):
            if not path.exists():
                return []
            d = pd.read_csv(path)
            keep = ~(d["pair"].isin(pairs) & d["method"].isin(args.methods)
                     & d["seed"].isin(args.seeds) & d["adapt_data"].isin(args.adapt_data))
            return d[keep].to_dict("records")
        rec, eff = prev(out / "raw_results.csv"), prev(out / "efficiency.csv")
        t0 = time.time()
        for pair in pairs:
            cfg = BENCH_PAIRS[pair]
            print(f"\n{'━'*64}\n  {pair}  ({cfg['src']} → {cfg['tgt']})\n{'━'*64}")
            for mode in args.adapt_data:
                for seed in args.seeds:
                    print(f"  ── adapt_data={mode} | seed {seed}")
                    r, e = run_one(pair, cfg, seed, mode, args, device, out)
                    rec += r; eff += e
                    pd.DataFrame(rec).to_csv(out / "raw_results.csv", index=False)
                    pd.DataFrame(eff).to_csv(out / "efficiency.csv", index=False)
        print(f"\n[bench] 완료 ({(time.time()-t0)/60:.1f}분)")

    analyze(out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_experiment1.py — 실험 1: 탐지 성능 평가 (+실험 4 효율 로깅 병행)
=====================================================================
기존 파일(config.py, models.py, trainer.py, run_cross_domain_adapter_v2.py)을
그대로 import — 기존 코드는 수정하지 않음.

실행 (anomaly_detection/ 폴더에서):
    python run_experiment1.py                          # 5 pair × 4 method × 5 seed
    python run_experiment1.py --pairs XB1toXB2 --seeds 0 1
    python run_experiment1.py --analyze_only           # 저장 score로 표·그림 재생성
    python run_experiment1.py --eval_on test           # test 20%만 평가 (기본: full)

파이프라인 (pair × seed):
  1) 데이터: v2와 동일 정규화 (배관=도메인별 MinMax / 베어링=source-fit MinMax)
  2) 라벨: 배관 fault_idx=2549 / 베어링 RMS μ+kσ (원본 CSV 기준)
  3) source pretrain 1회 → checkpoint 캐시 (method 4개가 공유)
  4) method별 적응 (v2의 train_adapter_v2 / full_finetune 그대로 호출)
     + 효율 로깅 (trainable %, adapt time, latency, peak mem)
  5) score: 전 target 시퀀스 window별 best-head MAE (기존 MAE 컨벤션 유지) → npy 저장
  6) 지표: AUROC·AUPR 주지표 + adapt_val 99.9% threshold F1 + Best F1(참고)
  7) analyze_results1: mean±std, Baseline 대비 paired t-test, k 민감도, Pareto
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
    FEATURE_COLS, LatentAdapter, AdaptedModel,
    train_adapter_v2, full_finetune,
    make_dataloader, make_windows, load_and_normalize,
    collect_source_stats, get_latent_dim,
)

# ── 실험 1 모듈 ──────────────────────────────────────────────────────────────
from exp1_labeling import (pipe_labels, bearing_pseudo_labels,
                           point_to_window_labels, PIPE_FAULT_IDX_DEFAULT)
from exp1_metrics import evaluate, VAL_QUANTILE
from exp1_efficiency import count_params, AdaptTimer, measure_inference_latency

# ---------------------------------------------------------------------------
# 확정 설정
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
METHODS = ["baseline", "adapter_recon", "adapter_mmd", "full_ft"]
MODEL_NAME = "transformer_ae"        # TF 확정 (동일 도메인 6개 중 4개 1등)
WINDOW_SIZE = 20                     # v2 main()과 동일 override
K_DEFAULT = 3.0

DOMAIN_PAIRS = {
    "Sim2Real": dict(kind="pipe",
                     src="data/pipe/simCase1_pfei.csv",
                     tgt="data/pipe/realCase1_pfei.csv",
                     alpha=0.1),
    "XB1toXB2": dict(kind="bearing",
                     src="data/xjtu_sy/XB1_combined.csv",
                     tgt="data/xjtu_sy/XB2_combined.csv", alpha=0.001),
    "XB2toXB1": dict(kind="bearing",
                     src="data/xjtu_sy/XB2_combined.csv",
                     tgt="data/xjtu_sy/XB1_combined.csv", alpha=0.001),
    "PB1toPB2": dict(kind="bearing",
                     src="data/phm2012/PB1_combined.csv",
                     tgt="data/phm2012/PB2_combined.csv", alpha=0.001),
    "PB2toPB1": dict(kind="bearing",
                     src="data/phm2012/PB2_combined.csv",
                     tgt="data/phm2012/PB1_combined.csv", alpha=0.001),
}

from pipe_columns import SIM_COLS, REAL_COLS  # NDA: names read from work/data/pipe/columns.json


def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 데이터 로드 (v2 main()의 방식 그대로)
# ---------------------------------------------------------------------------

def load_pair(cfg: dict):
    if cfg["kind"] == "pipe":
        from sklearn.preprocessing import MinMaxScaler
        src_df, tgt_df = pd.read_csv(cfg["src"]), pd.read_csv(cfg["tgt"])
        s1, s2 = MinMaxScaler(), MinMaxScaler()
        src = s1.fit_transform(src_df[SIM_COLS].values.astype(np.float32))
        tgt = s2.fit_transform(tgt_df[REAL_COLS].values.astype(np.float32))
        n_feat = 5
    else:
        src, tgt = load_and_normalize(cfg["src"], cfg["tgt"], FEATURE_COLS)  # minmax, source-fit
        n_feat = 8
    return src, tgt, n_feat


def build_labels(pair: str, cfg: dict, k: float, persistence: int, out: Path,
                 robust: bool = False):
    """반환: (window_labels, healthy_regions) — healthy_regions는 [(start, end)) point 구간."""
    if cfg["kind"] == "pipe":
        n_rows = len(pd.read_csv(cfg["tgt"]))
        res = pipe_labels(n_rows, PIPE_FAULT_IDX_DEFAULT)
        healthy = [(0, res["onset_index"])]
    else:
        res = bearing_pseudo_labels(cfg["tgt"], k=k, persistence=persistence,
                                    robust=robust)
        healthy = []
        for m in res["segment_meta"]:
            end = m["onset_global"] if m["onset_global"] is not None else m["start"] + m["length"]
            healthy.append((m["start"], end))
    win = point_to_window_labels(res["labels"], WINDOW_SIZE, rule="any")
    np.save(out / f"{pair}__window_labels.npy", win)
    meta = {kk: vv for kk, vv in res.items() if kk != "labels"}
    (out / f"{pair}__label_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  [label] {pair}: type={meta['type']} onset={meta['onset_index']} "
          f"window_anomaly_ratio={win.mean():.3f}")
    if win.mean() in (0.0, 1.0):
        print(f"  ⚠⚠⚠ [경고] {pair}: window 라벨이 단일 클래스(비율={win.mean():.0f}) → "
              f"AUROC/AUPR 정의 불가(nan). exp1_results/onset_table.csv에서 "
              f"k·persistence·robust 조합별 onset을 확인하고 "
              f"--persistence 5 또는 exp1_labeling.bearing_pseudo_labels(robust=True) 사용 검토.")
    n_healthy = sum(e - s0 for s0, e in healthy)
    print(f"  [label] healthy-adapt 가용 정상 구간: {len(healthy)}개, 총 {n_healthy} points")
    return win, healthy


def region_windows_split(tgt, regions, window, train_frac=0.8, min_len_extra=5):
    """세그먼트 경계를 넘지 않는 healthy window 생성 후 시간순 80/20 분할."""
    tr, va = [], []
    for s0, e in regions:
        if e - s0 < window + min_len_extra:
            continue
        w = make_windows(tgt[s0:e], window)
        c = max(int(len(w) * train_frac), 1)
        tr.append(w[:c]); va.append(w[c:])
    if not tr:
        raise RuntimeError("healthy 구간이 window보다 짧아 적응 데이터를 만들 수 없음")
    return np.concatenate(tr), (np.concatenate(va) if va else np.concatenate(tr)[-5:])


def loader_from_windows(w, batch_size, shuffle):
    t = torch.FloatTensor(np.asarray(w))
    return torch.utils.data.DataLoader(t, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def score_window_array(model, windows, batch_size, device):
    model.eval()
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.FloatTensor(windows[i:i + batch_size]).to(device)
        heads = model(x)
        best, _, _ = select_best_head(heads, x)
        out.append(torch.mean(torch.abs(best - x), dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Score: 전 시퀀스 window별 best-head MAE (배치 처리, 기존 MAE 컨벤션)
# ---------------------------------------------------------------------------

@torch.no_grad()
def window_scores_mae(model, data_norm, window_size, batch_size, device):
    model.eval()
    windows = make_windows(data_norm, window_size)
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.FloatTensor(windows[i:i + batch_size]).to(device)
        heads = model(x)                              # (N,B,W,F)
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
# Pretrain (pair×seed당 1회, 캐시)
# ---------------------------------------------------------------------------

def get_pretrained(pair, seed, src, n_feat, args, device, ckpt_dir: Path):
    model_cfg.n_heads = 5
    model_cfg.input_size = n_feat
    model_cfg.window_size = WINDOW_SIZE
    model_cfg.d_model = 128
    model_cfg.dim_ff = 512
    model_cfg.lstm_hidden = 128
    model_cfg.lstm_latent = 64
    model_cfg.tcn_channels = [128, 128, 128]

    path = ckpt_dir / f"{pair}__seed{seed}__pretrain.pt"
    if path.exists() and not args.force_pretrain:
        model = create_model(MODEL_NAME, model_cfg)
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  [pretrain] 캐시 로드: {path.name}")
        return model.to(device)

    n_tr = int(len(src) * 0.8)
    tr_loader = make_dataloader(src[:n_tr], WINDOW_SIZE, train_cfg.batch_size, shuffle=True)
    va_loader = make_dataloader(src[n_tr:], WINDOW_SIZE, train_cfg.batch_size, shuffle=False)
    train_cfg.seed = seed
    train_cfg.epochs = args.epochs_pretrain
    model, _, _ = train_model(MODEL_NAME, model_cfg, train_cfg,
                              tr_loader, va_loader, verbose=False)
    torch.save(model.state_dict(), path)
    return model.to(device)


# ---------------------------------------------------------------------------
# 한 (pair, seed): pretrain 공유 + 4 method 적응·score·효율
# ---------------------------------------------------------------------------

def run_pair_seed(pair, cfg, seed, args, device, out: Path, win_labels,
                  healthy_regions):
    set_all_seeds(seed)
    src, tgt, n_feat = load_pair(cfg)

    n = len(tgt)
    i60, i80 = int(n * 0.6), int(n * 0.8)
    if args.split_mode == "healthy":
        # 핵심 수정: pseudo onset '이전' 정상 구간만으로 적응 (세그먼트 경계 미횡단)
        tr_w, va_w = region_windows_split(tgt, healthy_regions, WINDOW_SIZE)
        tgt_tr_loader = loader_from_windows(tr_w, train_cfg.batch_size, shuffle=True)
        tgt_va_loader = loader_from_windows(va_w, train_cfg.batch_size, shuffle=False)
        print(f"    [split=healthy] adapt_train={len(tr_w)}w adapt_val={len(va_w)}w "
              f"(정상 구간만, 열화 오염 0%)")
    else:  # legacy 60/20/20
        tgt_tr_loader = make_dataloader(tgt[:i60], WINDOW_SIZE, train_cfg.batch_size, shuffle=True)
        tgt_va_loader = make_dataloader(tgt[i60:i80], WINDOW_SIZE, train_cfg.batch_size, shuffle=False)

    model = get_pretrained(pair, seed, src, n_feat, args, device, out / "ckpt")

    # source stats (MMD·consist용) — v2 방식 그대로 (zero adapter)
    latent_dim = get_latent_dim(MODEL_NAME)
    dummy = LatentAdapter(latent_dim).to(device)
    nn.init.zeros_(dummy.adapter[0].weight); nn.init.zeros_(dummy.adapter[0].bias)
    nn.init.zeros_(dummy.adapter[2].weight); nn.init.zeros_(dummy.adapter[2].bias)
    n_src_tr = int(len(src) * 0.8)
    src_loader_eval = make_dataloader(src[:n_src_tr], WINDOW_SIZE,
                                      train_cfg.batch_size, shuffle=False)
    src_latents, src_head_probs = collect_source_stats(
        AdaptedModel(model, dummy, MODEL_NAME).to(device), src_loader_eval, device)

    records, eff_rows = [], []
    for method in args.methods:
        set_all_seeds(seed)  # method 간 동일 초기조건
        with AdaptTimer() as t:
            if method == "baseline":
                adapted = model
            elif method in ("adapter_recon", "adapter_mmd"):
                adapter = LatentAdapter(latent_dim).to(device)
                base = copy.deepcopy(model)
                for p in base.parameters():
                    p.requires_grad = False
                adapted = AdaptedModel(base, adapter, MODEL_NAME).to(device)
                mode = "recon_only" if method == "adapter_recon" else "recon_mmd"
                pair_alpha = cfg["alpha"] if args.alpha is None else args.alpha
                a = 0.0 if method == "adapter_recon" else pair_alpha
                train_adapter_v2(adapted, tgt_tr_loader, tgt_va_loader,
                                 src_latents, src_head_probs,
                                 args.epochs_adapt, args.adapt_lr, device,
                                 alpha=a, beta=0.0, loss_mode=mode)
            elif method == "full_ft":
                adapted = copy.deepcopy(model)
                full_finetune(adapted, tgt_tr_loader, tgt_va_loader,
                              args.epochs_adapt, args.ft_lr, device)
            else:
                raise ValueError(method)

        eff = count_params(adapted)
        if method == "baseline":
            eff["trainable_params"], eff["trainable_pct"] = 0, 0.0
        eff["adapt_time_s"] = 0.0 if method == "baseline" else t.elapsed
        eff["peak_mem_mb"] = t.peak_mem_mb

        # score (전 target 시퀀스 + adapt_val)
        full_scores = window_scores_mae(adapted, tgt, WINDOW_SIZE,
                                        train_cfg.batch_size, device)
        if args.split_mode == "healthy":
            val_scores = score_window_array(adapted, va_w, train_cfg.batch_size, device)
        else:
            val_scores = window_scores_mae(adapted, tgt[i60:i80], WINDOW_SIZE,
                                           train_cfg.batch_size, device)
        full_eval_loader = make_dataloader(tgt, WINDOW_SIZE,
                                           train_cfg.batch_size, shuffle=False)
        eff["infer_ms_per_window"] = measure_inference_latency(
            make_score_fn(adapted), full_eval_loader, device)

        tag = f"{pair}__{method}__seed{seed}"
        np.save(out / "scores" / f"{tag}__full.npy", full_scores)
        np.save(out / "scores" / f"{tag}__val.npy", val_scores)

        # 지표 (eval_on: full=전 시퀀스 / test=마지막 20% window)
        if args.eval_on == "test":
            s = full_scores[i80:]                # window 시작 index i80 이후
            y = win_labels[i80:i80 + len(s)]
        else:
            s, y = full_scores, win_labels[:len(full_scores)]
        # threshold 산정용 정상 score
        if args.split_mode == "healthy":
            thr_scores = val_scores            # 구성상 전부 정상
            val_norm_frac = 1.0
        else:
            val_wl = win_labels[i60: i60 + len(val_scores)]
            val_normal = val_scores[val_wl == 0]
            thr_scores = val_normal if len(val_normal) >= 30 else val_scores
            val_norm_frac = float((val_wl == 0).mean())
        res = evaluate(s, y, thr_scores, quantile=VAL_QUANTILE)
        records.append({"pair": pair, "method": method, "seed": seed,
                        "eval_on": args.eval_on, "test_start_win": i80,
                        "val_normal_frac": val_norm_frac,
                        "split_mode": args.split_mode,
                        **res.as_dict()})
        eff_rows.append({"pair": pair, "method": method, "seed": seed, **eff})
        print(f"    [{method}] AUROC={res.auroc:.4f} AUPR={res.aupr:.4f} "
              f"F1(val-thr)={res.f1_val_thr:.4f} | adapt={eff['adapt_time_s']:.1f}s "
              f"trainable={eff['trainable_pct']:.2f}%")
    return records, eff_rows


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=list(DOMAIN_PAIRS))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--epochs_pretrain", type=int, default=200)
    ap.add_argument("--epochs_adapt", type=int, default=50)
    ap.add_argument("--adapt_lr", type=float, default=1e-3, help="adapter lr")
    ap.add_argument("--ft_lr", type=float, default=1e-4,
                    help="full FT lr (기존 v2와 동일하게 adapter보다 낮게)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="지정 시 전 pair의 MMD 가중치 override (예: 0.01)")
    ap.add_argument("--split_mode", choices=["healthy", "legacy"], default="healthy",
                    help="healthy: onset 이전 정상 구간만 적응(권장) / legacy: 앞 60%%")
    ap.add_argument("--k", type=float, default=K_DEFAULT)
    ap.add_argument("--persistence", type=int, default=5,
                    help="pseudo onset 연속 초과 조건 (기본 5=스파이크 방어, 1=계획서 원정의)")
    ap.add_argument("--robust", action="store_true",
                    help="정상구간 통계를 median+MAD로 (스파이크·드리프트 강건)")
    ap.add_argument("--eval_on", choices=["full", "test"], default="full")
    ap.add_argument("--out", default="exp1_results")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    device = train_cfg.device
    print(f"[exp1] device={device} | pairs={args.pairs} | seeds={args.seeds} "
          f"| eval_on={args.eval_on} | k={args.k} persistence={args.persistence}")

    if not args.analyze_only:
        # 기존 결과 로드 → 이번에 재실행할 (pair,method,seed) 조합만 교체 (분할 실행 지원)
        def _load_prev(path, keys):
            if not path.exists():
                return []
            prev = pd.read_csv(path)
            mask = ~(prev["pair"].isin(args.pairs)
                     & prev["method"].isin(args.methods)
                     & prev["seed"].isin(args.seeds))
            return prev[mask].to_dict("records")
        all_rec = _load_prev(out / "raw_results.csv", None)
        all_eff = _load_prev(out / "efficiency.csv", None)
        t0 = time.time()
        for pair in args.pairs:
            cfg = DOMAIN_PAIRS[pair]
            print(f"\n{'━'*60}\n  {pair}\n{'━'*60}")
            win_labels, healthy_regions = build_labels(pair, cfg, args.k,
                                                       args.persistence, out,
                                                       robust=args.robust)
            for seed in args.seeds:
                print(f"  ── seed {seed} " + "─" * 40)
                rec, eff = run_pair_seed(pair, cfg, seed, args, device, out,
                                         win_labels, healthy_regions)
                all_rec += rec; all_eff += eff
                # 중간 저장 (중단 대비)
                pd.DataFrame(all_rec).to_csv(out / "raw_results.csv", index=False)
                pd.DataFrame(all_eff).to_csv(out / "efficiency.csv", index=False)
        print(f"\n[exp1] 학습·평가 완료 ({(time.time()-t0)/60:.1f}분)")

    from analyze_results1 import analyze
    analyze(out, args.pairs, args.methods, args.seeds, DOMAIN_PAIRS,
            k_sweep=(2.0, 3.0, 4.0), persistence=args.persistence)


if __name__ == "__main__":
    main()

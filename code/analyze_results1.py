# -*- coding: utf-8 -*-
"""
analyze_results1.py — 저장된 score·라벨로 표·검정·민감도·그림 생성 (재학습 불필요)
=====================================================================================
생성물 (exp1_results/):
  summary_mean_std.csv      pair×method mean±std (PPT 메인 표)
  ttest_vs_baseline.csv     Baseline 대비 paired t-test (AUROC/AUPR/F1)
  ttest_all_pairs.csv       전 method 쌍 (Full FT vs Adapter+MMD 동등성 논거)
  sensitivity_k.csv/.png    베어링 pseudo label k=2/3/4 민감도 (score 재사용)
  onset_table.csv           k × persistence 별 onset 비교 (라벨 방어자료)
  efficiency_summary.csv    효율 mean±std
  pareto_auroc.png          x=trainable params(log), y=AUROC
  score_timeline_<pair>.png score + val 99.9% threshold + 이상구간 음영
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp1_labeling import bearing_pseudo_labels, point_to_window_labels, onset_table
from exp1_metrics import evaluate, VAL_QUANTILE
from exp1_stats import aggregate_mean_std, paired_t_vs_baseline, all_pairwise_t

METRIC_COLS = ["auroc", "aupr", "f1_val_thr", "precision_val_thr",
               "recall_val_thr", "best_f1", "anomaly_ratio"]
WINDOW_SIZE = 20


def _load(out: Path, pair, method, seed):
    v = out / "scores" / f"{pair}__{method}__seed{seed}__val.npy"
    f = out / "scores" / f"{pair}__{method}__seed{seed}__full.npy"
    return (np.load(v), np.load(f)) if v.exists() and f.exists() else None


def analyze(out, pairs, methods, seeds, domain_pairs,
            k_sweep=(2.0, 3.0, 4.0), persistence=1):
    out = Path(out)
    df = pd.read_csv(out / "raw_results.csv")

    # ── 1) 표 + t-test ───────────────────────────────────────────────────────
    aggregate_mean_std(df, METRIC_COLS).to_csv(out / "summary_mean_std.csv", index=False)
    tt_b = [paired_t_vs_baseline(df, m) for m in ("auroc", "aupr", "f1_val_thr")]
    tt_b = [t for t in tt_b if len(t)]
    if tt_b:
        pd.concat(tt_b, ignore_index=True).to_csv(out / "ttest_vs_baseline.csv", index=False)
    tt_a = [t for t in (all_pairwise_t(df, m) for m in ("auroc", "aupr")) if len(t)]
    if tt_a:
        pd.concat(tt_a, ignore_index=True).to_csv(out / "ttest_all_pairs.csv", index=False)

    # ── 2) k 민감도 + onset 비교 (베어링만) ─────────────────────────────────
    sens, onset_rows = [], []
    for pair in pairs:
        cfg = domain_pairs[pair]
        if cfg["kind"] != "bearing":
            continue
        ot = onset_table(cfg["tgt"], k_values=k_sweep, persistences=(1, 5))
        ot.insert(0, "pair", pair)
        onset_rows.append(ot)
        for k in k_sweep:
            res = bearing_pseudo_labels(cfg["tgt"], k=k, persistence=persistence)
            wl = point_to_window_labels(res["labels"], WINDOW_SIZE, "any")
            for method in methods:
                a_list, p_list = [], []
                for seed in seeds:
                    loaded = _load(out, pair, method, seed)
                    if loaded is None:
                        continue
                    vs, fs = loaded
                    y = wl[:len(fs)]
                    if len(np.unique(y)) < 2:
                        continue
                    r = evaluate(fs, y, vs, quantile=VAL_QUANTILE)
                    a_list.append(r.auroc); p_list.append(r.aupr)
                if a_list:
                    sens.append({"pair": pair, "k": k, "method": method,
                                 "onset_index": res["onset_index"],
                                 "auroc_mean": np.mean(a_list),
                                 "auroc_std": np.std(a_list, ddof=1) if len(a_list) > 1 else 0.0,
                                 "aupr_mean": np.mean(p_list)})
    if onset_rows:
        pd.concat(onset_rows, ignore_index=True).to_csv(out / "onset_table.csv", index=False)
    sens_df = pd.DataFrame(sens)
    if len(sens_df):
        sens_df.to_csv(out / "sensitivity_k.csv", index=False)
        n = sens_df["pair"].nunique()
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
        for ax, (pair, g) in zip(axes[0], sens_df.groupby("pair")):
            for method, gm in g.groupby("method"):
                ax.errorbar(gm["k"], gm["auroc_mean"], yerr=gm["auroc_std"],
                            marker="o", capsize=3, label=method)
            ax.set_title(pair, fontsize=10)
            ax.set_xlabel("k (mu+k*sigma)"); ax.set_ylabel("AUROC")
            ax.legend(fontsize=6)
        fig.suptitle("Pseudo-label threshold sensitivity", fontsize=11)
        fig.tight_layout(); fig.savefig(out / "sensitivity_k.png", dpi=150); plt.close(fig)

    # ── 3) 효율 + Pareto ─────────────────────────────────────────────────────
    eff_path = out / "efficiency.csv"
    if eff_path.exists():
        eff = pd.read_csv(eff_path)
        cols = ["trainable_params", "trainable_pct", "adapt_time_s",
                "infer_ms_per_window", "peak_mem_mb"]
        eff.groupby(["pair", "method"])[cols].agg(["mean", "std"]).to_csv(
            out / "efficiency_summary.csv")

        merged = (df.groupby(["pair", "method"])["auroc"].mean().reset_index()
                  .merge(eff.groupby(["pair", "method"])["trainable_params"]
                         .mean().reset_index(), on=["pair", "method"]))
        merged = merged.dropna(subset=["auroc"])
        if len(merged):
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                mk = {"baseline": "s", "adapter_recon": "^",
                      "adapter_mmd": "o", "full_ft": "D"}
                for method, g in merged.groupby("method"):
                    ax.scatter(np.clip(g["trainable_params"], 1, None), g["auroc"],
                               marker=mk.get(method, "x"), s=70, label=method, alpha=0.85)
                ax.set_xscale("log")
                ax.set_xlabel("Trainable parameters (log)")
                ax.set_ylabel("AUROC (pair mean)")
                ax.set_title("Performance-Efficiency Pareto")
                ax.legend(); fig.tight_layout()
                fig.savefig(out / "pareto_auroc.png", dpi=150)
            except Exception as e:
                print(f"[analyze] Pareto plot 생략: {e}")
            finally:
                plt.close("all")
        else:
            print("[analyze] AUROC가 전부 NaN → Pareto plot 생략 "
                  "(라벨이 단일 클래스인지 onset_table.csv 확인)")

    # ── 4) score timeline (seed 0, method 겹침) ──────────────────────────────
    for pair in pairs:
        lp = out / f"{pair}__window_labels.npy"
        if not lp.exists():
            continue
        wl = np.load(lp)
        fig, axes = plt.subplots(len(methods), 1, figsize=(11, 2.0 * len(methods)),
                                 sharex=True, squeeze=False)
        for ax, method in zip(axes[:, 0], methods):
            loaded = _load(out, pair, method, seeds[0])
            if loaded is None:
                continue
            vs, fs = loaded
            thr = float(np.quantile(vs, VAL_QUANTILE))
            y = wl[:len(fs)]
            ax.plot(fs, lw=0.7, color="dimgray")
            ax.axhline(thr, color="crimson", lw=0.8, ls="--")
            if y.sum():
                on = np.flatnonzero(y)
                ax.axvspan(on[0], on[-1], color="red", alpha=0.12)
            ax.set_ylabel(method, fontsize=8)
        axes[0, 0].set_title(f"{pair} - score / val-99.9% threshold / anomaly region")
        fig.tight_layout(); fig.savefig(out / f"score_timeline_{pair}.png", dpi=150)
        plt.close(fig)

    # ── 5) score 분포 진단 (정상/이상 score 평균·gap·threshold·예측비율) ─────
    gap_rows = []
    for pair in pairs:
        lp = out / f"{pair}__window_labels.npy"
        if not lp.exists():
            continue
        wl_full = np.load(lp)
        for method in methods:
            for seed in seeds:
                loaded = _load(out, pair, method, seed)
                if loaded is None:
                    continue
                vs, fs = loaded
                y = wl_full[:len(fs)]
                thr = float(np.quantile(vs, VAL_QUANTILE))
                nm = float(fs[y == 0].mean()) if (y == 0).any() else np.nan
                am = float(fs[y == 1].mean()) if (y == 1).any() else np.nan
                gap_rows.append({"pair": pair, "method": method, "seed": seed,
                                 "normal_score_mean": nm, "anomaly_score_mean": am,
                                 "gap": am - nm, "gap_ratio": am / max(nm, 1e-12),
                                 "val_threshold": thr,
                                 "pred_anomaly_ratio": float((fs >= thr).mean()),
                                 "label_anomaly_ratio": float(y.mean())})
    gap_df = pd.DataFrame(gap_rows)
    if len(gap_df):
        gap_df.to_csv(out / "score_gap.csv", index=False)
        g = gap_df.groupby(["pair", "method"])[["normal_score_mean", "anomaly_score_mean",
                                                "gap", "val_threshold",
                                                "pred_anomaly_ratio"]].mean().round(4)
        print("\n=== score 분포 진단 (seed 평균) ===")
        print(g.to_string())

    print("\n=== summary (mean ± std) ===")
    print(pd.read_csv(out / "summary_mean_std.csv")[
        ["pair", "method", "auroc", "aupr", "f1_val_thr"]].to_string(index=False))
    tt_path = out / "ttest_vs_baseline.csv"
    if tt_path.exists():
        tt = pd.read_csv(tt_path)
        print("\n=== paired t-test vs Baseline (AUROC) ===")
        print(tt[tt.metric == "auroc"][["pair", "method", "diff(method-baseline)",
                                        "p_value", "significant(p<0.05)"]].to_string(index=False))
    else:
        print("\n[analyze] seed 2개 미만 → paired t-test 생략")
    print(f"\n[analyze] 완료 → {out.resolve()}")

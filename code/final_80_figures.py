# -*- coding: utf-8 -*-
"""
final_80_figures.py — 논문 그림 일괄 생성
==========================================
Fig.2  누수 메커니즘 (IN/OUT 분리) — pair 단위 집계 + complete-case
Fig.3  Pareto : trainable params vs 적응 이득 (capacity 사다리)
Fig.4  Boiler 이상 구간 편중 (decile 별 이상비율)  <- 새 발견
Fig.5  contamination 곡선 (누수 통제, complete-case)

입력 (있는 것만 그린다)
  --rescored   results_rescored.csv
  --mechanism  mechanism_rows.csv (없으면 mechanism_summary.csv)
  --sweep      SWEEP_REPORT/results.csv
  --decile     diagnose_out/D2_auroc_by_time_decile.csv

실행:
  python final_80_figures.py --outdir figures_paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 300,
                     "savefig.bbox": "tight"})

ZERO_REMOVAL = ["SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2",
                "MSL_P-11toP-10", "MSL_P-10toP-11", "MSL_G-6toE-13"]
NICE = {"source_only": "No adaptation", "adapter_latent": "Adapter (1%)",
        "adapter_isca": "Adapter + self-refine", "full_ft": "Full fine-tuning",
        "adapter_input": "Adapter (input)", "adapter_output": "Adapter (output)"}
COL = {"source_only": "#808080", "adapter_latent": "#1f77b4",
       "adapter_isca": "#2ca02c", "full_ft": "#d62728"}
MM = 1 / 25.4


def load_rescored(path):
    d = pd.read_csv(path)
    if "auroc_nopool_temporal" not in d.columns:
        d["auroc_nopool_temporal"] = np.nan
    m = d.pair.isin(ZERO_REMOVAL) & d.auroc_nopool_temporal.isna() & d.auroc.notna()
    d.loc[m, "auroc_nopool_temporal"] = d.loc[m, "auroc"]
    return d


# ----------------------------------------------------------------- Fig.2
def fig_mechanism(path, outdir):
    d = pd.read_csv(path)
    need = {"pair", "method", "contam", "auroc_in", "auroc_out"}
    if not need.issubset(d.columns):
        print(f"[Fig2] 필요한 열이 없습니다({sorted(need - set(d.columns))}). "
              "mechanism_rows.csv 를 주세요."); return
    fig, ax = plt.subplots(1, 2, figsize=(190 * MM, 75 * MM), sharey=False)
    for k, (col, title) in enumerate([
            ("auroc_in", "(a) windows overlapping the adaptation pool"),
            ("auroc_out", "(b) windows disjoint from the adaptation pool")]):
        per = d.pivot_table(index="pair", columns=["method", "contam"],
                            values=col, aggfunc="mean")
        common = set(per.dropna().index)
        for meth in ["source_only", "adapter_latent", "adapter_isca", "full_ft"]:
            if meth not in per.columns.get_level_values(0):
                continue
            sub = per[meth].dropna()
            sub = sub.loc[[p for p in sub.index if p in common]]
            if sub.empty:
                continue
            x = [float(c) * 100 for c in sub.columns]
            ax[k].plot(x, sub.mean().values, "o-", color=COL.get(meth),
                       label=NICE.get(meth, meth), ms=4, lw=1.4)
        ax[k].set_title(title)
        ax[k].set_xlabel("Contamination ratio of the reference pool (%)")
        ax[k].set_ylabel("AUROC" if k == 0 else "")
        n = len(common)
        ax[k].text(0.98, 0.02, f"n = {n} pairs", transform=ax[k].transAxes,
                   ha="right", va="bottom", fontsize=7, color="0.35")
    ax[0].legend(frameon=False, fontsize=7)
    fig.savefig(Path(outdir) / "Fig2_leakage_mechanism.png")
    plt.close(fig)
    print("[Fig2] 저장 (pair 단위 집계 + complete-case)")


# ----------------------------------------------------------------- Fig.3
def fig_pareto(path, outdir):
    d = pd.read_csv(path)
    if "stage" in d.columns and (d.stage == "SW").any():
        d = d[d.stage == "SW"]        # results_rescored.csv 를 넘겨도 동작
    if "trainable_params" not in d.columns:
        print("[Fig3] trainable_params 열이 없습니다."); return
    col = "auroc_nopool_temporal" if "auroc_nopool_temporal" in d.columns else "auroc"
    d = d[d.get("diverged", 0) != 1]
    piv = d.pivot_table(index="pair", columns="config", values=col, aggfunc="mean")
    if "source_only" not in piv.columns:
        print("[Fig3] source_only 가 없어 Δ를 계산할 수 없습니다."); return
    base = piv["source_only"]
    prm = d.groupby("config")["trainable_params"].max()

    xs, ys, es, names = [], [], [], []
    for cfg in piv.columns:
        if cfg == "source_only":
            continue
        diff = (piv[cfg] - base).dropna()
        if diff.empty:
            continue
        xs.append(max(prm.get(cfg, 1), 1)); ys.append(diff.mean())
        es.append(diff.std() / max(np.sqrt(len(diff)), 1)); names.append(cfg)

    fig, ax = plt.subplots(figsize=(95 * MM, 75 * MM))
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, lw=1, capsize=2, color="#1f77b4")
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm.replace("_lr1e-3_e40", ""), (x, y), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(r"$\Delta$AUROC vs no adaptation")
    ax.set_title("Capacity vs adaptation gain (leakage-controlled)")
    fig.savefig(Path(outdir) / "Fig3_pareto_capacity.png")
    plt.close(fig)
    print(f"[Fig3] 저장 ({len(xs)}개 설정)")


# ----------------------------------------------------------------- Fig.4
def fig_concentration(path, outdir):
    d = pd.read_csv(path)
    if "anom_ratio" not in d.columns:
        print("[Fig4] anom_ratio 열이 없습니다."); return
    piv = d.pivot_table(index="decile", columns="pair", values="anom_ratio")
    fig, ax = plt.subplots(figsize=(120 * MM, 70 * MM))
    piv.plot(kind="bar", ax=ax, width=0.8, legend=True)
    ax.set_xlabel("Decile of the target test stream (1 = earliest)")
    ax.set_ylabel("Window-level anomaly ratio")
    ax.set_title("Anomalies concentrate in the first decile of the Boiler test stream")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.savefig(Path(outdir) / "Fig4_anomaly_concentration.png")
    plt.close(fig)
    print("[Fig4] 저장")


# ----------------------------------------------------------------- Fig.5
def fig_contamination(d, outdir):
    s = d[(d.stage == "S2") & d.auroc_nopool_temporal.notna()]
    if s.empty:
        print("[Fig5] S2 temporal 데이터가 없습니다."); return
    per = s.pivot_table(index="pair", columns=["method", "contam"],
                        values="auroc_nopool_temporal", aggfunc="mean")
    common = set(per.dropna().index)
    fig, ax = plt.subplots(figsize=(95 * MM, 75 * MM))
    for meth in ["source_only", "adapter_latent", "adapter_isca", "full_ft"]:
        if meth not in per.columns.get_level_values(0):
            continue
        sub = per[meth].dropna()
        sub = sub.loc[[p for p in sub.index if p in common]]
        if sub.empty:
            continue
        x = [float(c) * 100 for c in sub.columns]
        ax.plot(x, sub.mean().values, "o-", color=COL.get(meth),
                label=NICE.get(meth, meth), ms=4, lw=1.4)
    ax.set_xlabel("Contamination ratio of the reference pool (%)")
    ax.set_ylabel("AUROC (leakage-controlled)")
    ax.set_title(f"Contamination has no effect once leakage is controlled\n"
                 f"(n = {len(common)} pairs, complete case)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(Path(outdir) / "Fig5_contamination_controlled.png")
    plt.close(fig)
    print(f"[Fig5] 저장 (n={len(common)} pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescored", default="results_rescored.csv")
    ap.add_argument("--mechanism", default="mechanism_rows.csv")
    ap.add_argument("--sweep", default="jobs_out/final/SWEEP_REPORT/results.csv")
    ap.add_argument("--decile", default="diagnose_out/D2_auroc_by_time_decile.csv")
    ap.add_argument("--outdir", default="figures_paper")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    if Path(a.mechanism).exists():
        fig_mechanism(a.mechanism, out)
    else:
        print(f"[Fig2] 건너뜀 (없음: {a.mechanism})")
    if Path(a.sweep).exists():
        fig_pareto(a.sweep, out)
    else:
        print(f"[Fig3] 건너뜀 (없음: {a.sweep})")
    if Path(a.decile).exists():
        fig_concentration(a.decile, out)
    else:
        print(f"[Fig4] 건너뜀 (없음: {a.decile})")
    if Path(a.rescored).exists():
        fig_contamination(load_rescored(a.rescored), out)
    else:
        print(f"[Fig5] 건너뜀 (없음: {a.rescored})")
    print(f"\n[done] {out.resolve()}")


if __name__ == "__main__":
    main()

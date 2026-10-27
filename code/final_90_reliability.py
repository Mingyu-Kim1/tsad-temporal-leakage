# -*- coding: utf-8 -*-
"""
final_90_reliability.py — 신뢰성(하방 위험) 분석 + Fig.3 수정 + 지표 정합성
==========================================================================
GPU 불필요. 기존 표만으로 논문의 핵심 표/그림을 다시 만든다.

산출:
  R1_matched_metrics.csv   3개 지표를 '같은 pair 집합'에서 비교 (지표 비교 정합성)
  R2_reliability.csv       method별 mean/median/승패/최악 하락/부트스트랩 CI
  R3_pareto_report.csv     capacity Pareto (report mode 만, tune pair 제외)
  R4_noop_pairs.csv        adapter 가 identity 로 남은 pair (조기 종료 0 epoch)
  Fig3b_pareto_fixed.png   Fig.3 재생성 (mode=report 필터 + 라벨 겹침 제거)
  Fig6_reliability.png     pair 별 ΔAUROC 점도표 — 하방 위험 비교

실행:
  python final_90_reliability.py --tables tables_final ^
      --sweep jobs_out\\final\\SWEEP_REPORT\\results.csv --outdir tables_final
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

MM = 1 / 25.4
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
METRICS = ["auroc", "auroc_nopool_exact", "auroc_nopool_temporal"]
NICE = {"source_only": "No adaptation", "adapter_input": "Adapter (input)",
        "adapter_output": "Adapter (output)", "adapter_latent": "Adapter (latent, 1%)",
        "adapter_isca": "Adapter + self-refine", "full_ft": "Full fine-tuning"}


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(x):
    if wilcoxon is None or len(x) < 5:
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def load_pair_tables(tdir):
    out = {}
    for m in METRICS:
        f = Path(tdir) / f"T_S1_pairs__{m}.csv"
        if f.exists():
            out[m] = pd.read_csv(f).set_index("pair")
    return out


def summarise(piv, base_col="source_only", tag=""):
    rows = []
    if base_col not in piv.columns:
        return pd.DataFrame()
    base = piv[base_col]
    for c in piv.columns:
        if c == base_col:
            continue
        d = (piv[c] - base).dropna()
        if d.empty:
            continue
        lo, hi = boot_ci(d.values)
        neg = d[d < -1e-4]
        rows.append(dict(
            subset=tag, method=c, n_pairs=len(d),
            mean=d.mean(), median=d.median(), std=d.std(),
            ci_lo=lo, ci_hi=hi,
            wins=int((d > 1e-4).sum()), ties=int((d.abs() <= 1e-4).sum()),
            losses=int(len(neg)),
            worst_drop=float(d.min()), total_loss=float(neg.sum()) if len(neg) else 0.0,
            p_wilcoxon=wil(d.values)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables_final")
    ap.add_argument("--sweep", default="jobs_out/final/SWEEP_REPORT/results.csv")
    ap.add_argument("--sweep-metric", default="auto",
                    help="auroc_nopool_temporal / auroc_nopool / auto")
    ap.add_argument("--outdir", default="tables_final")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---------- R1 : 같은 pair 집합에서 3개 지표 비교 ----------
    pt = load_pair_tables(a.tables)
    if not pt:
        print(f"[오류] {a.tables}/T_S1_pairs__*.csv 가 없습니다."); return 1
    common = None
    for m, df in pt.items():
        s = set(df.dropna(how="all").index)
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"[R1] 3개 지표 공통 pair {len(common)}개: {', '.join(common)}")

    r1 = pd.concat([summarise(pt[m].loc[common], tag=f"matched-{m}")
                    for m in pt], ignore_index=True)
    r1.to_csv(out / "R1_matched_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- R2 : 신뢰성 (주 지표) ----------
    key = "auroc_nopool_temporal" if "auroc_nopool_temporal" in pt else list(pt)[0]
    r2 = summarise(pt[key].loc[common], tag=key)
    r2.to_csv(out / "R2_reliability.csv", index=False, encoding="utf-8-sig")

    # ---------- R4 : no-op pair ----------
    piv = pt[key].loc[common]
    noop = []
    for c in piv.columns:
        if c == "source_only":
            continue
        d = (piv[c] - piv["source_only"]).abs()
        for p in d[d <= 1e-4].index:
            noop.append(dict(method=c, pair=p,
                             source_only=piv.loc[p, "source_only"],
                             adapted=piv.loc[p, c]))
    pd.DataFrame(noop).to_csv(out / "R4_noop_pairs.csv", index=False,
                              encoding="utf-8-sig")

    # ---------- R3 : Pareto (report mode 만) ----------
    r3 = pd.DataFrame()
    sp = Path(a.sweep)
    if sp.exists():
        s = pd.read_csv(sp)
        if "mode" in s.columns:
            s = s[s["mode"] == "report"]           # ★ tune pair 혼입 차단
        if "diverged" in s.columns:
            s = s[s.diverged != 1]
        col = a.sweep_metric
        if col == "auto":
            col = ("auroc_nopool_temporal"
                   if "auroc_nopool_temporal" in s.columns
                   and s["auroc_nopool_temporal"].notna().any()
                   else "auroc_nopool")
        print(f"[R3] sweep 지표: {col}   pair {s.pair.nunique()}개, "
              f"config {s.config.nunique()}개")
        pv = s.pivot_table(index="pair", columns="config", values=col, aggfunc="mean")
        prm = s.groupby("config")["trainable_params"].max()
        r3 = summarise(pv, tag="sweep")
        r3["params"] = r3.method.map(prm)
        r3 = r3.sort_values("params")
        r3.to_csv(out / "R3_pareto_report.csv", index=False, encoding="utf-8-sig")

        # Fig 3b
        fig, ax = plt.subplots(figsize=(95 * MM, 72 * MM))
        x = r3["params"].values.astype(float)
        y = r3["mean"].values
        lo = y - r3["ci_lo"].values
        hi = r3["ci_hi"].values - y
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=4.5, lw=1, capsize=2.5,
                    color="#1f77b4", zorder=3)
        short = {"shift": "shift", "diag": "diag", "linear": "linear"}
        for xi, yi, nm in zip(x, y, r3.method):
            lab = nm.split("_")[0]
            lab = short.get(lab, lab.replace("bneck", "r="))
            if lab.startswith("fullft"):
                lab = "full FT"
            ax.annotate(lab, (xi, yi), fontsize=6, xytext=(0, 8),
                        textcoords="offset points", ha="center")
        ax.set_xscale("log")
        ax.axhline(0, color="0.5", lw=.8, ls="--")
        ax.set_xlabel("Trainable parameters per target (log scale)")
        ax.set_ylabel(r"$\Delta$AUROC vs no adaptation")
        ax.set_title(f"Capacity vs adaptation gain\n"
                     f"({len(pv)} leakage-free pairs, 95% bootstrap CI)", fontsize=8)
        fig.savefig(out / "Fig3b_pareto_fixed.png"); plt.close(fig)
        print("[Fig3b] 저장")

    # ---------- Fig 6 : 하방 위험 점도표 ----------
    show = [m for m in ["adapter_input", "adapter_latent", "adapter_isca", "full_ft"]
            if m in piv.columns]
    fig, ax = plt.subplots(figsize=(120 * MM, 72 * MM))
    for i, m in enumerate(show):
        d = (piv[m] - piv["source_only"]).dropna()
        ax.scatter(np.full(len(d), i) + np.random.default_rng(1).uniform(-.13, .13, len(d)),
                   d.values, s=18, alpha=.75,
                   color="#d62728" if m == "full_ft" else "#1f77b4", zorder=3)
        ax.hlines(d.median(), i - .28, i + .28, color="k", lw=1.6, zorder=4)
        ax.hlines(d.mean(), i - .28, i + .28, color="k", lw=1.0, ls=":", zorder=4)
    ax.axhline(0, color="0.5", lw=.8, ls="--")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([NICE.get(m, m).replace(" (", "\n(").replace(" + ", "\n+ ")
                        for m in show], fontsize=6.5)
    ax.set_ylabel(r"$\Delta$AUROC vs no adaptation (per pair)")
    ax.set_title("Per-pair adaptation gain: mean (dotted) vs median (solid)",
                 fontsize=8)
    fig.savefig(out / "Fig6_reliability.png"); plt.close(fig)
    print("[Fig6] 저장")

    # ---------- 콘솔 ----------
    print("\n===== R2 신뢰성 (주 지표, 공통 pair) =====")
    print(r2[["method", "n_pairs", "mean", "median", "wins", "ties", "losses",
              "worst_drop", "total_loss", "p_wilcoxon"]].round(4).to_string(index=False))
    if not r3.empty:
        print("\n===== R3 Pareto (report mode) =====")
        print(r3[["method", "params", "mean", "median", "wins", "losses",
                  "worst_drop", "p_wilcoxon"]].round(4).to_string(index=False))
    print(f"\n[done] {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""make_paper_figures.py — data figures of the paper (Fig. 3–7) from the stored results in work/.

    python run.py make_paper_figures.py              # writes work/figures_paper/Fig3..Fig7.png
    python run.py make_paper_figures.py --out figs

No retraining. Aggregation follows the paper:
    Fig. 3  Boiler 6 pairs, S2 (3 seeds), means over pair-seed cells, IN / OUT of the pool span
    Fig. 4  E2 contamination curves, complete-case pairs (13; Boiler_3to2 has diverged full-FT cells)
    Fig. 5  pair-level ΔAUROC vs no adaptation, E2(0), 14 pairs x 15 seeds
    Fig. 6  capacity ladder, 12 report pairs, mean with 95% bootstrap CI and worst observed loss
    Fig. 7  Boiler test streams by temporal decile (seed 0): anomaly ratio and no-adaptation AUROC
Fig. 1 and Fig. 2 are schematic drawings made separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIELD = ("PB", "XB", "Sim")
NICE = {"source_only": "No adaptation", "adapter_latent": "Adapter (latent, 1%)",
        "adapter_isca": "Adapter + self-refine", "adapter_input": "Adapter (input)",
        "adapter_output": "Adapter (output)", "full_ft": "Full fine-tuning"}
COL = {"source_only": "#7f7f7f", "adapter_latent": "#1f77b4", "adapter_isca": "#2ca02c",
       "adapter_input": "#9467bd", "adapter_output": "#ff7f0e", "full_ft": "#d62728"}
CURVE = ["source_only", "adapter_latent", "adapter_isca", "full_ft"]
plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "savefig.bbox": "tight"})


def e2(d: pd.DataFrame) -> pd.Series:
    """E2(0) with the paper's fill rule: only zero-removal pairs fall back to E0."""
    zero = d.groupby("pair")["frac_removed_temporal"].transform("first") == 0
    return pd.Series(np.where(zero & d["auroc_nopool_temporal"].isna(), d["auroc"],
                              d["auroc_nopool_temporal"]), index=d.index)


def fig3(out):
    m = pd.read_csv("mechanism_merged.csv", low_memory=False)
    m = m[m.pair.str.startswith("Boiler") & (m.stage == "S2")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for meth in CURVE:
        g = m[m.method == meth].groupby("contam")
        for k, col in enumerate(["auroc_in", "auroc_out"]):
            y = g[col].mean()
            ax[k].plot(y.index * 100, y.values, "o-", ms=3, color=COL[meth], label=NICE[meth])
            ax[k].annotate(f"{round(y.iloc[-1], 3) - round(y.iloc[0], 3):+.3f}", (y.index[-1] * 100, y.iloc[-1]),
                           xytext=(4, 0), textcoords="offset points", fontsize=6, color=COL[meth], va="center")
    for k, t in enumerate(["(a) IN: windows overlapping the pool", "(b) OUT: windows disjoint from the pool"]):
        ax[k].set_title(t, fontsize=8); ax[k].set_xlabel("Anomaly injection level ρ (% of clean windows)")
    ax[0].set_ylabel("AUROC"); ax[0].legend(frameon=False, fontsize=6)
    fig.savefig(out / "Fig3_in_out.png"); plt.close(fig)


def fig4(out):
    d = pd.read_csv("results_rescored.csv", low_memory=False)
    d = d[(d.stage == "S2") & ~d.pair.str.startswith(FIELD)].copy()
    d["m"] = e2(d)
    per = d.pivot_table(index="pair", columns=["method", "contam"], values="m", aggfunc="mean")
    cc = per.dropna().index
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for meth in CURVE:
        y = per[meth].loc[cc].mean()
        ax.plot(y.index * 100, y.values, "o-", ms=3, color=COL[meth], label=NICE[meth])
    ax.set_xlabel("Anomaly injection level ρ (% of clean windows)"); ax.set_ylabel("AUROC (leakage-controlled)")
    ax.text(0.98, 0.97, f"n = {len(cc)} pairs (complete case)", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower center")
    fig.savefig(out / "Fig4_contamination_controlled.png"); plt.close(fig)


def fig5(out):
    d = pd.read_csv("core_results_seed15/results_rescored.csv", low_memory=False)
    d = d[(d.stage == "S1") & ~d.pair.str.startswith(FIELD)].copy()
    d["m"] = e2(d)
    pm = d.groupby(["pair", "method"])["m"].mean().unstack()
    g = pm.sub(pm["source_only"], axis=0)
    meths = ["adapter_input", "adapter_isca", "adapter_latent", "adapter_output", "full_ft"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    rng = np.random.default_rng(0)
    for i, meth in enumerate(meths):
        y = g[meth].values
        ax.scatter(i + rng.uniform(-0.12, 0.12, y.size), y, s=9, color=COL[meth], alpha=0.85)
        ax.hlines(np.median(y), i - 0.25, i + 0.25, color="k", lw=1.2)
        ax.hlines(y.mean(), i - 0.25, i + 0.25, color="k", lw=0.8, ls=":")
        ax.annotate(f"worst {y.min():+.3f}", (i, y.min()), xytext=(0, -9), textcoords="offset points",
                    ha="center", fontsize=6)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(range(len(meths)), [NICE[m].replace(" (", "\n(") for m in meths], fontsize=6)
    ax.set_ylabel("ΔAUROC vs no adaptation (per pair)")
    ax.set_title("solid = median, dotted = mean, n = 14 pairs, 15 seeds", fontsize=7)
    fig.savefig(out / "Fig5_pair_gains.png"); plt.close(fig)


def fig6(out):
    r = pd.read_csv("tables_final/R3_pareto_report.csv")
    r = r[r.subset == "sweep"].sort_values("params")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    y = np.arange(len(r))
    ax.errorbar(r["mean"], y, xerr=[r["mean"] - r["ci_lo"], r["ci_hi"] - r["mean"]], fmt="o",
                color="#1f77b4", ms=3, capsize=2, label="mean ΔAUROC [95% bootstrap CI]")
    ax.scatter(r["worst_drop"], y, marker="v", color="#d62728", s=14, label="worst observed loss (12 pairs)")
    for yy, (_, row) in zip(y, r.iterrows()):
        ax.text(1.0, yy, f"{int(row.wins)}/{int(row.losses)}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6)
    ax.set_yticks(y, [f"{m.split('_')[0]} ({int(p):,})" for m, p in zip(r.method, r.params)], fontsize=6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("ΔAUROC vs no adaptation (leakage-controlled, 12 pairs)")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(out / "Fig6_capacity.png"); plt.close(fig)


def fig7(out):
    d = pd.read_csv("diagnose_out/D2_auroc_by_time_decile.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    tg = {"1": ["Boiler_2to1", "Boiler_3to1"], "2": ["Boiler_1to2", "Boiler_3to2"], "3": ["Boiler_1to3", "Boiler_2to3"]}
    w = 0.27
    for k, (t, ps) in enumerate(tg.items()):
        r = d[d.pair == ps[0]].sort_values("decile")
        ax[0].bar(r.decile + (k - 1) * w, r.anom_ratio * 100, width=w, label=f"target {t} ({', '.join(p[7:] for p in ps)})")
    ax[0].set_xlabel("Decile of the target test stream (1 = earliest)"); ax[0].set_ylabel("Window-level anomaly ratio (%)")
    ax[0].legend(frameon=False, fontsize=6); ax[0].set_title("(a) anomaly ratio per decile", fontsize=8)
    for p, r in d.groupby("pair"):
        r = r.sort_values("decile").dropna(subset=["auroc"])
        ax[1].plot(r.decile, r.auroc, "o-" if p[-1] in "12" else "o--", ms=2.5, label=p[7:])
    ax[1].set_xlabel("Decile of the target test stream"); ax[1].set_ylabel("AUROC within the decile")
    ax[1].legend(frameon=False, fontsize=6, ncol=2); ax[1].set_title("(b) no-adaptation AUROC per decile (seed 0)", fontsize=8)
    fig.savefig(out / "Fig7_boiler_deciles.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_paper")
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig3, fig4, fig5, fig6, fig7):
        f(out); print(f"[ok] {f.__name__}")
    print(f"[done] {out.resolve()}")


if __name__ == "__main__":
    main()

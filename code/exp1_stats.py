# -*- coding: utf-8 -*-
"""
stats_tests.py — Seed 5회 반복 통계 (확정: Baseline 대비 paired t-test)
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy import stats

REFERENCE_METHOD = "baseline"  # 확정: Baseline(Source-only) 대비 유의성 검정


def aggregate_mean_std(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """dataset(=domain pair) × method 별 'mean ± std' 표 (PPT용)."""
    rows = []
    for (ds, m), g in df.groupby(["pair", "method"]):
        row = {"pair": ds, "method": m, "n_seeds": len(g)}
        for met in metrics:
            v = g[met].astype(float).dropna()
            row[met] = (f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"
                        if len(v) > 1 else (f"{v.mean():.4f}" if len(v) else "nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair", "method"]).reset_index(drop=True)


def paired_t_vs_baseline(df: pd.DataFrame, metric: str,
                         reference: str = REFERENCE_METHOD) -> pd.DataFrame:
    """각 pair에서 reference(Baseline) vs 각 method, seed로 짝지은 t-test."""
    out = []
    for pair, g in df.groupby("pair"):
        pv = g.pivot_table(index="seed", columns="method", values=metric)
        if reference not in pv.columns:
            continue
        for m in [c for c in pv.columns if c != reference]:
            x, y = pv[m].dropna(), pv[reference].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 2:
                continue
            t, p = stats.ttest_rel(x[common], y[common])
            out.append({
                "pair": pair, "metric": metric, "method": m,
                "mean_method": float(x[common].mean()),
                "mean_baseline": float(y[common].mean()),
                "diff(method-baseline)": float((x[common] - y[common]).mean()),
                "t": float(t), "p_value": float(p),
                "significant(p<0.05)": bool(p < 0.05),
                "n_pairs": int(len(common)),
            })
    return pd.DataFrame(out)


def all_pairwise_t(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(옵션) 전 method 쌍 비교 — Full FT vs Adapter+MMD 동등성 논의 등에 사용."""
    out = []
    for pair, g in df.groupby("pair"):
        pv = g.pivot_table(index="seed", columns="method", values=metric)
        for a, b in itertools.combinations(pv.columns, 2):
            x, y = pv[a].dropna(), pv[b].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 2:
                continue
            t, p = stats.ttest_rel(x[common], y[common])
            out.append({"pair": pair, "metric": metric, "A": a, "B": b,
                        "diff(A-B)": float((x[common] - y[common]).mean()),
                        "t": float(t), "p_value": float(p)})
    return pd.DataFrame(out)

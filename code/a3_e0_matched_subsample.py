#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a3_e0_matched_subsample.py
E0 에서 시간 위치와 이상 기저율을 E2 에 맞춘 부분표본을 만들어 오염 기울기를 비교.

통제를 적용하면 평가 구간이 계열 후반부로 이동하고 이상 기저율도 함께 바뀐다.
따라서 E0 와 E2 의 기울기 차이에는 (i) 누수 제거 효과와 (ii) 평가 구간 이동
효과가 섞여 있다. E0 안에서 E2 와 같은 시간 분위 구성과 같은 이상 비율을 갖는
부분표본을 반복 추출해 기울기를 계산하면 두 효과를 분리할 수 있다.
E0-matched 가 E0 와 비슷하면 차이는 누수 제거에서 온 것이고, E2 와 비슷하면
구간 이동만으로도 설명된다.

[입력] window 단위 score CSV. 최소 열 구성:
    pair      : transfer pair 이름
    rho       : 주입 수준 (0, 0.05, ... 또는 0, 5, 20 처럼 % 값)
    start     : window 시작 index (시간 위치. 정렬 가능한 정수)
    label     : window 이상 여부 (1/0)
    score     : anomaly score (클수록 이상)
    in_e2     : E2(g) 에 포함되면 1, 아니면 0
  method 열이 있으면 --method 로 하나를 고른다. 없으면 전체를 하나로 본다.

[출력]
    <out>/a3_auroc_by_rho.csv   pair×rho×set 별 AUROC
    <out>/a3_slopes.csv         pair×set 별 오염 기울기와 집계
    <out>/a3_summary.md         원고용 표 초안

[실행 예]
    python a3_e0_matched_subsample.py --windows windows.csv --out results/
    python a3_e0_matched_subsample.py --windows windows.csv \
        --method adapter_latent --n-boot 200 --deciles 10 --out results/
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def matched_indices(df_e0, df_e2, deciles, rng):
    """E2 의 (시간 분위 × label) 구성 비율을 E0 안에서 그대로 재현한 표본."""
    lo, hi = df_e0["start"].min(), df_e0["start"].max()
    edges = np.linspace(lo, hi + 1e-9, deciles + 1)
    e0 = df_e0.assign(bin=np.digitize(df_e0["start"], edges[1:-1]))
    e2 = df_e2.assign(bin=np.digitize(df_e2["start"], edges[1:-1]))
    want = e2.groupby(["bin", "label"]).size()
    take = []
    for (b, y), n in want.items():
        pool = e0[(e0["bin"] == b) & (e0["label"] == y)]
        if len(pool) == 0:
            continue
        take.append(pool.sample(n=min(n, len(pool)), replace=len(pool) < n,
                                random_state=int(rng.integers(1 << 31))))
    return pd.concat(take) if take else e0.iloc[0:0]


def slope(rhos, aurocs):
    x = np.asarray(rhos, float); y = np.asarray(aurocs, float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan
    return stats.linregress(x[ok], y[ok]).slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--method", default=None)
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()

    df = pd.read_csv(a.windows)
    need = ["pair", "rho", "start", "label", "score", "in_e2"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[오류] 필요한 열이 없습니다: {miss}\n실제 열: {list(df.columns)}")
    if a.method and a.method_col in df.columns:
        df = df[df[a.method_col] == a.method]
    df["in_e2"] = df["in_e2"].astype(int)

    rows = []
    for (pair, rho), g in df.groupby(["pair", "rho"]):
        e0, e2 = g, g[g["in_e2"] == 1]
        rec = {"pair": pair, "rho": rho,
               "n_e0": len(e0), "n_e2": len(e2),
               "pos_rate_e0": e0["label"].mean(),
               "pos_rate_e2": e2["label"].mean() if len(e2) else np.nan,
               "auroc_e0": auroc(e0["label"], e0["score"]),
               "auroc_e2": auroc(e2["label"], e2["score"]) if len(e2) else np.nan}
        vals = []
        if len(e2) and e2["label"].nunique() > 1:
            for _ in range(a.n_boot):
                m = matched_indices(e0, e2, a.deciles, RNG)
                v = auroc(m["label"], m["score"])
                if not np.isnan(v):
                    vals.append(v)
        rec["auroc_e0_matched"] = np.mean(vals) if vals else np.nan
        rec["auroc_e0_matched_sd"] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        rows.append(rec)
    per = pd.DataFrame(rows).sort_values(["pair", "rho"])
    os.makedirs(a.out, exist_ok=True)
    per.to_csv(os.path.join(a.out, "a3_auroc_by_rho.csv"), index=False)

    sl = []
    for pair, g in per.groupby("pair"):
        sl.append({"pair": pair,
                   "slope_e0": slope(g["rho"], g["auroc_e0"]),
                   "slope_e0_matched": slope(g["rho"], g["auroc_e0_matched"]),
                   "slope_e2": slope(g["rho"], g["auroc_e2"])})
    slopes = pd.DataFrame(sl)
    slopes.to_csv(os.path.join(a.out, "a3_slopes.csv"), index=False)

    lines = ["| evaluation set | n pair | 평균 기울기 | median | Wilcoxon p |",
             "| --- | --- | --- | --- | --- |"]
    for col, name in [("slope_e0", "E0 (통제 없음)"),
                      ("slope_e0_matched", "E0-matched (시간 위치·기저율 정합)"),
                      ("slope_e2", "E2(g) (누수 통제)")]:
        x = slopes[col].dropna().values
        nz = x[x != 0]
        p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
        lines.append(f"| {name} | {len(x)} | {np.mean(x):+.4f} | {np.median(x):+.4f} | "
                     f"{'' if np.isnan(p) else f'{p:.3f}'} |")
    md = "\n".join(lines)
    with open(os.path.join(a.out, "a3_summary.md"), "w", encoding="utf-8") as f:
        f.write("# E0 matched subsample 기울기 비교\n\n" + md +
                "\n\n해석 지침: E0-matched 가 E0 에 가까우면 통제 전후 차이는 누수 제거에서 온 것이고, "
                "E2 에 가까우면 평가 구간 이동만으로 설명된다.\n")
    print(md)
    print(f"\n저장 위치: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

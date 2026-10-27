#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a4_ranking_region.py
AUROC 는 개선되는데 AUPR 은 개선되지 않는 이유를 직접 확인한다.

배경
    음성 window 가 다수인 조건에서는 오탐 수가 크게 늘어도 false positive rate
    변화가 작지만 precision 은 직접 영향을 받는다. 그래서 순위의 어느 구간이
    좋아졌는지에 따라 두 지표가 갈린다. 상위 구간이 좋아졌다면 AUPR 도 함께
    올라가야 하고, 중하위 구간에서만 좋아졌다면 AUROC 만 오른다.

계산 지표 (모두 E2(g) 구간에서, threshold 없이)
    AUROC              전체 순위 품질
    pAUROC(FPR<=0.1)   낮은 오경보 영역만 본 부분 면적(McClish 정규화)
    R-precision        상위 k개(k = 양성 수)에서의 precision
    recall@1%FAR       오경보율 1%에서의 재현율
    AUPR               precision-recall 면적

[입력] windows.csv  (dump_to_windows.py --method all 로 생성)
    pair, method, seed, rho, start, label, score, in_e2

[실행]
    python a4_ranking_region.py --windows windows.csv --out results_a4
    python a4_ranking_region.py --windows windows.csv --baseline source_only \\
        --method adapter_latent --rho 0 --out results_a4
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def p_auroc(y, s, max_fpr=0.1):
    """McClish 정규화 부분 AUROC. 값이 0.5면 무작위와 같다."""
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop < 2:
        return np.nan
    x, t = fpr[:stop], tpr[:stop]
    if x[-1] < max_fpr:                      # 경계까지 선형 보간
        t = np.append(t, np.interp(max_fpr, fpr, tpr)); x = np.append(x, max_fpr)
    area = np.trapezoid(t, x) if hasattr(np, "trapezoid") else np.trapz(t, x)
    min_a, max_a = 0.5 * max_fpr ** 2, max_fpr
    return 0.5 * (1 + (area - min_a) / (max_a - min_a))


def r_precision(y, s):
    k = int(np.sum(y))
    if k == 0 or k >= len(y):
        return np.nan
    idx = np.argsort(-np.asarray(s, float))[:k]
    return float(np.mean(np.asarray(y)[idx]))


def recall_at_far(y, s, far=0.01):
    y = np.asarray(y); s = np.asarray(s, float)
    neg = s[y == 0]
    if len(neg) == 0 or y.sum() == 0:
        return np.nan
    thr = np.quantile(neg, 1 - far)
    return float(np.mean(s[y == 1] >= thr))


def metrics(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return {}
    return {"AUROC": roc_auc_score(y, s),
            "pAUROC(FPR<=0.1)": p_auroc(y, s),
            "R-precision": r_precision(y, s),
            "recall@1%FAR": recall_at_far(y, s),
            "AUPR": average_precision_score(y, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default="results_a4")
    ap.add_argument("--baseline", default="source_only")
    ap.add_argument("--method", default="adapter_latent")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--all-windows", action="store_true",
                    help="E2(g) 로 제한하지 않고 전체 window 로 계산")
    a = ap.parse_args()

    df = pd.read_csv(a.windows)
    need = ["pair", "method", "label", "score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[오류] 필요한 열이 없습니다: {miss}\n실제 열: {list(df.columns)}")
    if "rho" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["rho"], errors="coerce"), a.rho)]
    if not a.all_windows and "in_e2" in df.columns:
        df = df[df["in_e2"].astype(int) == 1]
    have = set(df["method"].unique())
    for m in (a.baseline, a.method):
        if m not in have:
            sys.exit(f"[오류] method '{m}' 가 없습니다. 있는 값: {sorted(have)}")

    rows = []
    for (pair, method), g in df[df["method"].isin([a.baseline, a.method])].groupby(["pair", "method"]):
        per = []
        for seed, gs in (g.groupby("seed") if "seed" in g.columns else [(0, g)]):
            m = metrics(gs["label"], gs["score"])
            if m:
                per.append(m)
        if per:
            rec = pd.DataFrame(per).mean().to_dict()
            rec.update(pair=pair, method=method, n_seed=len(per))
            rows.append(rec)
    if not rows:
        sys.exit("[오류] 계산 가능한 pair 가 없습니다. label 이 한쪽만 있는지 확인하세요.")
    per_pair = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    per_pair.to_csv(os.path.join(a.out, "a4_per_pair.csv"), index=False)

    keys = ["AUROC", "pAUROC(FPR<=0.1)", "R-precision", "recall@1%FAR", "AUPR"]
    piv = per_pair.pivot_table(index="pair", columns="method", values=keys)
    lines = ["| 지표 | n pair | 평균 차이 | median | 개선/악화 | Wilcoxon p |",
             "| --- | --- | --- | --- | --- | --- |"]
    summ = []
    for k in keys:
        try:
            v = (piv[(k, a.method)] - piv[(k, a.baseline)]).dropna().values
        except KeyError:
            continue
        if len(v) < 3:
            continue
        p = wilcoxon(v).pvalue if np.any(v != 0) else np.nan
        summ.append({"metric": k, "n": len(v), "mean": v.mean(),
                     "median": np.median(v), "win": int((v > 1e-4).sum()),
                     "loss": int((v < -1e-4).sum()), "p": p})
        lines.append(f"| {k} | {len(v)} | {v.mean():+.4f} | {np.median(v):+.4f} | "
                     f"{int((v>1e-4).sum())}/{int((v<-1e-4).sum())} | "
                     f"{'' if np.isnan(p) else f'{p:.3f}'} |")
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "a4_summary.csv"), index=False)
    md = "\n".join(lines)
    with open(os.path.join(a.out, "a4_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# 순위 구간별 개선 ({a.method} - {a.baseline}, rho={a.rho})\n\n" + md +
                "\n\n해석: pAUROC 와 R-precision 이 함께 개선되면 상위 순위가 좋아진 것이고, "
                "AUROC 만 개선되면 중하위 구간의 재배열이며 오경보 부담은 줄지 않는다.\n")
    print(md)
    print(f"\n저장 위치: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

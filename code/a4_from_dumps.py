#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a4_from_dumps.py
window 점수 dump 에서 순위 구간별 지표를 바로 계산한다.

dump 파일에는 y, s, pair 만 있고 method·seed·contam 이 없다. 그래서 dump 로
계산한 전체 AUROC 를 results_rescored.csv 의 auroc 값과 대조해 어떤
(method, seed) 인지 역으로 찾는다. 누수 통제 마스크는 leakfix_masks.npz 에서
길이가 같은 key 를 찾아 적용한다.

계산 지표 (모두 E2(g) 구간)
    AUROC              전체 순위 품질
    pAUROC(FPR<=0.1)   낮은 오경보 영역만 본 부분 면적(McClish 정규화)
    R-precision        상위 k개(k = 양성 수)에서의 precision
    recall@1%FAR       오경보율 1%에서의 재현율
    AUPR               precision-recall 면적

[실행]
    python a4_from_dumps.py --results CLAUDE_REVIEW_20260806_110500\\core_results\\results_rescored.csv
    python a4_from_dumps.py --results ...\\results_rescored.csv --seeds 0 1 2 --out results_a4

[출력]
    <out>/a4_per_pair.csv   pair·method 별 지표
    <out>/a4_summary.md     adapter - source_only 차이와 검정 결과
    <out>/a4_matched.csv    dump 와 결과 행의 대조 기록(진단용)
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

DEFAULT_DUMPS = ["jobs_out/final/S1_main/scores_dump",
                 "jobs_out/final/BOILER6_S1/scores_dump",
                 "jobs_out/final/MSL_S1/scores_dump"]


def p_auroc(y, s, max_fpr=0.1):
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop < 2:
        return np.nan
    x, t = fpr[:stop], tpr[:stop]
    if x[-1] < max_fpr:
        t = np.append(t, np.interp(max_fpr, fpr, tpr)); x = np.append(x, max_fpr)
    area = np.trapezoid(t, x) if hasattr(np, "trapezoid") else np.trapz(t, x)
    lo, hi = 0.5 * max_fpr ** 2, max_fpr
    return 0.5 * (1 + (area - lo) / (hi - lo))


def r_precision(y, s):
    k = int(np.sum(y))
    if k == 0 or k >= len(y):
        return np.nan
    return float(np.mean(np.asarray(y)[np.argsort(-np.asarray(s, float))[:k]]))


def recall_at_far(y, s, far=0.01):
    y = np.asarray(y); s = np.asarray(s, float)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    thr = np.quantile(s[y == 0], 1 - far)
    return float(np.mean(s[y == 1] >= thr))


def metrics(y, s):
    if len(np.unique(y)) < 2:
        return None
    return {"AUROC": roc_auc_score(y, s), "pAUROC(FPR<=0.1)": p_auroc(y, s),
            "R-precision": r_precision(y, s), "recall@1%FAR": recall_at_far(y, s),
            "AUPR": average_precision_score(y, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_rescored.csv 경로")
    ap.add_argument("--dumps", nargs="*", default=None, help="scores_dump 폴더들")
    ap.add_argument("--masks", default="leakfix_masks.npz")
    ap.add_argument("--methods", nargs="*", default=["source_only", "adapter_latent"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--contam", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default="results_a4")
    a = ap.parse_args()

    # ---------- 결과 파일에서 대조표 만들기 ----------
    r = pd.read_csv(a.results, low_memory=False)
    if "stage" in r.columns:
        r = r[r["stage"].astype(str) == "S1"]
    if "contam" in r.columns:
        r = r[np.isclose(pd.to_numeric(r["contam"], errors="coerce"), a.contam)]
    r = r[r["method"].isin(a.methods)]
    if a.seeds:
        r = r[pd.to_numeric(r["seed"], errors="coerce").isin(a.seeds)]
    ctrl = next((c for c in ("auroc_nopool_temporal", "auroc_nopool")
                 if c in r.columns), None)
    cols = ["pair", "method", "seed", "auroc"] + ([ctrl] if ctrl else [])
    r = r[cols].drop_duplicates()
    r = r[r["auroc"].notna()]
    print(f"[대조 기준] 전체 AUROC 열 auroc, 통제 후 열 {ctrl}")
    print(f"[대조표] {len(r)}행, pair {r.pair.nunique()}개, method {sorted(r.method.unique())}, "
          f"seed {sorted(r.seed.unique())}")
    if r.empty:
        sys.exit("[오류] 조건에 맞는 결과 행이 없습니다. --seeds, --contam 을 확인하세요.")

    # ---------- 마스크 ----------
    masks = {}
    if os.path.exists(a.masks):
        z = np.load(a.masks, allow_pickle=True)
        for k in z.keys():
            arr = np.asarray(z[k]).ravel()
            masks.setdefault(len(arr), []).append((k, arr))
        print(f"[마스크] {a.masks} 에서 {sum(len(v) for v in masks.values())}개 "
              f"(길이 {sorted(masks)[:6]}...)")
    else:
        print(f"[경고] {a.masks} 가 없어 전체 window 로 계산합니다.")

    # ---------- dump 순회 ----------
    folders = a.dumps or [d for d in DEFAULT_DUMPS if os.path.isdir(d)]
    files = []
    for d in folders:
        files += sorted(glob.glob(os.path.join(d, "*.npz")))
    print(f"[dump] 폴더 {len(folders)}개, 파일 {len(files)}개")
    if not files:
        sys.exit("[오류] dump 파일을 찾지 못했습니다. --dumps 로 폴더를 지정하세요.")

    rows, matched, amb, nomatch, nomask = [], [], 0, 0, 0
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            y = np.asarray(z["y"]).ravel().astype(int)
            s = np.asarray(z["s"]).ravel().astype(float)
            pair = str(z["pair"]) if "pair" in z else None
        except Exception:
            continue
        if pair is None or len(np.unique(y)) < 2:
            continue
        cand = r[r["pair"] == pair]
        if cand.empty:
            continue
        mk = masks.get(len(y))
        if not mk:
            nomask += 1; continue
        keep = mk[0][1].astype(bool)
        au_all = roc_auc_score(y, s)
        au_keep = roc_auc_score(y[keep], s[keep]) if len(np.unique(y[keep])) > 1 else np.nan

        d = (cand["auroc"] - au_all).abs()
        if ctrl and not np.isnan(au_keep):
            d2 = (cand[ctrl] - au_keep).abs()
            d = d + d2.fillna(d)          # 통제 후 값이 없으면 전체 AUROC 만으로 비교
        hit = cand[d < a.tol * 2]
        if len(hit) == 0:
            nomatch += 1; continue
        row = hit.loc[d[hit.index].idxmin()]
        m = metrics(y[keep], s[keep])
        if m is None:
            continue
        m.update(pair=pair, method=row["method"], seed=int(row["seed"]),
                 n_eval=int(keep.sum()), file=os.path.basename(f),
                 dist=float(d[hit.index].min()))
        rows.append(m)
        matched.append({"file": os.path.basename(f), "pair": pair,
                        "method": row["method"], "seed": int(row["seed"]),
                        "auroc_dump": au_all, "auroc_result": row["auroc"],
                        "auroc_keep_dump": au_keep,
                        "auroc_keep_result": (row[ctrl] if ctrl else np.nan),
                        "dist": float(d[hit.index].min()), "mask_key": mk[0][0]})
    print(f"[대조] 성공 {len(rows)} / 중복 {amb} / 불일치 {nomatch} / 마스크 없음 {nomask}")
    if not rows:
        sys.exit("[오류] 대조된 dump 가 없습니다. --tol 을 키우거나 --seeds 를 넓혀 보세요.")

    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    # 같은 (pair, method, seed) 에 여러 dump 가 붙으면 가장 가까운 것만 남긴다
    n0 = len(df)
    df = df.sort_values("dist").drop_duplicates(subset=["pair", "method", "seed"], keep="first")
    print(f"[중복 정리] {n0} -> {len(df)}행 (pair·method·seed 당 1개)")
    cov = df.pivot_table(index="pair", columns="method", values="seed", aggfunc="count")
    print("\n[커버리지] pair 별 seed 수")
    print(cov.fillna(0).astype(int).to_string())
    both = cov.dropna()
    print(f"\n두 method 가 모두 있는 pair: {len(both)}개")
    pd.DataFrame(matched).to_csv(os.path.join(a.out, "a4_matched.csv"), index=False)
    keys = ["AUROC", "pAUROC(FPR<=0.1)", "R-precision", "recall@1%FAR", "AUPR"]
    per = df.groupby(["pair", "method"], as_index=False)[keys].mean()
    per["n_seed"] = df.groupby(["pair", "method"])["seed"].nunique().values
    per.to_csv(os.path.join(a.out, "a4_per_pair.csv"), index=False)
    print(f"\n[집계] pair {per.pair.nunique()}개")

    base, meth = a.methods[0], a.methods[-1]
    piv = per.pivot_table(index="pair", columns="method", values=keys)
    lines = [f"| 지표 | n pair | 평균 차이 | median | 개선/악화 | Wilcoxon p |",
             "| --- | --- | --- | --- | --- | --- |"]
    summ = []
    for k in keys:
        try:
            v = (piv[(k, meth)] - piv[(k, base)]).dropna().values
        except KeyError:
            continue
        if len(v) < 3:
            continue
        p = wilcoxon(v).pvalue if np.any(v != 0) else np.nan
        summ.append({"metric": k, "n": len(v), "mean": v.mean(), "median": float(np.median(v)),
                     "win": int((v > 1e-4).sum()), "loss": int((v < -1e-4).sum()), "p": p})
        lines.append(f"| {k} | {len(v)} | {v.mean():+.4f} | {np.median(v):+.4f} | "
                     f"{int((v>1e-4).sum())}/{int((v<-1e-4).sum())} | "
                     f"{'' if np.isnan(p) else f'{p:.3f}'} |")
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "a4_summary.csv"), index=False)
    md = "\n".join(lines)
    with open(os.path.join(a.out, "a4_summary.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# 순위 구간별 개선 ({meth} - {base}, E2(g), contam {a.contam})\n\n" + md +
                 "\n\n해석: pAUROC 와 R-precision 이 함께 개선되면 상위 순위가 좋아진 것이고, "
                 "AUROC 만 개선되면 중하위 구간의 재배열이다.\n")
    print("\n" + md)
    print(f"\n저장 위치: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

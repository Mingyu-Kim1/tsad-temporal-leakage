#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
final_60_report_all.py
----------------------
results_rescored.csv 를 받아 논문용 표를 "전부" 생성한다.

설계 원칙 (이 파일이 강제하는 것):
  1. 집계 단위는 항상 pair 다. seed / 중복 실행(outdir) 은 pair 안에서 먼저 평균낸다.
     -> Boiler_1to2 처럼 outdir 2개에서 돈 pair 가 2배 가중되는 문제를 제거한다.
  2. 제거율이 0인 pair 는 temporal == auroc 로 자동 채운다 (SMD/MSL).
  3. contamination 곡선은 complete-case 로만 만든다.
     한 method 라도 특정 rho 에서 값이 없으면 그 pair 는 그 method 의 곡선에서 통째로 뺀다.
     -> rho 마다 평균 대상 pair 집합이 달라지는 문제를 제거한다.
  4. 빠진 셀은 조용히 버리지 않고 divergence 표로 따로 보고한다.
  5. 지표 3종(auroc / nopool_exact / nopool_temporal)을 항상 나란히 낸다.
     하나만 고르지 않는다.

사용:
  python final_60_report_all.py --results results_rescored.csv --outdir tables_v1
  python final_60_report_all.py --results results_rescored.csv --outdir tables_v1 ^
      --primary-metric auroc_nopool_temporal --primary-method adapter_latent
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("scipy 가 필요합니다: pip install scipy")
    raise

METRICS = ["auroc", "auroc_nopool_exact", "auroc_nopool_temporal"]
BASELINE = "source_only"
ADAPT_METHODS = ["adapter_input", "adapter_output", "adapter_latent",
                 "adapter_isca", "full_ft"]


# ------------------------------------------------------------------ #
# 로드 및 정리
# ------------------------------------------------------------------ #
def load(path):
    d = pd.read_csv(path)
    for m in METRICS:
        if m not in d.columns:
            d[m] = np.nan
    return d


def fill_zero_removal(d, log):
    """제거율 0 인 pair 는 temporal 지표가 원 지표와 동일하다."""
    if "frac_removed_temporal" in d.columns:
        zero_pairs = (d.groupby("pair")["frac_removed_temporal"]
                       .max()
                       .pipe(lambda s: s[s.fillna(1.0) <= 1e-12].index.tolist()))
    else:
        zero_pairs = []

    # frac_removed_temporal 자체가 비어 있는 pair 는 rescore 가 못 붙인 것.
    # exact 제거율이 0 이면 temporal 도 0 이라고 볼 수 있는지는 확신할 수 없으므로
    # frac_removed(exact) == 0 인 pair 만 보수적으로 추가한다.
    if "frac_removed" in d.columns:
        cand = (d[d["frac_removed_temporal"].isna()]
                .groupby("pair")["frac_removed"].max())
        zero_pairs += [p for p, v in cand.items() if pd.notna(v) and v <= 1e-12]

    zero_pairs = sorted(set(zero_pairs))
    n_filled = 0
    for p in zero_pairs:
        m = (d.pair == p) & d.auroc_nopool_temporal.isna() & d.auroc.notna()
        n_filled += int(m.sum())
        d.loc[m, "auroc_nopool_temporal"] = d.loc[m, "auroc"]
        m2 = (d.pair == p) & d.auroc_nopool_exact.isna() & d.auroc.notna()
        d.loc[m2, "auroc_nopool_exact"] = d.loc[m2, "auroc"]
        d.loc[d.pair == p, "frac_removed_temporal"] = 0.0
        d.loc[d.pair == p, "evaluable_temporal"] = True
    log.append(f"[fill] 제거율 0 pair {len(zero_pairs)}개, {n_filled} cell 을 "
               f"temporal=auroc 로 채움: {', '.join(zero_pairs) if zero_pairs else '없음'}")
    return d


def evaluable_pairs(d, metric, log):
    """해당 지표로 채점 가능한 pair 목록."""
    if metric == "auroc_nopool_temporal":
        ok = (d.groupby("pair")[metric].apply(lambda s: s.notna().any()))
        pairs = sorted(ok[ok].index.tolist())
        dropped = sorted(set(d.pair.unique()) - set(pairs))
        log.append(f"[eval] {metric}: pair {len(pairs)}개 사용, "
                   f"{len(dropped)}개 제외 (평가 가능 window 0) -> {', '.join(dropped)}")
    else:
        ok = (d.groupby("pair")[metric].apply(lambda s: s.notna().any()))
        pairs = sorted(ok[ok].index.tolist())
    return pairs


# ------------------------------------------------------------------ #
# pair 단위 집계
# ------------------------------------------------------------------ #
def pair_table(d, stage, metric, index_extra=None):
    """seed / 중복실행을 pair 안에서 평균낸 뒤 pair x method 표를 만든다."""
    idx = ["pair"] + (index_extra or [])
    s = d[(d.stage == stage) & d[metric].notna()]
    if s.empty:
        return pd.DataFrame()
    return s.pivot_table(index=idx, columns="method", values=metric, aggfunc="mean")


def holm(pvals):
    """Holm-Bonferroni 보정. NaN 은 그대로 통과."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    n = len(vals)
    adj = np.empty(n)
    prev = 0.0
    for i, v in enumerate(vals):
        a = min(1.0, (n - i) * v)
        prev = max(prev, a)
        adj[i] = prev
    back = np.empty(n)
    back[idx] = adj
    out[ok] = back
    return out


def s1_delta(d, metric, primary_method):
    piv = pair_table(d, "S1", metric)
    if piv.empty or BASELINE not in piv.columns:
        return pd.DataFrame()
    base = piv[BASELINE]
    rows = []
    for meth in ADAPT_METHODS:
        if meth not in piv.columns:
            continue
        diff = (piv[meth] - base).dropna()
        if len(diff) == 0:
            continue
        try:
            p = wilcoxon(diff).pvalue if len(diff) >= 5 else np.nan
        except ValueError:
            p = np.nan
        rows.append(dict(metric=metric, method=meth, n_pairs=len(diff),
                         mean=diff.mean(), median=diff.median(), std=diff.std(),
                         wins=int((diff > 0).sum()), p_wilcoxon=p,
                         role="primary" if meth == primary_method else "ablation"))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # Holm 은 ablation 안에서만 적용. primary 는 사전 지정이므로 보정하지 않는다.
    ab = t.role == "ablation"
    t.loc[ab, "p_holm"] = holm(t.loc[ab, "p_wilcoxon"].values)
    return t


def s2_curve(d, metric, common_pairs=True):
    """complete-case contamination 곡선 + 누락 셀 보고.

    common_pairs=True 이면 method 전체가 공통으로 완전한 pair 집합만 쓴다.
    (한 그림 안에서 곡선마다 pair 집합이 다르면 비교가 성립하지 않는다)
    """
    s = d[(d.stage == "S2") & d[metric].notna()]
    if s.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    per = s.pivot_table(index="pair", columns=["method", "contam"],
                        values=metric, aggfunc="mean")

    keep = None
    if common_pairs:
        keep = set(per.dropna().index)

    curves, drops, missing = [], [], []
    for meth in sorted(s.method.unique()):
        if meth not in per.columns.get_level_values(0):
            continue
        sub = per[meth]
        complete = sub.dropna()
        incomplete = sub[sub.isna().any(axis=1)]
        if keep is not None:
            complete = complete.loc[[p for p in complete.index if p in keep]]
        for p, row in incomplete.iterrows():
            for c in row.index[row.isna()]:
                missing.append(dict(metric=metric, method=meth, pair=p, contam=c))
        if complete.empty:
            continue
        for c in complete.columns:
            curves.append(dict(metric=metric, method=meth, contam=c,
                               mean=complete[c].mean(), std=complete[c].std(),
                               n_pairs=len(complete)))
        if 0.0 in complete.columns:
            for c in complete.columns:
                if c == 0.0:
                    continue
                dd = complete[c] - complete[0.0]
                try:
                    p = wilcoxon(dd).pvalue if len(dd) >= 5 else np.nan
                except ValueError:
                    p = np.nan
                drops.append(dict(metric=metric, method=meth, contam=c,
                                  n_pairs=len(dd), mean_drop=dd.mean(),
                                  median_drop=dd.median(), p_wilcoxon=p))
    return (pd.DataFrame(curves), pd.DataFrame(drops), pd.DataFrame(missing))


def divergence_report(d):
    """실행됐어야 하는데 결과 행이 없는 셀 = 발산/실패."""
    s = d[d.stage == "S2"]
    if s.empty:
        return pd.DataFrame()
    rows = []
    for pair, g in s.groupby("pair"):
        seeds = sorted(g.seed.unique())
        contams = sorted(g.contam.dropna().unique())
        for meth in sorted(g.method.unique()):
            gm = g[g.method == meth]
            have = set(zip(gm.seed, gm.contam))
            want = {(sd, c) for sd in seeds for c in contams}
            miss = want - have
            if miss:
                rows.append(dict(pair=pair, method=meth,
                                 n_expected=len(want), n_missing=len(miss),
                                 missing_contam=", ".join(
                                     sorted({f"{c:g}" for _, c in miss}))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_rescored.csv")
    ap.add_argument("--outdir", default="tables_v1")
    ap.add_argument("--primary-metric", default="auroc_nopool_temporal")
    ap.add_argument("--primary-method", default="adapter_latent")
    ap.add_argument("--no-common-pairs", action="store_true",
                    help="S2 곡선에서 method 공통 pair 집합 강제를 끈다")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log = []

    d = load(args.results)
    log.append(f"[load] {args.results}: {len(d)} rows, "
               f"{d.pair.nunique()} pairs, stages={sorted(d.stage.unique())}")
    d = fill_zero_removal(d, log)

    for m in METRICS:
        evaluable_pairs(d, m, log)

    # --- S1 ---
    s1 = pd.concat([s1_delta(d, m, args.primary_method) for m in METRICS],
                   ignore_index=True)
    s1.to_csv(f"{args.outdir}/T_S1_delta.csv", index=False, encoding="utf-8-sig")

    for m in METRICS:
        pt = pair_table(d, "S1", m)
        if not pt.empty:
            pt.to_csv(f"{args.outdir}/T_S1_pairs__{m}.csv", encoding="utf-8-sig")

    # --- S2 ---
    cur, drp, mis = [], [], []
    for m in METRICS:
        a, b, c = s2_curve(d, m, common_pairs=not args.no_common_pairs)
        cur.append(a); drp.append(b); mis.append(c)
    pd.concat(cur, ignore_index=True).to_csv(
        f"{args.outdir}/T_S2_curve.csv", index=False, encoding="utf-8-sig")
    pd.concat(drp, ignore_index=True).to_csv(
        f"{args.outdir}/T_S2_drop.csv", index=False, encoding="utf-8-sig")
    pd.concat(mis, ignore_index=True).to_csv(
        f"{args.outdir}/T_S2_excluded_cells.csv", index=False, encoding="utf-8-sig")

    div = divergence_report(d)
    div.to_csv(f"{args.outdir}/T_divergence.csv", index=False, encoding="utf-8-sig")

    # --- 콘솔 요약 ---
    print("\n".join(log))
    print("\n===== S1: ΔAUROC vs source_only (pair 단위) =====")
    if not s1.empty:
        show = s1.copy()
        show["mean"] = show["mean"].round(4)
        show["median"] = show["median"].round(4)
        show["p_wilcoxon"] = show["p_wilcoxon"].round(4)
        print(show[["metric", "method", "n_pairs", "mean", "median",
                    "wins", "p_wilcoxon", "role"]].to_string(index=False))

    print(f"\n===== S2 곡선 ({args.primary_metric}, complete-case) =====")
    cc = pd.concat(cur, ignore_index=True)
    if not cc.empty:
        sel = cc[cc.metric == args.primary_metric]
        if not sel.empty:
            print(sel.pivot_table(index="contam", columns="method",
                                  values="mean").round(4).to_string())
            print("pair 수:", sel.n_pairs.min(), "~", sel.n_pairs.max())

    print("\n===== 발산/누락 셀 =====")
    print(div.to_string(index=False) if not div.empty else "없음")

    with open(f"{args.outdir}/RUN_LOG.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print(f"\n[done] {args.outdir}/ 에 표 저장 완료")


if __name__ == "__main__":
    sys.exit(main())

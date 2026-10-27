#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a2_detector_valid_subset.py
무적응 AUROC가 0.5 미만인 pair를 분리한 재집계.

통제 후 무적응 AUROC가 0.5 미만인 pair에서는 score 방향이 역전되어 있어
성능이 더 내려갈 여지가 작다. 이런 pair가 섞이면 오염 기울기와 적응 이득이
모두 축소되는 방향으로 편향될 수 있으므로, 전체 집계와 AUROC ≥ 0.5인
부분집합 집계를 나란히 제시한다.

[입력] pair 단위 CSV. 최소 열 구성:
    pair, method, gain            (a1 과 동일)
    baseline_auroc                (해당 pair의 무적응 AUROC, 통제 후 값)
  baseline_auroc 가 따로 없고 method 안에 무적응 행이 있으면
  --baseline-method none 처럼 지정하면 auroc 열에서 자동으로 뽑는다.
  이때는 auroc 열 이름을 --auroc-col 로 알려 준다.

[출력]
    <out>/a2_subset.csv     전체 / 부분집합 집계
    <out>/a2_excluded.csv   제외된 pair 목록과 그 값
    <out>/a2_summary.md     원고용 표 초안

[실행 예]
    python a2_detector_valid_subset.py --gains gains.csv --out results/
    python a2_detector_valid_subset.py --gains auroc_long.csv \
        --auroc-col auroc --baseline-method none --out results/
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)


def agg(x, label):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    nz = x[x != 0]
    bs = [np.mean(RNG.choice(x, len(x), replace=True)) for _ in range(10000)] if len(x) >= 3 else [np.nan]
    return {"subset": label, "n": len(x), "mean": np.mean(x) if len(x) else np.nan,
            "median": np.median(x) if len(x) else np.nan,
            "sem": stats.sem(x) if len(x) > 1 else np.nan,
            "win": int((x > 0).sum()), "tie": int((x == 0).sum()),
            "loss": int((x < 0).sum()),
            "worst": np.min(x) if len(x) else np.nan,
            "wilcoxon_p": stats.wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan,
            "ttest_p": stats.ttest_1samp(x, 0.0).pvalue if len(x) >= 3 else np.nan,
            "ci_lo": np.nanpercentile(bs, 2.5), "ci_hi": np.nanpercentile(bs, 97.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--pair-col", default="pair")
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--gain-col", default="gain")
    ap.add_argument("--baseline-col", default="baseline_auroc")
    ap.add_argument("--auroc-col", default=None)
    ap.add_argument("--baseline-method", default=None,
                    help="무적응 baseline 을 가리키는 method 값 (예: none, source_only)")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    df = pd.read_csv(a.gains).rename(columns={a.pair_col: "pair",
                                              a.method_col: "method",
                                              a.gain_col: "gain"})
    if a.baseline_method:
        if not a.auroc_col or a.auroc_col not in df.columns:
            sys.exit("[오류] --baseline-method 를 쓰려면 --auroc-col 이 필요합니다.")
        base = (df[df["method"] == a.baseline_method]
                .set_index("pair")[a.auroc_col].rename("baseline_auroc"))
        df = df[df["method"] != a.baseline_method].join(base, on="pair")
    elif a.baseline_col in df.columns:
        df = df.rename(columns={a.baseline_col: "baseline_auroc"})
    else:
        sys.exit(f"[오류] baseline AUROC 열을 찾지 못했습니다. 실제 열: {list(df.columns)}")

    if "gain" not in df.columns and a.auroc_col in df.columns:
        df["gain"] = df[a.auroc_col] - df["baseline_auroc"]

    bad = (df[["pair", "baseline_auroc"]].drop_duplicates()
           .query("baseline_auroc < @a.threshold").sort_values("baseline_auroc"))
    os.makedirs(a.out, exist_ok=True)
    bad.to_csv(os.path.join(a.out, "a2_excluded.csv"), index=False)

    rows = []
    for method, g in df.groupby("method"):
        rows.append({**agg(g["gain"].values, "all"), "method": method})
        keep = g[g["baseline_auroc"] >= a.threshold]
        rows.append({**agg(keep["gain"].values, f"baseline AUROC ≥ {a.threshold}"),
                     "method": method})
    res = pd.DataFrame(rows)[["method", "subset", "n", "mean", "median", "sem",
                              "win", "tie", "loss", "worst", "wilcoxon_p",
                              "ttest_p", "ci_lo", "ci_hi"]]
    res.to_csv(os.path.join(a.out, "a2_subset.csv"), index=False)

    f = lambda v, n=4: "" if pd.isna(v) else f"{v:.{n}f}"
    lines = ["| method | 집계 대상 | n | Mean | Median | W/T/L | 최대 손실 | Wilcoxon p | paired t p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in res.iterrows():
        lines.append(f"| {r['method']} | {r['subset']} | {int(r['n'])} | {f(r['mean'])} | "
                     f"{f(r['median'])} | {r['win']}/{r['tie']}/{r['loss']} | {f(r['worst'])} | "
                     f"{f(r['wilcoxon_p'],3)} | {f(r['ttest_p'],3)} |")
    md = "\n".join(lines)
    with open(os.path.join(a.out, "a2_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# 무적응 AUROC 기준 부분집합 재집계\n\n" + md +
                 "\n\n제외된 pair\n\n" + bad.to_markdown(index=False) + "\n")
    print(md)
    print("\n제외된 pair:")
    print(bad.to_string(index=False))
    print(f"\n저장 위치: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a0_prepare_gains.py
results_rescored.csv (실행 단위 long 포맷) -> pair 단위 이득 표 gains.csv

a1, a2 가 필요로 하는 열(pair, method, gain, baseline_auroc)을 만들어 준다.
기본 동작은 다음과 같다.
    - 지표: auroc_nopool_temporal (시간 구간 통제 후 AUROC)
    - 무적응 기준: method == source_only
    - 오염 수준: contam == 0
    - diverged 가 참인 행과 evaluable_temporal 이 거짓인 행은 제외
    - 같은 pair, method 안에서 seed 평균
    - gain = (method AUROC) - (source_only AUROC)

[먼저 확인할 때]
    python a0_prepare_gains.py --results results_rescored.csv --list
      -> 파일에 들어 있는 stage, method, contam, 사용 가능한 지표 열을 보여 준다.

[변환]
    python a0_prepare_gains.py --results results_rescored.csv --out gains.csv
    python a0_prepare_gains.py --results results_rescored.csv --out gains.csv --metric auroc_nopool_exact
    python a0_prepare_gains.py --results results_rescored.csv --out gains.csv --stage S1 --contam 0
"""
import argparse, sys
import numpy as np
import pandas as pd

METRIC_CANDIDATES = ["auroc_nopool_temporal", "auroc_nopool_exact",
                     "auroc_nopool", "auroc"]
TRUE = {"1", "true", "t", "yes", "y", "True", "TRUE"}


def truthy(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().isin(TRUE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="gains.csv")
    ap.add_argument("--metric", default=None, help="기본값 auroc_nopool_temporal")
    ap.add_argument("--baseline-method", default="source_only")
    ap.add_argument("--methods", default=None,
                    help="쉼표로 구분. 예: adapter_latent,full_ft (기본은 전부)")
    ap.add_argument("--contam", default="0", help="오염 수준. 'all' 이면 필터하지 않음")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--mode", default=None, help="예: report (tune 행 제외)")
    ap.add_argument("--family", default=None, help="예: bench")
    ap.add_argument("--keep-diverged", action="store_true")
    ap.add_argument("--keep-nonevaluable", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="필터를 하나씩 적용하며 남는 행 수를 보여 준다")
    ap.add_argument("--common-support", action="store_true",
                    help="요청한 method 가 모두 값을 가진 pair 만 남긴다(방법 간 비교용)")
    ap.add_argument("--strict", action="store_true",
                    help="stage, mode, family 값이 비어 있는 행도 제외한다")
    a = ap.parse_args()

    df = pd.read_csv(a.results, low_memory=False)
    cols = list(df.columns)

    if a.list:
        print(f"행 수: {len(df)}")
        for c in ("stage", "method", "contam", "family", "mode", "config"):
            if c in cols:
                v = sorted(map(str, df[c].dropna().unique()))
                print(f"\n{c} ({len(v)}종): {v[:25]}")
        print("\n사용 가능한 지표 열:",
              [c for c in METRIC_CANDIDATES if c in cols])
        print("pair 수:", df["pair"].nunique() if "pair" in cols else "pair 열 없음")
        return

    metric = a.metric or next((c for c in METRIC_CANDIDATES if c in cols), None)
    if metric is None or metric not in cols:
        sys.exit(f"[오류] 지표 열을 찾지 못했습니다. --metric 으로 지정하세요.\n열: {cols}")
    for need in ("pair", "method"):
        if need not in cols:
            sys.exit(f"[오류] '{need}' 열이 없습니다.")

    n0 = len(df)
    steps = []

    def keep(mask, label):
        """비어 있는 값은 통과시킨다(--strict 면 제외)."""
        nonlocal df
        before = len(df)
        df = df[mask]
        steps.append((label, before, len(df)))

    def eqfilter(col, val):
        if not val or col not in cols:
            return
        v = df[col].astype(str).str.strip()
        m = v == str(val)
        if not a.strict:
            m = m | df[col].isna() | (v == "") | (v.str.lower() == "nan")
        keep(m, f"{col} == {val}" + ("" if a.strict else " (빈 값 포함)"))

    eqfilter("stage", a.stage)
    eqfilter("mode", a.mode)
    eqfilter("family", a.family)
    if a.contam != "all" and "contam" in cols:
        keep(np.isclose(pd.to_numeric(df["contam"], errors="coerce"),
                        float(a.contam)), f"contam == {a.contam}")
    if not a.keep_diverged and "diverged" in cols:
        keep(~truthy(df["diverged"]), "diverged 제외")
    if not a.keep_nonevaluable and "evaluable_temporal" in cols \
            and metric.endswith("temporal"):
        keep(truthy(df["evaluable_temporal"]), "evaluable_temporal 만")
    keep(pd.to_numeric(df[metric], errors="coerce").notna(), f"{metric} 값 존재")

    if a.diagnose or df.empty:
        print(f"\n[진단] 원본 {n0}행 -> 단계별 잔여 행 수")
        for label, b, af in steps:
            print(f"  {label:<40} {b:>6} -> {af:>6}")
        if df.empty:
            print("\n  0행이 된 단계의 조건을 빼거나 값을 바꿔 보세요.")
            print("  값 확인: python a0_prepare_gains.py --results ... --list")
    print(f"필터 후 행 수: {len(df)} (원본 {n0})")
    if df.empty:
        sys.exit("[오류] 조건에 맞는 행이 없습니다. --list 로 값을 확인하세요.")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    cell = df.groupby(["pair", "method"], as_index=False)[metric].mean()

    base = cell[cell["method"] == a.baseline_method] \
        .set_index("pair")[metric].rename("baseline_auroc")
    if base.empty:
        sys.exit(f"[오류] baseline method '{a.baseline_method}' 행이 없습니다. "
                 f"있는 method: {sorted(cell['method'].unique())}")

    out = cell[cell["method"] != a.baseline_method].join(base, on="pair")
    out = out[out["baseline_auroc"].notna()]
    if a.methods:
        keep = [m.strip() for m in a.methods.split(",")]
        out = out[out["method"].isin(keep)]
    if a.common_support:
        n_m = out["method"].nunique()
        full = out.groupby("pair")["method"].nunique()
        keep_pairs = full[full == n_m].index
        dropped = sorted(set(out["pair"]) - set(keep_pairs))
        out = out[out["pair"].isin(keep_pairs)]
        if dropped:
            print(f"[common support] 일부 method 값이 없어 제외한 pair {len(dropped)}개: {dropped}")
    out["gain"] = out[metric] - out["baseline_auroc"]
    out = out.rename(columns={metric: "adapted_auroc"})[
        ["pair", "method", "adapted_auroc", "baseline_auroc", "gain"]]
    out.to_csv(a.out, index=False)

    print(f"\n지표: {metric} / baseline: {a.baseline_method} / contam: {a.contam}")
    print(f"pair 수: {out['pair'].nunique()},  method: {sorted(out['method'].unique())}")
    print(out.groupby("method")["gain"].agg(["count", "mean", "median", "min"]).round(4))
    print(f"\n저장: {a.out}")
    low = out[["pair", "baseline_auroc"]].drop_duplicates().query("baseline_auroc < 0.5")
    if len(low):
        print(f"\n무적응 AUROC < 0.5 인 pair {len(low)}개:")
        print(low.sort_values("baseline_auroc").to_string(index=False))


if __name__ == "__main__":
    main()

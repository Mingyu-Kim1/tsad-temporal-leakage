#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a1_target_sensitivity.py
target 단위 재집계 민감도 분석.

Boiler 6개 pair는 target이 같으면 test stream을 공유한다. pair를 독립 관측으로
보고 검정하면 유효 표본이 부풀려지므로, 같은 target에 속한 pair를 먼저 평균한
뒤 target 단위로 다시 검정한다. 같은 표에 pair 단위 결과와 target 단위 결과를
나란히 두고, Wilcoxon 외에 paired t-test와 target 단위 bootstrap 신뢰구간을
함께 보고한다.

[입력] pair 단위 적응 이득 CSV. 최소 열 구성:
    pair    : transfer pair 이름 (예: Boiler_2to1, SMD_1-2to1-1)
    method  : 비교 대상 이름 (예: adapter_latent, full_ft)
    gain    : 무적응 대비 ΔAUROC (부호 있는 값)
  선택 열:
    target  : target 설비 식별자. 없으면 pair 이름에서 자동 추출한다.
    seed    : 있으면 pair×method 단위로 먼저 평균한다.

[출력]
    <out>/a1_pair_level.csv     pair 단위 집계
    <out>/a1_target_level.csv   target 단위 집계
    <out>/a1_summary.md         원고에 넣을 표 초안

[실행 예]
    python a1_target_sensitivity.py --gains gains.csv --out results/
    python a1_target_sensitivity.py --gains gains.csv \
        --pair-col pair --method-col method --gain-col delta_auroc
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)


def infer_target(pair: str) -> str:
    """'Boiler_2to1' -> 'Boiler_1', 'SMD_1-2to1-1' -> 'SMD_1-1'."""
    m = re.match(r"^(?P<ds>[A-Za-z]+)[_\-](?P<rest>.+)$", str(pair))
    if not m:
        return str(pair)
    ds, rest = m.group("ds"), m.group("rest")
    m2 = re.search(r"to(?P<tgt>[\w\-\.]+)$", rest)
    return f"{ds}_{m2.group('tgt')}" if m2 else f"{ds}_{rest}"


def wtl(x, tol=1e-12):
    x = np.asarray(x, float)
    return int((x > tol).sum()), int((np.abs(x) <= tol).sum()), int((x < -tol).sum())


def tests(x):
    """부호 검정과 평균 검정을 함께 돌린다. 표본이 작으면 None 을 돌려준다."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    out = {"n": len(x), "mean": np.mean(x) if len(x) else np.nan,
           "median": np.median(x) if len(x) else np.nan,
           "sd": np.std(x, ddof=1) if len(x) > 1 else np.nan,
           "sem": stats.sem(x) if len(x) > 1 else np.nan}
    out["win"], out["tie"], out["loss"] = wtl(x)
    nz = x[x != 0]
    out["wilcoxon_p"] = (stats.wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan)
    out["ttest_p"] = (stats.ttest_1samp(x, 0.0).pvalue if len(x) >= 3 else np.nan)
    if len(x) >= 3:                      # 표본 재추출 신뢰구간
        bs = [np.mean(RNG.choice(x, len(x), replace=True)) for _ in range(10000)]
        out["ci_lo"], out["ci_hi"] = np.percentile(bs, [2.5, 97.5])
    else:
        out["ci_lo"] = out["ci_hi"] = np.nan
    return out


def fmt(v, n=4):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--pair-col", default="pair")
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--gain-col", default="gain")
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--seed-col", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.gains)
    need = [a.pair_col, a.method_col, a.gain_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[오류] 필요한 열이 없습니다: {miss}\n실제 열: {list(df.columns)}")
    df = df.rename(columns={a.pair_col: "pair", a.method_col: "method",
                            a.gain_col: "gain"})
    if a.seed_col and a.seed_col in df.columns:
        df = df.groupby(["pair", "method"], as_index=False)["gain"].mean()

    if a.target_col and a.target_col in df.columns:
        df["target"] = df[a.target_col]
    else:
        df["target"] = df["pair"].map(infer_target)

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "a1_pair_level.csv"), index=False)

    tgt = df.groupby(["method", "target"], as_index=False)["gain"].mean()
    tgt.to_csv(os.path.join(a.out, "a1_target_level.csv"), index=False)

    rows = []
    for method, g in df.groupby("method"):
        r = tests(g["gain"].values); r.update(method=method, unit="pair")
        rows.append(r)
        t = tests(tgt.loc[tgt["method"] == method, "gain"].values)
        t.update(method=method, unit="target")
        rows.append(t)
    res = pd.DataFrame(rows)[["method", "unit", "n", "mean", "median", "sem",
                              "win", "tie", "loss", "wilcoxon_p", "ttest_p",
                              "ci_lo", "ci_hi"]]
    res.to_csv(os.path.join(a.out, "a1_aggregate.csv"), index=False)

    lines = ["| method | 집계 단위 | n | Mean | Median | SEM | W/T/L | Wilcoxon p | paired t p | 95% CI |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in res.iterrows():
        lines.append("| {m} | {u} | {n} | {mn} | {md} | {se} | {w}/{t}/{l} | {wp} | {tp} | [{lo}, {hi}] |".format(
            m=r["method"], u="pair" if r["unit"] == "pair" else "target",
            n=int(r["n"]), mn=fmt(r["mean"]), md=fmt(r["median"]), se=fmt(r["sem"]),
            w=r["win"], t=r["tie"], l=r["loss"], wp=fmt(r["wilcoxon_p"], 3),
            tp=fmt(r["ttest_p"], 3), lo=fmt(r["ci_lo"], 3), hi=fmt(r["ci_hi"], 3)))
    md = "\n".join(lines)
    with open(os.path.join(a.out, "a1_summary.md"), "w", encoding="utf-8") as f:
        f.write("# target 단위 재집계 민감도\n\n" + md + "\n")
    print(md)
    print(f"\ntarget 수: {tgt.groupby('method')['target'].nunique().to_dict()}")
    print(f"저장 위치: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()

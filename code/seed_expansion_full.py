#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seed_expansion_full.py
14개 pair 전체를 seed 15개로 맞추고, 통제 지표까지 붙여 5.8절 근거를 다시 만든다.

현재 상태
    - results_rescored.csv : 14 pair x seed 0-4, 통제 지표(auroc_nopool_temporal) 있음
    - run_all_out_seed/results.csv : 7 pair x seed 0-14, 통제 지표 없음
    -> 통제 지표가 없는 pair 는 병합 과정에서 기존 값을 NaN 으로 덮어써 탈락한다.

이 스크립트가 하는 일
    plan    현재 seed 커버리지와 통제 지표 유무를 표로 보여 주고, 부족한 pair 와
            seed 를 채우는 실행 명령을 만들어 준다.
    run     그 명령을 실제로 실행한다(--execute 없이는 출력만 한다).
    merge   기존 결과와 새 결과를 seed_stability.py 와 동일한 규칙으로 합치고,
            14 pair x 15 seed 가 통제 지표를 갖췄는지 검사한다.
    decide  병합본으로 seed_stability.py decide 를 돌린다.

실행 순서
    python seed_expansion_full.py plan   --results CLAUDE_REVIEW_20260806_110500/core_results/results_rescored.csv --new run_all_out_seed/results.csv
    python seed_expansion_full.py run    --results ... --new ... --execute
    (여기서 rescoring 스크립트를 돌려 통제 지표를 붙인다. plan 이 후보를 찾아 준다)
    python seed_expansion_full.py merge  --results ... --new run_all_out_seed15/results_rescored.csv --out merged_seed15.csv
    python seed_expansion_full.py decide --merged merged_seed15.csv --out seed_decide_s15
"""
import argparse, glob, os, subprocess, sys
import numpy as np
import pandas as pd

METRIC = "auroc_nopool_temporal"
BASE = "source_only"
FOCUS = ["adapter_latent", "full_ft"]
ZERO_OVERLAP = ["SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2",
                "MSL_P-11toP-10", "MSL_P-10toP-11", "MSL_G-6toE-13"]
PAIRS14 = ["Boiler_1to2", "Boiler_1to3", "Boiler_2to1", "Boiler_2to3",
           "Boiler_3to1", "Boiler_3to2", "MSL_G-6toE-1", "MSL_G-6toE-13",
           "MSL_P-10toP-11", "MSL_P-11toP-10", "SMD_1-1to1-2", "SMD_1-2to1-1",
           "SMD_2-1to2-2", "SMD_3-1to3-2"]
DEDUP_KEY = ["stage", "pair", "seed", "method", "contam"]


# ------------------------------------------------------------------ 공통
def autofind(name="results_rescored.csv"):
    c = sorted(glob.glob(f"**/{name}", recursive=True))
    return c[0] if c else None


def read(path, label, required=False):
    if path and not os.path.exists(path) and "..." in str(path):
        print(f"[오류] 경로에 '...' 이 그대로 들어 있습니다: {path}")
        print("       실제 경로를 적어 주세요. 예: "
              "CLAUDE_REVIEW_20260806_110500\\core_results\\results_rescored.csv")
        sys.exit(1)
    if not path or not os.path.exists(path):
        alt = autofind() if "기존" in label else None
        if alt:
            print(f"[자동] {label} 경로를 찾지 못해 {alt} 를 사용합니다.")
            path = alt
        else:
            print(f"[없음] {label}: {path}")
            if required:
                sys.exit(f"[중단] {label} 파일이 필요합니다.")
            return None
    d = pd.read_csv(path, low_memory=False)
    print(f"[읽음] {label}: {path}  ({len(d)}행, {len(d.columns)}열, "
          f"{METRIC} {'있음' if METRIC in d.columns else '없음'})")
    return d


def s1_only(d):
    d = d[d.get("stage").astype(str) == "S1"] if "stage" in d.columns else d
    if "contam" in d.columns:
        d = d[np.isclose(pd.to_numeric(d["contam"], errors="coerce"), 0.0)]
    return d


def coverage(d, label):
    """pair 별 seed 수와 통제 지표 보유 여부."""
    if d is None or d.empty:
        return pd.DataFrame()
    x = s1_only(d)
    if x.empty:
        return pd.DataFrame()
    has = METRIC in x.columns
    rows = []
    for pair, g in x.groupby("pair"):
        seeds = sorted(pd.to_numeric(g["seed"], errors="coerce").dropna().astype(int).unique())
        n_metric = int(g[METRIC].notna().sum()) if has else 0
        rows.append({"pair": pair, "source": label, "n_seed": len(seeds),
                     "seed_max": max(seeds) if seeds else -1,
                     "metric_rows": n_metric,
                     "methods": g["method"].nunique()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plan
def cmd_plan(a):
    old = read(a.results, "기존 결과", required=True)
    new = read(a.new, "재실행 결과")
    cov = pd.concat([c for c in (coverage(old, "old"), coverage(new, "new"))
                     if not c.empty], ignore_index=True)
    if cov.empty:
        sys.exit("S1, contam 0 행을 찾지 못했습니다.")
    print("\n[1] pair 별 커버리지")
    piv = cov.pivot_table(index="pair", columns="source",
                          values=["n_seed", "metric_rows"], aggfunc="max").fillna(0)
    print(piv.astype(int).to_string())

    need_seed, need_metric = [], []
    for p in PAIRS14:
        n_new = cov.query("pair == @p and source == 'new'")["n_seed"]
        n_old = cov.query("pair == @p and source == 'old'")["n_seed"]
        n_have = int(max([0] + list(n_new) + list(n_old)))
        if n_have < a.target_seeds:
            need_seed.append(p)
        m_new = cov.query("pair == @p and source == 'new'")["metric_rows"]
        if len(m_new) and int(m_new.iloc[0]) == 0 and p not in ZERO_OVERLAP:
            need_metric.append(p)

    print(f"\n[2] seed {a.target_seeds}개에 못 미치는 pair {len(need_seed)}개")
    print("   ", need_seed if need_seed else "없음")
    print(f"\n[3] 재실행은 되었으나 통제 지표가 없는 pair {len(need_metric)}개 "
          f"(zero-overlap 은 auroc 로 대체 가능하므로 제외)")
    print("   ", need_metric if need_metric else "없음")

    seeds = " ".join(str(i) for i in range(a.target_seeds))
    if need_seed:
        pairs = " ".join(need_seed)
        print(f"\n[4] 부족한 seed 를 채우는 명령 (resume 이므로 기존 seed 는 건너뜁니다)")
        print(f"    python {a.runner} --stages S1 --pairs {pairs} "
              f"--seeds_main {seeds} --seeds {seeds} --outdir {a.outdir}")
    if need_metric:
        print(f"\n[5] 통제 지표가 필요한 pair 는 rescoring 을 거쳐야 합니다: {need_metric}")

    cands = sorted(set(glob.glob("**/*rescor*.py", recursive=True)
                       + glob.glob("**/*leakfix*.py", recursive=True)
                       + glob.glob("**/*nopool*.py", recursive=True)))
    print("\n[6] rescoring 스크립트 후보")
    print("   ", cands if cands else "찾지 못했습니다. 기존 results_rescored.csv 를 만든 "
                                     "스크립트를 확인하세요.")
    print("\n    rescoring 이 끝나면 병합 단계로 넘어갑니다:")
    print(f"    python seed_expansion_full.py merge --results {a.results} "
          f"--new {a.outdir}/results_rescored.csv --out merged_seed15.csv")


# ------------------------------------------------------------------ run
def cmd_run(a):
    old = read(a.results, "기존 결과", required=True)
    new = read(a.new, "재실행 결과")
    cov = pd.concat([c for c in (coverage(old, "old"), coverage(new, "new"))
                     if not c.empty], ignore_index=True)
    need = []
    for p in PAIRS14:
        n = cov.query("pair == @p")["n_seed"]
        if int(max([0] + list(n))) < a.target_seeds:
            need.append(p)
    if not need:
        print("모든 pair 가 이미 목표 seed 수를 채웠습니다.")
        return
    seeds = [str(i) for i in range(a.target_seeds)]
    cmd = [sys.executable, a.runner, "--stages", "S1", "--pairs", *need,
           "--seeds_main", *seeds, "--seeds", *seeds, "--outdir", a.outdir]
    print("\n실행할 명령:\n  " + " ".join(cmd))
    if not a.execute:
        print("\n--execute 를 붙이면 실제로 실행합니다.")
        print("실행 전에 pair 이름이 실험 코드에 정의되어 있는지 확인하세요:")
        print("  python inspect_pairs.py")
        return
    if not os.path.exists(a.runner):
        sys.exit(f"[오류] 실행 파일을 찾지 못했습니다: {a.runner}")
    r = subprocess.run(cmd)
    print(f"\n종료 코드 {r.returncode}")
    print(f"끝나면 {a.outdir}/results.csv 에 rescoring 을 적용한 뒤 merge 로 넘어가세요.")


# ------------------------------------------------------------------ merge
def cmd_merge(a):
    frames = []
    for path, label in [(a.results, "기존"), (a.new, "재실행")]:
        d = read(path, label, required=(label == "기존"))
        if d is not None:
            frames.append(d)
    if not frames:
        sys.exit("합칠 파일이 없습니다.")
    d = pd.concat(frames, ignore_index=True)

    if METRIC in d.columns and "auroc" in d.columns:
        m = d["pair"].isin(ZERO_OVERLAP) & d[METRIC].isna() & d["auroc"].notna()
        d.loc[m, METRIC] = d.loc[m, "auroc"]
        print(f"[fill] zero-overlap pair 의 빈 {METRIC} 를 auroc 로 채움: {int(m.sum())}행")

    key = [c for c in DEDUP_KEY if c in d.columns]
    # 통제 지표가 있는 행을 뒤로 보내 keep='last' 가 유효한 값을 남기도록 한다
    d["_has_metric"] = d[METRIC].notna().astype(int) if METRIC in d.columns else 0
    d = d.sort_values("_has_metric")
    n0 = len(d)
    d = d.drop_duplicates(subset=key, keep="last").drop(columns=["_has_metric"])
    print(f"[dedup] {n0} -> {len(d)}행 (키 {key}, 통제 지표가 있는 행 우선 보존)")

    x = s1_only(d)
    x = x[x[METRIC].notna()] if METRIC in x.columns else x
    print("\n[검사] pair 별 seed 수 (S1, contam 0, 통제 지표 보유 기준)")
    chk = (x.groupby("pair")["seed"].nunique().reindex(PAIRS14).fillna(0).astype(int))
    print(chk.to_string())
    bad = chk[chk < a.target_seeds]
    print(f"\n목표 {a.target_seeds} seed 미달 pair: "
          f"{list(bad.index) if len(bad) else '없음'}")
    miss = [p for p in PAIRS14 if p not in set(x['pair'])]
    if miss:
        print(f"아예 빠진 pair: {miss}")

    d.to_csv(a.out, index=False)
    print(f"\n저장: {a.out}")
    if len(bad) == 0 and not miss:
        print("14개 pair 전부 목표 seed 를 채웠습니다. decide 로 넘어가세요.")
        print(f"  python seed_expansion_full.py decide --merged {a.out} --out seed_decide_s15")


# ------------------------------------------------------------------ decide
def cmd_decide(a):
    if not os.path.exists(a.seed_stability):
        sys.exit(f"[오류] {a.seed_stability} 를 찾지 못했습니다.")
    cmd = [sys.executable, a.seed_stability, "decide", "--results", a.merged,
           "--new", a.merged, "--split", str(a.split), "--out", a.out]
    print("실행:\n  " + " ".join(cmd))
    r = subprocess.run(cmd)
    print(f"\n종료 코드 {r.returncode}")
    if r.returncode == 0:
        print(f"{a.out}/E1_headline_before_after.csv 와 E3_aggregate_sem.csv 를 확인하세요.")
        print("before 행이 표 8(adapter 0.0369, worst -0.0088, 11/1/2)과 일치하면 "
              "확대 전후 비교가 표 8과 같은 기반 위에서 이루어진 것입니다.")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("plan", cmd_plan), ("run", cmd_run)):
        p = sp.add_parser(name)
        p.add_argument("--results", required=True)
        p.add_argument("--new", default="run_all_out_seed/results.csv")
        p.add_argument("--runner", default="run_all_xjtu_fixed.py")
        p.add_argument("--outdir", default="run_all_out_seed15")
        p.add_argument("--target-seeds", type=int, default=15)
        if name == "run":
            p.add_argument("--execute", action="store_true")
        p.set_defaults(func=fn)

    p = sp.add_parser("merge")
    p.add_argument("--results", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", default="merged_seed15.csv")
    p.add_argument("--target-seeds", type=int, default=15)
    p.set_defaults(func=cmd_merge)

    p = sp.add_parser("decide")
    p.add_argument("--merged", required=True)
    p.add_argument("--seed-stability", default="seed_stability.py")
    p.add_argument("--split", type=int, default=5)
    p.add_argument("--out", default="seed_decide_s15")
    p.set_defaults(func=cmd_decide)

    a = ap.parse_args()
    a.func(a)

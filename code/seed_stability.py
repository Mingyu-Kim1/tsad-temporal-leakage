# -*- coding: utf-8 -*-
"""
seed_stability.py
=================
목적
----
논문 4.5절에서 보고한 "독립 실행 사이 최대 편차 5.7x10^-2"가
(1) 어느 cell 에서 발생하는지,
(2) seed 수를 늘리면 주 결론이 안정되는지
를 판정한다.

핵심 논리
---------
문제가 되는 것은 개별 실행의 산포(SD) 자체가 아니라, 그 산포가
pair 단위 평균 추정치를 얼마나 흔드느냐이다. seed 를 S 개 쓰면
pair 단위 평균의 표준오차는 SEM = SD / sqrt(S) 로 줄어든다.
따라서 SD 가 커도 S 를 늘려 SEM 을 효과 크기(제안 배치 평균 이득 0.037)
보다 충분히 작게 만들 수 있으면 결론은 유지된다. 이 스크립트는
그 S 를 데이터로부터 역산하고, 실제로 seed 를 늘린 뒤 결론이
바뀌지 않았는지 확인한다.

사용법
------
# 1단계. 지금 있는 결과만으로 진단 (재학습 없음)
python seed_stability.py diagnose --results results_rescored.csv --out seed_diag

# 2단계. 재실행 후 판정 (아래 run_seed_expansion.sh 실행 뒤)
python seed_stability.py decide --results results_rescored.csv \
       --new run_all_out_seed/results.csv --out seed_decide
"""
import argparse, itertools, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ------------------------------------------------------------------ 설정
METRIC = "auroc_nopool_temporal"          # 논문 주 지표(시간 구간 통제)
BASE = "source_only"
FOCUS = ["adapter_latent", "full_ft"]     # 주 비교 대상
EFFECT = 0.037                            # 제안 배치 평균 이득 (기준 효과 크기)
SEM_TARGET = 0.010                         # 목표: pair 평균의 SEM <= 0.010
ZERO_OVERLAP = ["SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2",
                "MSL_P-11toP-10", "MSL_P-10toP-11", "MSL_G-6toE-13"]


def load(paths):
    """여러 결과 파일을 합치고 중복 (stage,pair,seed,method,contam) 은 뒤엣것을 남긴다."""
    dfs = []
    for p in paths:
        if p and os.path.exists(p):
            dfs.append(pd.read_csv(p))
            print(f"[load] {p}: {len(dfs[-1])} rows")
    if not dfs:
        sys.exit("결과 파일을 찾지 못했습니다.")
    d = pd.concat(dfs, ignore_index=True)
    # 중복이 없는 데이터셋은 temporal 열이 비어 있으므로 통제 전 값으로 채운다
    m = d.pair.isin(ZERO_OVERLAP) & d[METRIC].isna() & d["auroc"].notna()
    d.loc[m, METRIC] = d.loc[m, "auroc"]
    key = ["stage", "pair", "seed", "method", "contam"]
    d = d.drop_duplicates(subset=key, keep="last")
    return d


def s1(d):
    return d[(d.stage == "S1") & d[METRIC].notna()].copy()


# ------------------------------------------------------------------ 진단
def cell_stats(x):
    """cell = (pair, method). seed 간 산포와, 그 산포가 만드는 평균의 표준오차."""
    g = x.groupby(["pair", "method"])[METRIC]
    t = g.agg(n_seed="count", mean="mean", sd="std", vmin="min", vmax="max").reset_index()
    t["range"] = t["vmax"] - t["vmin"]
    t["max_dev"] = t[["vmax", "vmin"]].sub(t["mean"], axis=0).abs().max(axis=1)
    t["sem"] = t["sd"] / np.sqrt(t["n_seed"])
    # 현재 SD 가 유지된다고 볼 때 SEM_TARGET 을 만족하는 최소 seed 수
    t["n_needed"] = np.ceil((t["sd"] / SEM_TARGET) ** 2).replace([np.inf], np.nan)
    return t.sort_values("max_dev", ascending=False)


def pair_delta(x, methods=FOCUS, seeds=None):
    """seed 를 pair 안에서 먼저 평균한 뒤 baseline 대비 차이를 만든다(논문 4.4절 절차)."""
    if seeds is not None:
        x = x[x.seed.isin(seeds)]
    piv = x.pivot_table(index="pair", columns="method", values=METRIC, aggfunc="mean")
    if BASE not in piv:
        return pd.DataFrame()
    out = {}
    for m in methods:
        if m in piv:
            out[m] = (piv[m] - piv[BASE]).dropna()
    return pd.DataFrame(out)


def headline(dd, method):
    """논문이 보고하는 수치 묶음."""
    v = dd[method].dropna().values
    if len(v) < 3:
        return dict(n=len(v), mean=np.nan, median=np.nan, win=np.nan,
                    loss=np.nan, worst=np.nan, p=np.nan)
    try:
        p = wilcoxon(v).pvalue
    except Exception:
        p = np.nan
    return dict(n=len(v), mean=v.mean(), median=float(np.median(v)),
                win=int((v > 1e-4).sum()), loss=int((v < -1e-4).sum()),
                worst=v.min(), p=p)


def subsample_curve(x, method, reps=400, rng=None):
    """seed 를 S 개만 썼다면 결론이 어떻게 흔들렸을지를 재표집으로 본다."""
    rng = rng or np.random.default_rng(0)
    seeds = sorted(x.seed.unique())
    rows = []
    for S in range(2, len(seeds) + 1):
        ms, ws, ps = [], [], []
        for _ in range(reps):
            sub = list(rng.choice(seeds, size=S, replace=False))
            dd = pair_delta(x, [method], seeds=sub)
            if dd.empty:
                continue
            h = headline(dd, method)
            ms.append(h["mean"]); ws.append(h["win"]); ps.append(h["p"])
        if not ms:
            continue
        ms = np.array(ms); ps = np.array(ps, dtype=float)
        rows.append(dict(method=method, n_seed=S,
                         mean_of_mean=ms.mean(), sd_of_mean=ms.std(ddof=1),
                         mean_lo=np.percentile(ms, 2.5), mean_hi=np.percentile(ms, 97.5),
                         sign_stable=float((ms > 0).mean()),
                         win_med=float(np.median(ws)),
                         p_lt05=float(np.nanmean(ps < 0.05))))
    return pd.DataFrame(rows)



def delta_by_seed(x, method):
    """seed 를 맞춰 baseline 을 뺀 pair x seed 단위 차이. 대응 설계이므로 산포가 작아진다."""
    p = x[x.method.isin([BASE, method])].pivot_table(
        index=["pair", "seed"], columns="method", values=METRIC, aggfunc="mean")
    if BASE not in p or method not in p:
        return pd.DataFrame()
    p = p.dropna(subset=[BASE, method])
    p["delta"] = p[method] - p[BASE]
    return p.reset_index()[["pair", "seed", "delta"]]


def aggregate_plan(x, method):
    """pair 단위 평균(논문의 주 수치)의 표준오차를 분해하고, 필요한 seed 수를 산출한다."""
    dl = delta_by_seed(x, method)
    if dl.empty:
        return pd.DataFrame(), {}
    g = dl.groupby("pair")["delta"].agg(n_seed="count", mean="mean", sd="std").reset_index()
    g["sem"] = g["sd"] / np.sqrt(g["n_seed"])
    g["n_needed_cell"] = np.ceil((g["sd"] / SEM_TARGET) ** 2)
    n_pair = len(g)
    # pair 평균의 표준오차 = sqrt(sum SEM_pair^2)/n_pair  (seed 잡음 성분)
    sem_agg = np.sqrt((g["sem"] ** 2).sum()) / n_pair
    # 모든 cell 에서 seed 를 S 개 쓸 때의 총합 표준오차
    sd_pool = np.sqrt((g["sd"] ** 2).sum()) / n_pair
    need_agg = int(np.ceil((sd_pool / (SEM_TARGET / 2)) ** 2)) if sd_pool > 0 else 0
    info = dict(method=method, n_pair=n_pair, sd_pool=sd_pool,
                sem_agg_now=sem_agg, n_seed_now=int(g["n_seed"].min()),
                target_agg=SEM_TARGET / 2, n_seed_needed_agg=need_agg,
                effect=EFFECT, ratio_now=sem_agg / EFFECT)
    return g.sort_values("sem", ascending=False), info


def cmd_diagnose(a):
    os.makedirs(a.out, exist_ok=True)
    d = load([a.results, a.new])
    x = s1(d)

    print(f"\n[1] S1 stage rows = {len(x)}, seeds = {sorted(x.seed.unique())}")

    cs = cell_stats(x)
    cs.round(4).to_csv(f"{a.out}/D1_cell_variability.csv", index=False)
    print("\n[2] seed 간 편차가 큰 cell 상위 10개")
    print(cs.head(10)[["pair", "method", "n_seed", "mean", "sd",
                       "max_dev", "sem", "n_needed"]].round(4).to_string(index=False))

    big = cs[cs.max_dev >= 0.05]
    print(f"\n    최대 편차 0.05 이상인 cell: {len(big)}개 / 전체 {len(cs)}개")
    if len(big):
        print("    해당 pair:", sorted(big.pair.unique()))
        print("    해당 method:", sorted(big.method.unique()))

    foc = cs[cs.method.isin(FOCUS + [BASE])]
    print("\n[3] 주 비교 대상 cell 의 현재 SEM 과 목표(0.010) 달성에 필요한 seed 수")
    print(foc.sort_values("sem", ascending=False).head(8)[
        ["pair", "method", "n_seed", "sd", "sem", "n_needed"]].round(4).to_string(index=False))
    need = foc["n_needed"].dropna()
    if len(need):
        print(f"\n    -> 주 비교 대상에서 SEM<={SEM_TARGET} 을 만족하려면 "
              f"cell 당 median {int(np.median(need))}개, 최대 {int(need.max())}개 seed 가 필요")

    print("\n[4] seed 수를 줄여서 재표집했을 때 주 결론이 흔들리는 정도")
    cur = []
    for m in FOCUS:
        c = subsample_curve(x[x.method.isin([BASE, m])], m, reps=a.reps)
        cur.append(c)
        if not c.empty:
            print(f"\n  -- {m}")
            print(c.round(4).to_string(index=False))
    if cur:
        pd.concat(cur).round(5).to_csv(f"{a.out}/D2_subsample_curve.csv", index=False)

    print("\n[5] pair 단위 평균의 표준오차 분해와 필요한 seed 수")
    plans, infos = [], []
    for m in FOCUS:
        g, info = aggregate_plan(x, m)
        if not len(g):
            continue
        g.insert(0, "method", m)
        plans.append(g); infos.append(info)
        print(f"\n  -- {m}")
        print(f"     pair {info['n_pair']}개, 현재 cell 당 seed {info['n_seed_now']}개")
        print(f"     pair 단위 평균의 표준오차 = {info['sem_agg_now']:.4f} "
              f"(효과 크기 {EFFECT} 의 {100*info['ratio_now']:.1f}%)")
        print(f"     목표 {info['target_agg']:.4f} 를 만족하려면 cell 당 seed {info['n_seed_needed_agg']}개")
        print("     seed 편차가 큰 pair 상위 5개:")
        print(g.head(5)[["pair", "n_seed", "sd", "sem", "n_needed_cell"]].round(4).to_string(index=False))
    if plans:
        pl = pd.concat(plans)
        pl.round(5).to_csv(f"{a.out}/D4_pair_sem_plan.csv", index=False)
        pd.DataFrame(infos).round(5).to_csv(f"{a.out}/D5_aggregate_plan.csv", index=False)
        rec = int(max(i["n_seed_needed_agg"] for i in infos))
        rec = int(min(max(rec, 10), 30))
        print(f"\n  => 권장: cell 당 seed {rec}개까지 확대. "
              f"현재 {infos[0]['n_seed_now']}개이므로 {rec - infos[0]['n_seed_now']}개를 추가 실행한다.")
        with open(f"{a.out}/D6_recommended_seeds.txt", "w") as f:
            f.write(" ".join(str(i) for i in range(rec)))
        print(f"     (seed 목록을 {a.out}/D6_recommended_seeds.txt 에 저장)")

    print("\n[6] 현재 seed 전부를 쓴 기준 수치")
    dd = pair_delta(x)
    base_rows = []
    for m in FOCUS:
        if m in dd:
            h = headline(dd, m); h["method"] = m; base_rows.append(h)
    br = pd.DataFrame(base_rows)
    br.round(4).to_csv(f"{a.out}/D3_baseline_headline.csv", index=False)
    print(br.round(4).to_string(index=False))

    print(f"\n[결과] {a.out}/ 에 D1~D3 저장. "
          f"D1 의 n_needed 열이 재실행에 필요한 seed 수의 근거이다.")


# ------------------------------------------------------------------ 판정
def cmd_decide(a):
    os.makedirs(a.out, exist_ok=True)
    d = load([a.results, a.new])
    x = s1(d)
    seeds = sorted(x.seed.unique())
    old = [s for s in seeds if s < a.split]
    new = seeds
    print(f"\n기존 seed {old} ({len(old)}개) vs 확대 seed {new} ({len(new)}개)")

    rows = []
    for tag, ss in [("before", old), ("after", new)]:
        dd = pair_delta(x, seeds=ss)
        for m in FOCUS:
            if m in dd:
                h = headline(dd, m); h.update(method=m, setting=tag, n_seed=len(ss))
                rows.append(h)
    t = pd.DataFrame(rows)[["setting", "method", "n_seed", "n", "mean", "median",
                            "win", "loss", "worst", "p"]]
    t.round(4).to_csv(f"{a.out}/E1_headline_before_after.csv", index=False)
    print("\n[1] 주 수치 비교")
    print(t.round(4).to_string(index=False))

    cs_old = cell_stats(x[x.seed.isin(old)])
    cs_new = cell_stats(x)
    mg = cs_old.merge(cs_new, on=["pair", "method"], suffixes=("_old", "_new"))
    mg["sem_drop"] = mg.sem_old - mg.sem_new
    mg.round(4).to_csv(f"{a.out}/E2_cell_sem_change.csv", index=False)
    foc = mg[mg.method.isin(FOCUS + [BASE])].sort_values("sem_new", ascending=False)
    print("\n[2] cell 별 SEM 변화 (주 비교 대상, 상위 8개)")
    print(foc.head(8)[["pair", "method", "n_seed_old", "sem_old",
                       "n_seed_new", "sem_new", "sem_drop"]].round(4).to_string(index=False))

    print("\n[3] pair 단위 평균의 표준오차 변화 (주 판정 기준)")
    agg_rows = []
    for tag, ss in [("before", old), ("after", new)]:
        xs = x[x.seed.isin(ss)]
        for m in FOCUS:
            g, info = aggregate_plan(xs, m)
            if info:
                info["setting"] = tag
                agg_rows.append(info)
    ag = pd.DataFrame(agg_rows)[["setting", "method", "n_seed_now", "sem_agg_now",
                                 "ratio_now", "n_seed_needed_agg"]]
    ag.round(4).to_csv(f"{a.out}/E3_aggregate_sem.csv", index=False)
    print(ag.round(4).to_string(index=False))

    print("\n[4] 판정")
    tgt = SEM_TARGET / 2
    aft = ag[ag.setting == "after"]
    ok_agg = bool((aft.sem_agg_now <= tgt).all())
    sign_ok, sig_ok = True, True
    for m in FOCUS:
        b = t[(t.method == m) & (t.setting == "before")]
        n_ = t[(t.method == m) & (t.setting == "after")]
        if len(b) and len(n_):
            sign_ok &= np.sign(b["mean"].iloc[0]) == np.sign(n_["mean"].iloc[0])
            if m == "adapter_latent":
                sig_ok &= bool(n_["p"].iloc[0] < 0.05)
    print(f"  - pair 평균의 표준오차 <= {tgt:.4f}: {'예' if ok_agg else '아니오'}")
    print(f"  - 평균 이득의 부호 유지: {'예' if sign_ok else '아니오'}")
    print(f"  - 제안 배치의 유의성(p<0.05) 유지: {'예' if sig_ok else '아니오'}")

    unstable = foc[foc.sem_new > SEM_TARGET]
    if ok_agg and sign_ok and sig_ok:
        print("\n  => seed 확대로 해결되었다. 본문에는 개별 실행의 최대 편차와 함께")
        print("     pair 단위 평균의 표준오차를 병기하고, 주 결론이 seed 수에")
        print("     의존하지 않음을 보고하면 된다.")
        if len(unstable):
            print(f"     다만 cell 단위로는 아직 {len(unstable)}개가 SEM>{SEM_TARGET} 이다:")
            print(unstable.head(5)[["pair", "method", "sem_new"]].round(4).to_string(index=False))
            print("     이들은 한계 절에 pair 단위 불안정성으로 따로 적는다.")
    elif not ok_agg:
        w = aft.sort_values("sem_agg_now", ascending=False).iloc[0]
        print(f"\n  => 아직 부족하다. {w['method']} 의 pair 평균 표준오차가 "
              f"{w['sem_agg_now']:.4f} 로 목표를 넘는다.")
        print(f"     cell 당 seed 를 {int(w['n_seed_needed_agg'])}개까지 늘려야 한다 "
              f"(현재 {int(w['n_seed_now'])}개).")
    else:
        print("\n  => seed 를 늘려도 부호나 유의성이 흔들린다. 이 경우는 seed 문제가")
        print("     아니라 해당 pair 자체의 불안정성이므로, 본문에 그대로 보고하고")
        print("     원인(예: pool 크기, 정규화 극단값)을 따로 분석해야 한다.")

    print(f"\n[결과] {a.out}/ 에 E1, E2 저장.")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("diagnose", help="재학습 없이 기존 결과만으로 진단")
    p1.add_argument("--results", default="results_rescored.csv")
    p1.add_argument("--new", default=None, help="추가 실행 결과(있으면 합침)")
    p1.add_argument("--out", default="seed_diag")
    p1.add_argument("--reps", type=int, default=400)
    p1.set_defaults(func=cmd_diagnose)

    p2 = sp.add_parser("decide", help="seed 확대 후 판정")
    p2.add_argument("--results", default="results_rescored.csv")
    p2.add_argument("--new", default="run_all_out_seed/results.csv")
    p2.add_argument("--out", default="seed_decide")
    p2.add_argument("--split", type=int, default=5,
                    help="이 값 미만이 기존 seed. 기본 5 (seed 0-4)")
    p2.set_defaults(func=cmd_decide)

    a = ap.parse_args()
    a.func(a)

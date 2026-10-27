#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s2_matched.py
S2(오염 주입) 실행의 window 점수로 E0 부분표본 기울기 비교를 수행한다.

질문
    통제 후(E2) 오염 기울기가 달라진 것이 누수 제거 때문인가, 평가 구간이
    계열 뒤쪽으로 이동하고 이상 비율이 바뀐 때문인가.

비교하는 네 가지 AUROC (같은 dump 에서 계산)
    E0          전체 평가 window (통제 없음)
    E2          시간 구간 통제 후 window
    E0-matched  E0 에서 E2 와 같은 (시간 10분위 x label) 구성으로 뽑은 표본
                -> 위치와 이상 비율은 E2 와 같고 중복 window 는 섞여 있음
    E0/E2-mix   중복과 비중복 window 가 함께 있는 10분위만 남긴 E0, E2
                -> 같은 시간 위치에서 중복 여부만 다름 (보조 지표)

해석
    E0-matched 기울기가 E0 에 가까우면  -> 차이는 누수 제거에서 온 것
    E0-matched 기울기가 E2 에 가까우면  -> 구간 이동만으로 설명됨
    매칭 표본의 중복 비율(leak_share)이 거의 0 이면 그 pair 는 분리 불가

마스크와 조건 매칭은 final_35_rescore.py 와 같은 방식을 쓴다.
    exact mask   : leakfix_masks.npz 에서 labels_hash(y) 로 조회
    temporal mask: exact mask 를 W-1 만큼 dilate (gap_mult 1.0)
    조건 매칭    : 같은 실행 폴더 results.csv 의 (n_eval_windows, auroc) 키

사용법
    1) 기존 S2 dump 로 가능한지 확인 (재실행 전 반드시)
       python s2_matched.py check --runs jobs_out\\final\\S2_main jobs_out\\final\\MSL_S2
    2) 부족하면 재실행 명령 출력
       python s2_matched.py plan
    3) 분석
       python s2_matched.py analyze --runs jobs_out\\final\\S2_dump_main jobs_out\\final\\S2_dump_boiler6 jobs_out\\final\\S2_dump_msl
"""
import argparse, glob, hashlib, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon
from sklearn.metrics import roc_auc_score

BENCH = ("Boiler", "SMD", "MSL", "SMAP")
PAIRS_MAIN = ["Boiler_1to2", "Boiler_2to3", "Boiler_3to1",
              "SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2"]
PAIRS_BOILER6 = ["Boiler_1to3", "Boiler_2to1", "Boiler_3to2"]
PAIRS_MSL = ["MSL_G-6toE-1", "MSL_G-6toE-13", "MSL_P-10toP-11", "MSL_P-11toP-10"]


# ------------------------------------------------------------ 공통 유틸
def lhash(y):
    y = np.ascontiguousarray(np.asarray(y).astype(np.int8))
    return "%d_%s" % (len(y), hashlib.blake2b(y.tobytes(), digest_size=12).hexdigest())


def dilate(mask, radius):
    if radius <= 0 or not mask.any():
        return mask.copy()
    n = mask.size
    diff = np.zeros(n + 1, np.int32)
    idx = np.flatnonzero(mask)
    np.add.at(diff, np.maximum(idx - radius, 0), 1)
    np.add.at(diff, np.minimum(idx + radius + 1, n), -1)
    return np.cumsum(diff[:-1]) > 0


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def load_masks(path):
    if not Path(path).exists():
        sys.exit(f"[오류] {path} 가 없습니다. 프로젝트 루트에서 실행하세요.")
    z = np.load(path, allow_pickle=True)
    return {k: np.asarray(z[k]).astype(bool) for k in z.files if k != "__meta__"}


def load_lut(run_dirs, stage, methods, seeds):
    """실행 폴더들의 results.csv 로 (n_eval, auroc) -> 조건 목록 대조표를 만든다."""
    frames = []
    for d in run_dirs:
        f = Path(d) / "results.csv"
        if f.exists():
            r = pd.read_csv(f, low_memory=False)
            r["_dir"] = str(d)
            frames.append(r)
        else:
            print(f"[경고] {f} 없음")
    if not frames:
        sys.exit("[오류] results.csv 를 하나도 찾지 못했습니다.")
    r = pd.concat(frames, ignore_index=True)
    need = ["pair", "method", "seed", "contam", "auroc", "n_eval_windows"]
    miss = [c for c in need if c not in r.columns]
    if miss:
        sys.exit(f"[오류] results.csv 에 열이 없습니다: {miss}")
    if stage and "stage" in r.columns:
        r = r[r["stage"].astype(str) == stage]
    r = r[r["method"].isin(methods)]
    if seeds:
        r = r[pd.to_numeric(r["seed"], errors="coerce").isin(seeds)]
    r = r.dropna(subset=["auroc", "n_eval_windows"])
    lut = defaultdict(list)
    for row in r.itertuples(index=False):
        key = (int(round(row.n_eval_windows)), round(float(row.auroc), 12))
        lut[key].append((row.pair, row.method, int(row.seed), float(row.contam)))
    return r, lut


def iter_dumps(run_dirs):
    for d in run_dirs:
        for f in sorted(glob.glob(str(Path(d) / "scores_dump" / "*.npz"))):
            yield f


def read_dump(f):
    z = np.load(f, allow_pickle=True)
    return (np.asarray(z["y"]).astype(int), np.asarray(z["s"]).astype(float),
            str(z["pair"]) if "pair" in z.files else None)


def resolve(lut, y, s):
    """dump 한 개를 조건 하나에 대응시킨다. 여러 조건에 걸리면 None."""
    key = (int(y.size), round(auroc(y, s), 12))
    hits = set(lut.get(key, []))
    if len(hits) == 1:
        return next(iter(hits)), "ok"
    return None, ("none" if not hits else "ambiguous")


# ------------------------------------------------------------ check
def cmd_check(a):
    masks = load_masks(a.masks)
    r, lut = load_lut(a.runs, a.stage, a.methods, a.seeds)
    print(f"[결과표] {len(r)}행 | contam {sorted(r.contam.unique())} | "
          f"method {sorted(r.method.unique())} | seed {sorted(r.seed.unique())}")
    stat = defaultdict(int)
    cov = defaultdict(set)
    nfile = 0
    for f in iter_dumps(a.runs):
        nfile += 1
        try:
            y, s, _ = read_dump(f)
        except Exception:
            stat["read_fail"] += 1
            continue
        if len(np.unique(y)) < 2:
            stat["one_class"] += 1
            continue
        cond, st = resolve(lut, y, s)
        stat[st] += 1
        if cond is None:
            continue
        if lhash(y) not in masks:
            stat["no_mask"] += 1
            continue
        pair, method, seed, contam = cond
        cov[(pair, method)].add((contam, seed))
    print(f"[dump] {nfile}개 | 매칭 {stat['ok']} | 결과표에 없음 {stat['none']} | "
          f"조건 중복 {stat['ambiguous']} | 마스크 없음 {stat['no_mask']} | "
          f"한쪽 class {stat['one_class']}")
    if not cov:
        print("\n사용 가능한 dump 가 없습니다. plan 으로 재실행 명령을 확인하세요.")
        return
    rows = []
    for (pair, method), cs in sorted(cov.items()):
        contams = sorted({c for c, _ in cs})
        rows.append({"pair": pair, "method": method, "n_contam": len(contams),
                     "contams": contams, "n_cells": len(cs)})
    t = pd.DataFrame(rows)
    print("\n[커버리지] pair x method 별 오염 수준 수")
    print(t.to_string(index=False))
    ok = t[t.n_contam >= 3]
    print(f"\n기울기 계산 가능(오염 수준 3개 이상): {ok.pair.nunique()}개 pair")
    print("판단: 중복이 있는 pair(Boiler 6개, MSL_G-6toE-1)가 모두 포함되면 재실행 없이 analyze 로 넘어가면 됩니다.")


# ------------------------------------------------------------ plan
def cmd_plan(a):
    seeds = " ".join(str(x) for x in a.seeds)
    out = a.prefix
    print("S2 를 score dump 와 함께 새 폴더에 다시 실행합니다(기존 결과는 건드리지 않음).")
    print("dump 저장은 final_20_run.py 기본값이라 별도 옵션이 필요 없습니다.\n")
    print(f"python final_20_run.py --stage S2 --pairs {' '.join(PAIRS_MAIN)} "
          f"--seeds {seeds} --seeds-main {seeds} --outdir {out}_main")
    print(f"python final_20_run.py --stage S2 --pairs {' '.join(PAIRS_BOILER6)} "
          f"--seeds {seeds} --seeds-main {seeds} --outdir {out}_boiler6 --register-boiler")
    print(f"python final_20_run.py --stage S2 --pairs {' '.join(PAIRS_MSL)} "
          f"--seeds {seeds} --seeds-main {seeds} --outdir {out}_msl")
    print("\n끝나면:")
    print(f"python s2_matched.py check --runs {out}_main {out}_boiler6 {out}_msl")
    print(f"python s2_matched.py analyze --runs {out}_main {out}_boiler6 {out}_msl")


# ------------------------------------------------------------ analyze
def matched_auroc(y, s, keep, n_boot, rng, n_bins=10):
    """E0 에서 E2 와 같은 (시간 분위 x label) 구성으로 표본을 뽑는다."""
    n = y.size
    bins = (np.arange(n) * n_bins) // n
    pool = {}
    for b in range(n_bins):
        for lab in (0, 1):
            pool[(b, lab)] = np.flatnonzero((bins == b) & (y == lab))
    want = {(b, lab): int(((bins == b) & (y == lab) & keep).sum())
            for b in range(n_bins) for lab in (0, 1)}
    vals, leak = [], []
    for _ in range(n_boot):
        take = []
        for k, m in want.items():
            if m == 0:
                continue
            src = pool[k]
            if src.size == 0:
                continue
            take.append(rng.choice(src, size=m, replace=src.size < m))
        if not take:
            break
        idx = np.concatenate(take)
        v = auroc(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
            leak.append(float((~keep[idx]).mean()))
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.mean(leak))


def mixed_auroc(y, s, keep, n_bins=10, min_pos=20):
    """중복과 비중복이 함께 있는 분위만 남겨 E0, E2 를 계산한다.

    반환: (E0_mix, E2_mix, 혼합 분위 수, 중복 window 중 혼합 분위에 있는 비율)
    혼합 분위가 2개 미만이거나 양성이 min_pos 미만이면 AUROC 는 NaN.
    """
    n = y.size
    bins = (np.arange(n) * n_bins) // n
    ov = ~keep
    mixed = [b for b in range(n_bins)
             if keep[bins == b].any() and ov[bins == b].any()]
    mix_cov = float(ov[np.isin(bins, mixed)].sum() / ov.sum()) if ov.any() else 0.0
    if len(mixed) < 2:
        return np.nan, np.nan, len(mixed), mix_cov
    sel = np.isin(bins, mixed)
    if y[sel].sum() < min_pos or y[sel & keep].sum() < min_pos:
        return np.nan, np.nan, len(mixed), mix_cov
    return auroc(y[sel], s[sel]), auroc(y[sel & keep], s[sel & keep]), len(mixed), mix_cov


def slope(x, v):
    x = np.asarray(x, float); v = np.asarray(v, float)
    ok = ~np.isnan(v)
    if ok.sum() < 3 or np.unique(x[ok]).size < 3:
        return np.nan
    return float(linregress(x[ok], v[ok]).slope)


def cmd_analyze(a):
    masks = load_masks(a.masks)
    r, lut = load_lut(a.runs, a.stage, a.methods, a.seeds)
    print(f"[결과표] {len(r)}행 | contam {sorted(r.contam.unique())}")
    rng = np.random.default_rng(a.seed)
    rows, stat = [], defaultdict(int)
    files = list(iter_dumps(a.runs))
    print(f"[dump] {len(files)}개 처리 시작 (n_boot={a.n_boot})")
    for i, f in enumerate(files, 1):
        try:
            y, s, _ = read_dump(f)
        except Exception:
            stat["read_fail"] += 1
            continue
        if len(np.unique(y)) < 2:
            stat["one_class"] += 1
            continue
        cond, st = resolve(lut, y, s)
        if cond is None:
            stat[st] += 1
            continue
        ke = masks.get(lhash(y))
        if ke is None or ke.size != y.size:
            stat["no_mask"] += 1
            continue
        pair, method, seed, contam = cond
        W = a.w_bench if pair.startswith(BENCH) else a.w_field
        keep = ~dilate(~ke, max(int(round(a.gap_mult * W)) - 1, 0))
        if len(np.unique(y[keep])) < 2:
            stat["not_evaluable"] += 1
            continue
        au_m, leak = matched_auroc(y, s, keep, a.n_boot, rng)
        au_x0, au_x2, n_mix, mix_cov = mixed_auroc(y, s, keep)
        rows.append({"pair": pair, "method": method, "seed": seed, "contam": contam,
                     "E0": auroc(y, s), "E2": auroc(y[keep], s[keep]),
                     "E0_matched": au_m, "leak_share": leak,
                     "E0_mix": au_x0, "E2_mix": au_x2, "n_mixed_bins": n_mix,
                     "mix_cov": mix_cov,
                     "n_E0": int(y.size), "n_E2": int(keep.sum()),
                     "overlap_frac": float((~keep).mean()), "file": Path(f).name})
        stat["ok"] += 1
        if i % 200 == 0:
            print(f"  {i}/{len(files)}")
    print(f"[처리] 사용 {stat['ok']} | 결과표에 없음 {stat['none']} | 조건 중복 {stat['ambiguous']} | "
          f"마스크 없음 {stat['no_mask']} | 통제 후 채점 불가 {stat['not_evaluable']}")
    if not rows:
        sys.exit("[오류] 분석할 dump 가 없습니다. check 결과를 먼저 확인하세요.")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates(subset=["pair", "method", "seed", "contam"])
    df.to_csv(out / "s2_cells.csv", index=False)

    sets = ["E0", "E2", "E0_matched", "E0_mix", "E2_mix"]
    cell = df.groupby(["pair", "method", "contam"], as_index=False)[
        sets + ["leak_share", "overlap_frac", "mix_cov"]].mean()
    cell.to_csv(out / "s2_by_contam.csv", index=False)

    sl = []
    for (pair, method), g in cell.groupby(["pair", "method"]):
        rec = {"pair": pair, "method": method, "n_contam": g.contam.nunique(),
               "overlap_frac": g.overlap_frac.mean(), "mix_cov": g.mix_cov.mean(),
               "leak_share": g.leak_share.mean()}
        for k in sets:
            rec["slope_" + k] = slope(g.contam, g[k])
        sl.append(rec)
    sl = pd.DataFrame(sl)
    # 중복 window 의 상당 부분이 비중복과 같은 시간 분위에 섞여 있어야
    # 위치와 누수를 떼어 볼 수 있다. 몰려 있으면 두 효과가 같은 구간에 겹친다.
    sl["separable"] = (sl["overlap_frac"] > 0) & (sl["mix_cov"] >= a.min_mix) \
                      & (sl["leak_share"] >= a.min_leak)
    sl.to_csv(out / "s2_slopes.csv", index=False)

    lines = [f"# E0 부분표본 기울기 비교 (S2, gap_mult {a.gap_mult})", ""]
    for method, g in sl.groupby("method"):
        g = g[g["overlap_frac"] > 0]           # 중복이 있는 pair 만
        lines.append(f"## {method}  (중복 있는 pair {len(g)}개, 분리 가능 {int(g.separable.sum())}개)")
        lines.append("")
        lines.append("| 집합 | 평균 기울기 | median |")
        lines.append("| --- | --- | --- |")
        for k in sets:
            v = g["slope_" + k].dropna()
            if len(v):
                lines.append(f"| {k} | {v.mean():+.4f} | {v.median():+.4f} |")
        gs = g[g.separable].dropna(subset=["slope_E0", "slope_E2", "slope_E0_matched"])
        if len(gs) >= 3:
            d_leak = (gs.slope_E0_matched - gs.slope_E2).values
            d_shift = (gs.slope_E0 - gs.slope_E0_matched).values
            p1 = wilcoxon(d_leak).pvalue if np.any(d_leak != 0) else np.nan
            p2 = wilcoxon(d_shift).pvalue if np.any(d_shift != 0) else np.nan
            lines += ["",
                      f"- 누수 제거 효과 (E0-matched − E2): 평균 {d_leak.mean():+.4f}, p = {p1:.3f}, n = {len(gs)}",
                      f"- 구간 이동 효과 (E0 − E0-matched): 평균 {d_shift.mean():+.4f}, p = {p2:.3f}, n = {len(gs)}"]
        else:
            lines += ["", f"- 분리 가능한 pair 가 {len(gs)}개라 검정하지 않음"]
        lines.append("")
    lines += ["## pair 별", "",
              sl.round(4).to_markdown(index=False) if hasattr(sl, "to_markdown") else sl.round(4).to_string(index=False),
              "",
              f"separable 판정: 중복 window 중 비중복과 같은 10분위에 있는 비율(mix_cov) >= {a.min_mix} "
              f"이고 매칭 표본의 중복 비율(leak_share) >= {a.min_leak}.",
              "separable 이 False 인 pair 는 중복이 특정 구간에 몰려 있어 누수와 구간 이동이 같은 위치에 겹친다. "
              "이 경우 E0-matched 가 E2 에 가깝게 나와도 구간 이동의 증거로 해석하면 안 된다."]
    nonsep = sl[(sl.overlap_frac > 0) & (~sl.separable)]
    if len(nonsep):
        lines += ["", "분리 불가 pair: " + ", ".join(sorted(nonsep.pair.unique()))]
    (out / "s2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[:40]))
    print(f"\n저장: {out.resolve()}")


# ------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "analyze"):
        q = sp.add_parser(name)
        q.add_argument("--runs", nargs="+", required=True, help="results.csv 와 scores_dump 가 있는 실행 폴더")
        q.add_argument("--stage", default="S2")
        q.add_argument("--methods", nargs="+", default=["full_ft", "adapter_latent"])
        q.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
        q.add_argument("--masks", default="leakfix_masks.npz")
        if name == "analyze":
            q.add_argument("--out", default="results_s2")
            q.add_argument("--n-boot", type=int, default=50)
            q.add_argument("--gap-mult", type=float, default=1.0)
            q.add_argument("--w-bench", type=int, default=100)
            q.add_argument("--w-field", type=int, default=20)
            q.add_argument("--min-leak", type=float, default=0.02)
            q.add_argument("--min-mix", type=float, default=0.5)
            q.add_argument("--seed", type=int, default=0)
    q = sp.add_parser("plan")
    q.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    q.add_argument("--prefix", default=r"jobs_out\final\S2_dump")
    a = ap.parse_args()
    {"check": cmd_check, "plan": cmd_plan, "analyze": cmd_analyze}[a.cmd](a)

"""
final_35_rescore.py
===================
**재학습 없이** 누수 통제를 강화해서 다시 채점한다.

왜 필요한가
-----------
지금까지의 leakage control 은 adaptation pool 의 window 와 evaluation window 를
**바이트 단위로 정확히 일치**하는 것만 제거했다. 그런데 파이프라인은
    pool  = _win(series, W, stride=5)
    eval  = _win(series, W, stride=1)
로 만들어진다. 따라서 pool window 하나당 정확히 일치하는 eval window 는 1개뿐이고,
그 좌우로 1~4 sample 씩 밀린 window 4개는 **내용의 95~99 % 를 공유하면서도 남는다.**

실측 결과 exact 제거율은 field 20 %, Boiler 16 %/4 % 였고, 그 결과
auroc_nopool 이 auroc 와 사실상 같았다. 이는 "누수가 없다"는 뜻이 아니라
**통제가 약하다**는 뜻이다.

올바른 통제는 시간 구간 기반이다.
    "adaptation pool 이 덮은 시간 구간과 겹치는 evaluation window 는 모두 제외"

구현
----
eval 이 stride 1 window 배열이므로, exact 일치 지점 i 를 중심으로
i-(W-1) ~ i+(W-1) 범위의 window 가 그 pool window 와 시간상 겹친다.
따라서 exact mask 를 W-1 만큼 팽창(dilate)시키면 시간 구간 기반 제외와
정확히 같아진다. 원 시계열이 없어도 계산할 수 있다는 것이 이 방법의 장점이다.

입력
----
- scores_dump/*.npz : leakfix 가 덤프한 per-window 점수 (y, s, pair)
- leakfix_masks.npz : pair 별 exact mask
- results.csv       : 러너가 만든 결과표

출력
----
- results_rescored.csv : auroc_nopool_exact / auroc_nopool_temporal 등 추가
- rescore_summary.csv  : pair 별 제거율과 평가 가능 여부

실행
    set PYTHONUTF8=1
    python final_35_rescore.py --dumps "jobs_out\\final\\*\\scores_dump" ^
                               --results "jobs_out\\final\\*\\results.csv"
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_PREFIX = ("Boiler", "SMD", "MSL", "SMAP")


# ----------------------------------------------------------------------
def window_len_for(pair: str, w_bench: int, w_field: int) -> int:
    return w_bench if str(pair).startswith(BENCH_PREFIX) else w_field


def labels_hash(y) -> str:
    y = np.ascontiguousarray(np.asarray(y).astype(np.int8))
    return f"{len(y)}_{hashlib.blake2b(y.tobytes(), digest_size=12).hexdigest()}"


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """True 위치를 좌우 radius 만큼 확장. 시간 구간 겹침과 동치."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    n = mask.size
    diff = np.zeros(n + 1, dtype=np.int32)
    idx = np.flatnonzero(mask)
    lo = np.maximum(idx - radius, 0)
    hi = np.minimum(idx + radius + 1, n)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi, -1)
    return np.cumsum(diff[:-1]) > 0


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    try:
        return float(roc_auc_score(y, np.asarray(s, dtype=float)))
    except Exception:
        return np.nan


def aupr(y, s):
    from sklearn.metrics import average_precision_score

    y = np.asarray(y).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    try:
        return float(average_precision_score(y, np.asarray(s, dtype=float)))
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
def load_masks(path: Path) -> tuple[dict, dict]:
    masks, meta = {}, {}
    if not path.exists():
        print(f"[경고] mask 파일 없음: {path}")
        return masks, meta
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    for k in z.files:
        if k == "__meta__":
            continue
        masks[k] = z[k].astype(bool)   # True = 유지(exact 기준)
        meta[k] = m.get(k, {})
    print(f"[ok] mask {len(masks)}개 로드")
    return masks, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=["jobs_out/final/*/scores_dump"])
    ap.add_argument("--results", nargs="*", default=["jobs_out/final/*/results.csv"])
    ap.add_argument("--masks", default="leakfix_masks.npz")
    ap.add_argument("--w-bench", type=int, default=100)
    ap.add_argument("--w-field", type=int, default=20)
    ap.add_argument("--gap-mult", type=float, default=1.0,
                    help="purge gap 배수. radius = gap_mult*W - 1. "
                         "1.0 = 시간 구간 겹침(기본), 2.0 = g=2L, 5.0 = g=5L")
    ap.add_argument("--tag", default="",
                    help="출력 파일명 접미어 (예: _g2L). 민감도 실행 시 사용")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[설정] gap_mult={args.gap_mult}  "
          f"(radius = {args.gap_mult}*W - 1;  1.0 이면 기존 시간구간 통제와 동일)")
    masks, meta = load_masks(Path(args.masks))

    # ---------- 점수 덤프 읽기 ----------
    files = []
    for pat in args.dumps:
        for d in glob.glob(pat):
            files += sorted(glob.glob(str(Path(d) / "*.npz")))
    if not files:
        print("[오류] score dump 를 찾지 못했습니다. leakfix 가 --no-dump-scores 로 "
              "돌았거나 경로가 다릅니다.")
        return 1
    print(f"[ok] score dump {len(files)}개")

    rows, per_pair = [], {}
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            y = z["y"].astype(int)
            s = z["s"].astype(float)
            pair = str(z["pair"])
        except Exception as e:
            print(f"[skip] {f}: {e!r}")
            continue

        W = window_len_for(pair, args.w_bench, args.w_field)
        yh = labels_hash(y)
        keep_exact = masks.get(yh)
        if keep_exact is None or keep_exact.size != y.size:
            keep_exact = np.ones(y.size, dtype=bool)
        inpool = ~keep_exact
        radius = max(int(round(args.gap_mult * W)) - 1, 0)
        keep_temporal = ~dilate(inpool, radius)

        rec = dict(
            file=Path(f).name, pair=pair, n_eval=int(y.size),
            auroc_all=auroc(y, s), aupr_all=aupr(y, s),
            n_keep_exact=int(keep_exact.sum()),
            frac_removed_exact=float(1 - keep_exact.mean()),
            auroc_nopool_exact=auroc(y[keep_exact], s[keep_exact]),
            aupr_nopool_exact=aupr(y[keep_exact], s[keep_exact]),
            gap_mult=float(args.gap_mult), purge_radius=int(radius),
            n_keep_temporal=int(keep_temporal.sum()),
            frac_removed_temporal=float(1 - keep_temporal.mean()),
            pos_temporal=int(y[keep_temporal].sum()),
            neg_temporal=int((1 - y[keep_temporal]).sum()),
            auroc_nopool_temporal=auroc(y[keep_temporal], s[keep_temporal]),
            aupr_nopool_temporal=aupr(y[keep_temporal], s[keep_temporal]),
        )
        rec["evaluable_temporal"] = bool(rec["pos_temporal"] > 0 and rec["neg_temporal"] > 0)
        rows.append(rec)
        per_pair.setdefault(pair, rec)

    df = pd.DataFrame(rows)
    TAG = args.tag
    df.to_csv(out / f"rescore_rows{TAG}.csv", index=False, encoding="utf-8-sig")

    # ---------- pair 별 요약 ----------
    summ = (df.groupby("pair")
              .agg(n_eval=("n_eval", "first"),
                   frac_removed_exact=("frac_removed_exact", "first"),
                   frac_removed_temporal=("frac_removed_temporal", "first"),
                   n_keep_temporal=("n_keep_temporal", "first"),
                   pos_temporal=("pos_temporal", "first"),
                   neg_temporal=("neg_temporal", "first"),
                   gap_mult=("gap_mult", "first"),
                   purge_radius=("purge_radius", "first"),
                   evaluable=("evaluable_temporal", "first"),
                   n_cells=("file", "count"))
              .reset_index())
    summ.to_csv(out / f"rescore_summary{TAG}.csv", index=False, encoding="utf-8-sig")
    print("\n=== pair 별 누수 제거 규모 ===")
    print(summ.to_string(index=False))

    bad = summ[~summ.evaluable]
    if len(bad):
        print("\n[중요] 시간 구간 기준으로 평가 집합이 남지 않는 pair:")
        print("       " + ", ".join(bad["pair"].tolist()))
        print("       -> 이 pair 들은 leakage-free 평가가 불가능합니다. 논문에서는")
        print("          transductive 설정임을 명시하거나 데이터 분할을 바꿔야 합니다.")

    # ---------- results.csv 와 병합 ----------
    rfiles = []
    for pat in args.results:
        rfiles += sorted(glob.glob(pat))
    if rfiles:
        res = pd.concat([pd.read_csv(f).assign(__src=f) for f in rfiles],
                        ignore_index=True, sort=False)
        key_cols = ["n_eval_windows", "auroc"]
        if all(c in res.columns for c in key_cols):
            res["_k"] = list(zip(res["n_eval_windows"].round(0), res["auroc"].round(12)))
            df["_k"] = list(zip(df["n_eval"].round(0), df["auroc_all"].round(12)))
            add = ["auroc_nopool_exact", "auroc_nopool_temporal",
                   "aupr_nopool_temporal", "frac_removed_exact",
                   "frac_removed_temporal", "n_keep_temporal",
                   "pos_temporal", "neg_temporal", "evaluable_temporal"]
            d2 = df.drop_duplicates("_k").set_index("_k")
            for c in add:
                res[c] = res["_k"].map(d2[c])
            res = res.drop(columns=["_k"])
            n_ok = int(res["auroc_nopool_temporal"].notna().sum())
            print(f"\n[ok] results {len(res)}행 중 {n_ok}행에 재채점 지표 부여")
            res["gap_mult"] = float(args.gap_mult)
            res.to_csv(out / f"results_rescored{TAG}.csv", index=False,
                       encoding="utf-8-sig")
            print(f"저장: {out / ('results_rescored' + TAG + '.csv')}")
        else:
            print("[경고] results.csv 에 n_eval_windows/auroc 가 없어 병합 생략")
    else:
        print("[경고] results.csv 를 찾지 못해 병합 생략")

    print("""
다음 단계
---------
python final_30_analysis.py --results results_rescored.csv
  -> auroc / auroc_nopool_exact / auroc_nopool_temporal 세 지표로 결론이
     어떻게 달라지는지 확인하세요. 논문에는 세 가지를 모두 보고하는 것이
     가장 방어적입니다.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

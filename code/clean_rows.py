"""clean_rows.py
results_rescored.csv 와 mechanism 결과에서 중복/시험 실행 행을 정리한다.

work/results_rescored.csv 는 jobs_out/final/*/results.csv 를 모두 합친 파일이라
  (1) smoke 실행(*_smoke, SWEEP_SMOKE, MSL_SMOKE) 행과
  (2) 같은 셀을 다시 돌린 *_leaked 실행 행이
본 실행 행과 함께 들어 있다. 같은 셀(stage, pair, seed, method, contam, alpha, n_heads, lb)이
여러 번 나타나면 평균이 섞인다. 정리 규칙:
  - smoke 행 제외
  - 셀마다 하나만 남김: 본 실행이면서 E2 가 있는 행 > leaked 이면서 E2 가 있는 행 > 본 실행 > leaked
원고 v9 의 Table 4, 5, 6, 7(기울기 행)과 Fig. 3, 4 는 이 정리본으로 계산한다.
S1 수치(Table 7–9, 12–14)는 단일 출처인 core_results_seed15/ 를 쓴다 (Table 12 의 5-seed 행은 그 중 seed 0–4).

사용:
    python code/clean_rows.py              # work/results_rescored_clean.csv 생성
"""
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["stage", "pair", "seed", "method", "contam", "alpha", "n_heads", "lb"]


STAGES = ("S2",)   # 원고 수치에 쓰이는 중복 단계(S2)만 정리한다. S1 은 core_results_seed15 를 쓴다.


def clean(d: pd.DataFrame) -> pd.DataFrame:
    if "__src" not in d:
        return d
    keep = d[~d["stage"].isin(STAGES)]
    d = d[d["stage"].isin(STAGES)]
    src = d["__src"].astype(str)
    d = d[~src.str.contains("smoke|SMOKE")].copy()
    src = d["__src"].astype(str)
    lk = src.str.contains("_leaked")
    e2 = d["auroc_nopool_temporal"].notna()
    d["_r"] = np.where(~lk & e2, 0, np.where(lk & e2, 1, np.where(~lk, 2, 3)))
    key = [c for c in KEY if c in d]
    d = d.sort_values("_r", kind="stable").drop_duplicates(key).drop(columns="_r")
    return pd.concat([d, keep]).sort_index()


def attach_inout(res: pd.DataFrame, mech_rows: pd.DataFrame) -> pd.DataFrame:
    """정리된 결과 행에 dump 단위 IN/OUT AUROC(mechanism_rows.csv)를 붙인다.
    결합 키는 final_35 와 같은 (평가 창 수, AUROC 소수 12자리)."""
    res = res.copy()
    res["_k"] = list(zip(res["n_eval_windows"].round(0), res["auroc"].round(12)))
    m = mech_rows.copy()
    m["_k"] = list(zip(m["n_eval"].round(0), m["auroc_all"].round(12)))
    m = m.drop_duplicates("_k").set_index("_k")
    for c in ["auroc_in", "auroc_out"]:
        res[c] = res["_k"].map(m[c])
    return res.drop(columns="_k")


if __name__ == "__main__":
    W = Path(__file__).resolve().parents[1] / "work"
    for tag in ["", "_g2L", "_g5L"]:
        src = W / f"results_rescored{tag}.csv"
        d = pd.read_csv(src, low_memory=False)
        c = clean(d)
        c.to_csv(W / f"results_rescored{tag}_clean.csv", index=False)
        print(f"{src.name}: {len(d)} -> {len(c)} rows")

"""
final_45_leakage_mechanism.py
=============================
"오염에 의한 성능 붕괴가 leakage artifact 였다"는 것을 **직접 증명**한다.
재학습 없이 score dump 만으로 계산한다.

논리
----
adaptation pool 이 evaluation set 안에 들어 있으면, full fine-tuning 은 pool 에
섞인 anomalous window 를 잘 재구성하도록 학습되고 그 window 들이 그대로 채점된다.
따라서 AUROC 가 무너진다. 그런데 pool 과 시간적으로 겹치지 않는 구간에서는
그럴 이유가 없다.

이 스크립트는 evaluation set 을 두 구간으로 나눠 각각 AUROC 를 계산한다.
    IN  : pool 이 덮은 시간 구간과 겹치는 window
    OUT : 겹치지 않는 window (leakage-free)
예측이 맞다면
    IN  구간에서는 오염이 커질수록 AUROC 가 급락하고
    OUT 구간에서는 거의 변하지 않는다.

출력
----
- mechanism_rows.csv   : cell 단위 IN/OUT AUROC
- mechanism_summary.csv: method x contamination 요약
- fig_leakage_mechanism.png : 논문 Figure 용

실행
    set PYTHONUTF8=1
    python final_45_leakage_mechanism.py --dumps "jobs_out\\final\\*\\scores_dump" ^
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


def labels_hash(y) -> str:
    y = np.ascontiguousarray(np.asarray(y).astype(np.int8))
    return f"{len(y)}_{hashlib.blake2b(y.tobytes(), digest_size=12).hexdigest()}"


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    n = mask.size
    diff = np.zeros(n + 1, dtype=np.int32)
    idx = np.flatnonzero(mask)
    np.add.at(diff, np.maximum(idx - radius, 0), 1)
    np.add.at(diff, np.minimum(idx + radius + 1, n), -1)
    return np.cumsum(diff[:-1]) > 0


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    try:
        return float(roc_auc_score(y, np.asarray(s, float)))
    except Exception:
        return np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=["jobs_out/final/*/scores_dump"])
    ap.add_argument("--results", nargs="*", default=["jobs_out/final/*/results.csv"])
    ap.add_argument("--masks", default="leakfix_masks.npz")
    ap.add_argument("--w-bench", type=int, default=100)
    ap.add_argument("--w-field", type=int, default=20)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- masks ----
    masks = {}
    mp = Path(args.masks)
    if mp.exists():
        z = np.load(mp, allow_pickle=True)
        for k in z.files:
            if k != "__meta__":
                masks[k] = z[k].astype(bool)
    print(f"[ok] mask {len(masks)}개")

    # ---- dumps ----
    files = []
    for pat in args.dumps:
        for d in glob.glob(pat):
            files += sorted(glob.glob(str(Path(d) / "*.npz")))
    if not files:
        print("[오류] score dump 없음")
        return 1
    print(f"[ok] dump {len(files)}개")

    rows = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            y, s, pair = z["y"].astype(int), z["s"].astype(float), str(z["pair"])
        except Exception:
            continue
        W = args.w_bench if pair.startswith(BENCH_PREFIX) else args.w_field
        keep_exact = masks.get(labels_hash(y))
        if keep_exact is None or keep_exact.size != y.size:
            continue
        inpool = dilate(~keep_exact, W - 1)      # pool 구간과 겹치는 window
        outpool = ~inpool
        rows.append(dict(
            file=Path(f).name, pair=pair, n_eval=int(y.size),
            auroc_all=auroc(y, s),
            n_in=int(inpool.sum()), pos_in=int(y[inpool].sum()),
            auroc_in=auroc(y[inpool], s[inpool]),
            n_out=int(outpool.sum()), pos_out=int(y[outpool].sum()),
            auroc_out=auroc(y[outpool], s[outpool]),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[오류] 매칭된 dump 가 없습니다.")
        return 1
    df.to_csv(out / "mechanism_rows.csv", index=False, encoding="utf-8-sig")

    # ---- results 와 병합해서 method/contam 을 붙인다 ----
    rfiles = []
    for pat in args.results:
        rfiles += sorted(glob.glob(pat))
    if not rfiles:
        print("[경고] results.csv 없음 — cell 단위 요약만 저장")
        return 0
    res = pd.concat([pd.read_csv(f) for f in rfiles], ignore_index=True, sort=False)
    res["_k"] = list(zip(res["n_eval_windows"].round(0), res["auroc"].round(12)))
    df["_k"] = list(zip(df["n_eval"].round(0), df["auroc_all"].round(12)))
    d2 = df.drop_duplicates("_k").set_index("_k")
    for c in ["auroc_in", "auroc_out", "n_in", "n_out", "pos_in", "pos_out"]:
        res[c] = res["_k"].map(d2[c])
    res = res.drop(columns=["_k"])
    res.to_csv(out / "mechanism_merged.csv", index=False, encoding="utf-8-sig")

    s2 = res[(res["stage"] == "S2") & res["auroc_out"].notna()]
    if s2.empty:
        print("[경고] S2 매칭 행이 없습니다.")
        return 0
    summ = (s2.groupby(["method", "contam"])[["auroc", "auroc_in", "auroc_out"]]
              .agg(["mean", "count"]).round(4))
    summ.to_csv(out / "mechanism_summary.csv", encoding="utf-8-sig")
    print("\n=== S2: 평가 구간별 AUROC (pool 과 겹치는 IN vs 겹치지 않는 OUT) ===")
    print(s2.groupby(["method", "contam"])[["auroc", "auroc_in", "auroc_out"]]
            .mean().round(4).to_string())

    # ---- 하락폭 비교 ----
    g = s2.groupby(["pair", "method", "contam"])[["auroc_in", "auroc_out"]].mean()
    print("\n=== 0 -> 20 % 하락폭 ===")
    for meth in sorted(s2["method"].unique()):
        try:
            sub = g.xs(meth, level="method").unstack()
        except KeyError:
            continue
        for col in ["auroc_in", "auroc_out"]:
            try:
                dr = (sub[(col, 0.20)] - sub[(col, 0.0)]).dropna()
            except KeyError:
                continue
            if len(dr):
                print(f"  {meth:<15s} {col:<10s} mean {dr.mean():+.4f}  n={len(dr)}")

    # ---- Figure ----
    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            piv = s2.groupby(["method", "contam"])[["auroc_in", "auroc_out"]].mean()
            methods = [m for m in ["source_only", "adapter_latent", "adapter_isca", "full_ft"]
                       if m in piv.index.get_level_values(0)]
            label = {"source_only": "No adaptation", "adapter_latent": "Adapter (1%)",
                     "adapter_isca": "Adapter + self-refine", "full_ft": "Full fine-tuning"}
            colour = {"source_only": "#808080", "adapter_latent": "#1f77b4",
                      "adapter_isca": "#2ca02c", "full_ft": "#b03a2e"}
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
            for ax, col, ttl in zip(
                    axes, ["auroc_in", "auroc_out"],
                    ["(a) windows overlapping the adaptation pool",
                     "(b) windows disjoint from the adaptation pool"]):
                for m in methods:
                    sub = piv.xs(m, level="method")[col].sort_index()
                    ax.plot(sub.index * 100, sub.values, marker="o", ms=5,
                            color=colour.get(m, None), label=label.get(m, m))
                ax.set_title(ttl, fontsize=9)
                ax.set_xlabel("Contamination ratio of the reference pool (%)", fontsize=8)
                ax.grid(alpha=.3)
                ax.tick_params(labelsize=8)
            axes[0].set_ylabel("AUROC", fontsize=8)
            axes[0].legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out / "fig_leakage_mechanism.png", dpi=400)
            print(f"\n저장: {out/'fig_leakage_mechanism.png'}")
        except Exception as e:
            print(f"[경고] figure 생성 실패: {e!r}")

    print("""
해석
----
(a) 와 (b) 의 차이가 크면, 보고되던 '오염에 의한 붕괴'는 adaptation 에 쓴 window 를
    그대로 채점한 데서 생긴 artifact 다.
(b) 에서도 하락이 남으면 그 부분이 진짜 오염 효과다.
두 그림을 나란히 싣는 것이 이 논문에서 가장 강한 근거가 된다.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

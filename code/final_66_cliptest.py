# -*- coding: utf-8 -*-
"""
final_66_cliptest.py — 정규화 결함의 비용을 재학습 없이 측정
=============================================================
진단 결과: exp9_unified.prepare() 의 `sc = B.MinMax().fit(S["train"])` 가
source train 에서 거의 상수인 feature 를 만나면 분모가 0 에 가까워져,
target 값이 최대 ±8.4e6 까지 폭발한다 (feature 16, 18, 14, 17, 3, 5).

이 스크립트는 네 가지 처리를 같은 ckpt 로 비교한다. 학습은 하지 않는다.

  raw        현행 (아무 처리 없음)
  clip       target 을 [0,1] 로 clip.  source 통계만 사용 -> target 누수 없음
  drop       source train range 가 tol 이하인 feature 를 양쪽에서 제거
  winsor     source train 의 [q, 1-q] 분위수로 clip (기본 q=0.001)

동시에 adaptation pool(clean/anom)에도 극단값이 있는지 확인한다.
Boiler_3to2 full_ft 가 contam>0 에서만 NaN 으로 발산한 원인 가설
("오염 주입으로 극단값 window 가 pool 에 들어가 gradient 가 넘친다")을
직접 검증하기 위한 것이다.

실행:
  set PYTHONUTF8=1
  python final_66_cliptest.py --ckpt-dir jobs_out\\final\\S2_main --register-boiler
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARGS = SimpleNamespace(indicator="rms", k=3.0, persistence=5,
                       epochs_pretrain=120, patience=15,
                       force_pretrain=False, msl_max=4, smoke=False)


def auroc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def find_ckpt(pair, seed, W, n_heads, lb, dirs):
    name = f"{pair}__s{seed}__h{n_heads}__lb{lb:g}__w{W}.pt"
    for g in dirs:
        hits = glob.glob(str(Path(g) / "ckpt" / name))
        if hits:
            return Path(hits[0])
    return None


def extreme_stats(arr, name):
    if arr is None or len(arr) == 0:
        return dict(**{f"{name}_n": 0, f"{name}_max": np.nan,
                       f"{name}_frac_out": np.nan})
    a = np.asarray(arr, dtype=np.float32)
    return {f"{name}_n": int(len(a)),
            f"{name}_max": float(np.abs(a).max()),
            f"{name}_frac_out": float(((a > 1.0) | (a < 0.0)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[
        "Boiler_1to2", "Boiler_1to3", "Boiler_2to1",
        "Boiler_2to3", "Boiler_3to1", "Boiler_3to2"])
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-search", nargs="*", default=["jobs_out/final/*"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cliptest_out")
    ap.add_argument("--winsor-q", type=float, default=0.001)
    ap.add_argument("--drop-tol", type=float, default=1e-8,
                    help="source train range 가 이 값 이하인 feature 를 제거")
    ap.add_argument("--rescore-summary", default="rescore_summary.csv",
                    help="n_keep_temporal 을 읽어 누수 통제 구간 AUROC 도 계산")
    ap.add_argument("--register-boiler", action="store_true")
    ap.add_argument("--boiler-spec", default=None)
    cli = ap.parse_args()

    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = ([cli.ckpt_dir] if cli.ckpt_dir else [])
    for g in cli.ckpt_search:
        dirs.extend(sorted(glob.glob(g)) or [g])

    keep_temporal = {}
    if Path(cli.rescore_summary).exists():
        try:
            rs = pd.read_csv(cli.rescore_summary)
            keep_temporal = dict(zip(rs["pair"], rs["n_keep_temporal"]))
            print(f"[info] {cli.rescore_summary} 에서 누수 통제 구간 크기를 읽었습니다")
        except Exception as e:
            print(f"[warn] rescore_summary 읽기 실패: {e!r}")

    try:
        import final_20_run as F20
        if cli.register_boiler:
            F20.register_boiler(cli.boiler_spec)
    except Exception as e:
        print(f"[warn] boiler 등록 실패: {e!r}")

    import torch
    from config import train_cfg, model_cfg
    from models import create_model
    from exp9_unified import prepare, MODEL_NAME
    from exp2_robust import scores as tf_scores

    N_HEADS, LB = 5, 0.1
    device, bs = train_cfg.device, train_cfg.batch_size
    rows, pool_rows = [], []

    for pair in cli.pairs:
        print(f"\n{'='*62}\n{pair}\n{'='*62}")
        try:
            D = prepare(pair, ARGS)
        except Exception as e:
            print(f"  [skip] prepare 실패: {type(e).__name__}: {e}")
            continue

        ev = np.asarray(D["eval_x"], dtype=np.float32)
        y = np.asarray(D["eval_y"]).astype(int)
        src = np.asarray(D["src"], dtype=np.float32)
        W, C = ev.shape[1], ev.shape[2]

        ck = find_ckpt(pair, cli.seed, W, N_HEADS, LB, dirs)
        if ck is None:
            print("  [skip] ckpt 없음")
            continue

        # ---------- pool 극단값 점검 (발산 원인 가설) ----------
        pr = dict(pair=pair)
        pr.update(extreme_stats(D.get("clean"), "clean"))
        pr.update(extreme_stats(D.get("anom"), "anom"))
        pr.update(extreme_stats(ev, "eval"))
        pool_rows.append(pr)
        print(f"  pool clean: n={pr['clean_n']} max|x|={pr['clean_max']:.3g}")
        print(f"  pool anom : n={pr['anom_n']} max|x|={pr['anom_max']:.3g}"
              f"   <- contam>0 일 때 여기서 뽑아 pool 에 넣습니다")

        # ---------- 처리 4종 ----------
        lo = src.reshape(-1, C).min(0)
        hi = src.reshape(-1, C).max(0)
        rngw = hi - lo
        degenerate = np.where(rngw <= cli.drop_tol)[0]
        qlo = np.quantile(src.reshape(-1, C), cli.winsor_q, axis=0)
        qhi = np.quantile(src.reshape(-1, C), 1 - cli.winsor_q, axis=0)

        variants = {
            "raw":    ev,
            "clip":   np.clip(ev, 0.0, 1.0),
            "winsor": np.clip(ev, qlo, qhi),
        }
        if len(degenerate):
            kp = np.setdiff1d(np.arange(C), degenerate)
            variants["drop"] = ev[:, :, kp]
        print(f"  source range 퇴화 feature ({cli.drop_tol:g} 이하): "
              f"{degenerate.tolist() if len(degenerate) else '없음'}")

        n_tail = int(keep_temporal.get(pair, max(1, int(len(ev) * 0.2))))
        n_tail = min(n_tail, len(ev))

        for vname, X in variants.items():
            if vname == "drop":
                # feature 수가 달라지므로 이 ckpt 로는 채점 불가
                print("  [drop] feature 수가 바뀌어 기존 ckpt 로 채점 불가 -> 건너뜀")
                continue
            model_cfg.n_heads = N_HEADS
            model_cfg.input_size = X.shape[2]
            model_cfg.window_size = W
            model_cfg.d_model, model_cfg.dim_ff = 128, 512
            m = create_model(MODEL_NAME, model_cfg)
            m.load_state_dict(torch.load(ck, map_location=device))
            m = m.to(device).eval()
            s = np.asarray(tf_scores(m, X, bs, device), dtype=float)
            a_all = auroc(y, s)
            a_tail = auroc(y[-n_tail:], s[-n_tail:])
            rows.append(dict(pair=pair, variant=vname,
                             auroc_full=a_all, auroc_tail=a_tail,
                             n_tail=n_tail,
                             tail_anom_ratio=float(y[-n_tail:].mean()),
                             max_abs=float(np.abs(X).max()),
                             mean_score=float(np.nanmean(s)),
                             max_score=float(np.nanmax(s))))
            print(f"  {vname:7s} AUROC(전체)={a_all:.4f}  "
                  f"AUROC(누수통제 구간)={a_tail:.4f}  "
                  f"max|x|={np.abs(X).max():.3g}  max점수={np.nanmax(s):.3g}")

    if not rows:
        print("\n[오류] 결과가 없습니다. ckpt 경로를 확인하세요.")
        return 1

    t = pd.DataFrame(rows)
    t.to_csv(out / "C1_normalization_variants.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pool_rows).to_csv(out / "C2_pool_extremes.csv",
                                   index=False, encoding="utf-8-sig")

    print(f"\n{'='*62}\n전체 AUROC\n{'='*62}")
    print(t.pivot_table(index="pair", columns="variant",
                        values="auroc_full").round(4).to_string())
    print(f"\n누수 통제 구간 AUROC (주 지표에 해당)")
    print(t.pivot_table(index="pair", columns="variant",
                        values="auroc_tail").round(4).to_string())

    piv = t.pivot_table(index="pair", columns="variant", values="auroc_tail")
    if "raw" in piv and "clip" in piv:
        d = (piv["clip"] - piv["raw"]).dropna()
        print(f"\nclip - raw (누수 통제 구간): 평균 {d.mean():+.4f}, "
              f"최대 {d.abs().max():.4f}")
        print("  |변화| 가 어느 pair 에서도 0.02 미만이면 -> 각주로만 보고하고 재실행 불필요")
        print("  0.02 이상이면 -> prepare() 를 고치고 target 쪽만 재실행")
    print(f"\n[done] {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

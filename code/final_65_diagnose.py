# -*- coding: utf-8 -*-
"""
final_65_diagnose.py — AUROC < 0.5 pair 원인 진단 (재학습 없음)
================================================================
문제: source 가 무엇이든 target 이 boiler 1 이면 source_only AUROC 가
      0.28~0.37 로 뒤집힌다. transfer 문제가 아니라 target 쪽 문제다.

원인 후보 4개를 구분한다.

  (A) 정규화 clipping
      exp9_unified.prepare() 는 `sc = B.MinMax().fit(S["train"])` 로
      source train 통계만으로 스케일러를 맞춘 뒤 target test 에 그대로
      적용한다. target 값이 [0,1] 을 크게 벗어나면 이상 구간이 포화되어
      오히려 재구성이 쉬워질 수 있다.
  (B) 라벨 정렬 어긋남
      window label 이 시간축으로 밀려 있으면 부호가 뒤집힌다.
  (C) 진폭 감소형 이상
      이상 구간이 정상보다 평탄하면 재구성이 더 쉬워 AUROC < 0.5 가
      정상적으로 나온다. 버그가 아니라 데이터 특성이다.
  (D) 평가 구간 이질성
      앞/뒤 구간의 AUROC 가 크게 다르면 test 파일 구조 문제다.
      (Boiler 는 test.csv 가 train.csv 를 prefix 로 포함한다)

실행:
  set PYTHONUTF8=1
  python final_65_diagnose.py --ckpt-dir jobs_out\\final\\S2_main --register-boiler

ckpt 를 못 찾으면 --ckpt-search 로 여러 폴더를 뒤진다(기본: jobs_out/final/*).
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

# run_all_xjtu_fixed.main() 의 argparse 기본값과 동일
ARGS = SimpleNamespace(
    indicator="rms", k=3.0, persistence=5,
    epochs_pretrain=120, patience=15, force_pretrain=False,
    msl_max=4, smoke=False,
)


def auroc(y, s):
    """rank 기반 AUROC (동점 처리 포함). sklearn 불필요."""
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


def find_ckpt(pair, seed, W, n_heads, lb, search_dirs):
    name = f"{pair}__s{seed}__h{n_heads}__lb{lb:g}__w{W}.pt"
    for g in search_dirs:
        hits = glob.glob(str(Path(g) / "ckpt" / name))
        if hits:
            return Path(hits[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[
        "Boiler_1to2", "Boiler_1to3", "Boiler_2to1",
        "Boiler_2to3", "Boiler_3to1", "Boiler_3to2"])
    ap.add_argument("--ckpt-dir", default=None,
                    help="우선 탐색할 outdir (예: jobs_out\\final\\S2_main)")
    ap.add_argument("--ckpt-search", nargs="*", default=["jobs_out/final/*"],
                    help="ckpt 를 찾을 glob 목록")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="diagnose_out")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--shift-max", type=int, default=300)
    ap.add_argument("--shift-step", type=int, default=5)
    ap.add_argument("--register-boiler", action="store_true")
    ap.add_argument("--boiler-spec", default=None)
    ap.add_argument("--register-msl", action="store_true")
    cli = ap.parse_args()

    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)

    search_dirs = []
    if cli.ckpt_dir:
        search_dirs.append(cli.ckpt_dir)
    for g in cli.ckpt_search:
        search_dirs.extend(sorted(glob.glob(g)) or [g])
    print(f"[ckpt] 탐색 대상 {len(search_dirs)}곳: {search_dirs[:5]}")

    # ---- pair 등록 (Boiler 확장 / MSL) ----
    try:
        import final_20_run as F20
        if cli.register_boiler:
            F20.register_boiler(cli.boiler_spec)
        if cli.register_msl:
            F20.register_msl(cli.ckpt_dir or "jobs_out/final", msl_max=4)
    except Exception as e:
        print(f"[warn] pair 등록 실패(계속 진행): {type(e).__name__}: {e}")

    import torch
    from config import train_cfg, model_cfg
    from models import create_model
    from exp9_unified import prepare, MODEL_NAME
    from exp2_robust import scores as tf_scores

    N_HEADS, LB = 5, 0.1
    device = train_cfg.device
    bs = train_cfg.batch_size

    rows, dec_rows, shift_rows, feat_rows = [], [], [], []

    for pair in cli.pairs:
        print(f"\n{'='*60}\n{pair}\n{'='*60}")
        try:
            D = prepare(pair, ARGS)
        except Exception as e:
            print(f"  [skip] prepare 실패: {type(e).__name__}: {e}")
            continue

        ev = np.asarray(D["eval_x"], dtype=np.float32)
        y = np.asarray(D["eval_y"]).astype(int)
        src = np.asarray(D["src"], dtype=np.float32)
        W, C = ev.shape[1], ev.shape[2]

        ck = find_ckpt(pair, cli.seed, W, N_HEADS, LB, search_dirs)
        if ck is None:
            print(f"  [skip] ckpt 없음: "
                  f"{pair}__s{cli.seed}__h{N_HEADS}__lb{LB:g}__w{W}.pt")
            continue
        print(f"  ckpt: {ck}")

        # ---------- (A) 정규화 clipping ----------
        gt1 = float((ev > 1.0).mean())
        lt0 = float((ev < 0.0).mean())
        for c in range(C):
            v = ev[:, :, c]
            feat_rows.append(dict(pair=pair, feature=c,
                                  frac_gt1=float((v > 1.0).mean()),
                                  frac_lt0=float((v < 0.0).mean()),
                                  vmin=float(v.min()), vmax=float(v.max())))

        # ---------- source_only 점수 ----------
        model_cfg.n_heads = N_HEADS
        model_cfg.input_size = D["n_feat"]
        model_cfg.window_size = W
        model_cfg.d_model = 128
        model_cfg.dim_ff = 512
        m = create_model(MODEL_NAME, model_cfg)
        m.load_state_dict(torch.load(ck, map_location=device))
        m = m.to(device).eval()

        s = np.asarray(tf_scores(m, ev, bs, device), dtype=float)
        a_all = auroc(y, s)

        mu1 = float(np.nanmean(s[y == 1])) if y.sum() else float("nan")
        mu0 = float(np.nanmean(s[y == 0])) if (len(y) - y.sum()) else float("nan")

        # ---------- (C) 진폭 방향 ----------
        amp = ev.reshape(len(ev), -1).std(axis=1)
        amp1 = float(np.nanmean(amp[y == 1])) if y.sum() else float("nan")
        amp0 = float(np.nanmean(amp[y == 0])) if (len(y) - y.sum()) else float("nan")

        # ---------- (D) 구간별 AUROC ----------
        edges = np.linspace(0, len(s), cli.deciles + 1).astype(int)
        for i in range(cli.deciles):
            sl = slice(edges[i], edges[i + 1])
            dec_rows.append(dict(
                pair=pair, decile=i + 1, n=int(edges[i + 1] - edges[i]),
                anom_ratio=float(y[sl].mean()) if edges[i + 1] > edges[i] else np.nan,
                auroc=auroc(y[sl], s[sl]),
                mean_score=float(np.nanmean(s[sl])),
                mean_input_std=float(np.nanmean(amp[sl]))))

        # ---------- (B) 라벨 시프트 ----------
        best_sh, best_a = 0, a_all
        for sh in range(-cli.shift_max, cli.shift_max + 1, cli.shift_step):
            if sh == 0:
                continue
            yy, ss = (y[sh:], s[:-sh]) if sh > 0 else (y[:sh], s[-sh:])
            if len(yy) < 200 or yy.sum() == 0:
                continue
            a = auroc(yy, ss)
            if np.isfinite(a) and np.isfinite(best_a) and \
                    abs(a - 0.5) > abs(best_a - 0.5):
                best_sh, best_a = sh, a
        shift_rows.append(dict(pair=pair, auroc_shift0=a_all,
                               best_shift=best_sh, auroc_best_shift=best_a,
                               gain=abs(best_a - 0.5) - abs(a_all - 0.5)))

        # ---------- 판정 ----------
        verdicts = []
        if gt1 > 0.05 or lt0 > 0.05:
            verdicts.append("A:정규화 clipping 의심")
        if abs(best_a - 0.5) - abs(a_all - 0.5) > 0.10:
            verdicts.append(f"B:라벨 시프트 의심(shift={best_sh})")
        if np.isfinite(amp1) and np.isfinite(amp0) and amp1 < amp0 * 0.9:
            verdicts.append("C:진폭 감소형 이상(버그 아님)")
        dsub = [r for r in dec_rows if r["pair"] == pair and np.isfinite(r["auroc"])]
        if dsub:
            spread = max(r["auroc"] for r in dsub) - min(r["auroc"] for r in dsub)
            if spread > 0.35:
                verdicts.append(f"D:구간 이질성(AUROC 폭 {spread:.2f})")
        if not verdicts:
            verdicts.append("특이 소견 없음")

        rows.append(dict(
            pair=pair, n_eval=len(s), n_feat=C, window=W,
            anom_ratio=float(y.mean()), auroc=a_all,
            auroc_if_inverted=1 - a_all,
            mean_score_anom=mu1, mean_score_norm=mu0,
            score_dir="정상>이상(뒤집힘)" if mu1 < mu0 else "이상>정상(정상)",
            input_std_anom=amp1, input_std_norm=amp0,
            amp_dir="이상이 더 평탄" if amp1 < amp0 else "이상이 더 큼",
            frac_gt1=gt1, frac_lt0=lt0,
            eval_min=float(ev.min()), eval_max=float(ev.max()),
            src_min=float(src.min()), src_max=float(src.max()),
            best_shift=best_sh, auroc_best_shift=best_a,
            verdict=" | ".join(verdicts)))

        print(f"  AUROC          = {a_all:.4f}   (뒤집으면 {1-a_all:.4f})")
        print(f"  평균 점수       이상={mu1:.6f}  정상={mu0:.6f}")
        print(f"  입력 표준편차   이상={amp1:.4f}  정상={amp0:.4f}")
        print(f"  eval 범위      [{ev.min():.3f}, {ev.max():.3f}]"
              f"   >1: {gt1:.2%}   <0: {lt0:.2%}")
        print(f"  라벨 시프트     best={best_sh}  AUROC={best_a:.4f}")
        print(f"  >>> 판정: {' | '.join(verdicts)}")

    if not rows:
        print("\n[오류] 진단된 pair 가 없습니다.")
        print("  ckpt 폴더를 확인하세요:  dir jobs_out\\final\\S2_main\\ckpt")
        return 1

    pd.DataFrame(rows).to_csv(out / "D1_pair_summary.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(dec_rows).to_csv(out / "D2_auroc_by_time_decile.csv",
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(shift_rows).to_csv(out / "D3_label_shift.csv",
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(feat_rows).to_csv(out / "D4_feature_range.csv",
                                   index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}\n요약\n{'='*60}")
    t = pd.DataFrame(rows)[["pair", "auroc", "score_dir", "amp_dir",
                            "frac_gt1", "best_shift", "verdict"]]
    print(t.round(4).to_string(index=False))
    print(f"\n[done] {out.resolve()}")
    print("""
해석:
  A 가 뜨면  -> 정규화가 원인. target 통계로도 스케일하는 변형을 시험할 근거가 생깁니다.
               (단, target 정보를 쓰는 것이므로 논문에 반드시 명시해야 합니다)
  B 가 뜨면  -> 라벨 정렬 버그. exp1_benchmark.win_labels 를 점검하세요.
  C 만 뜨면  -> 버그가 아닙니다. reconstruction 기반 탐지의 구조적 한계로 서술하세요.
  D 가 뜨면  -> 평가 구간이 이질적. 시간 구간 통제 결과와 함께 해석해야 합니다.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

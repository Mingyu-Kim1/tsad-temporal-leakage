# -*- coding: utf-8 -*-
"""
final_70_sweep.py — capacity 사다리 + adaptation hyperparameter sweep
=====================================================================
PRESPEC_v1.md 의 규약을 코드로 강제한다.
  * 튜닝 pair 와 보고 pair 를 분리 (--mode tune / report)
  * adapter 와 full_ft 에 같은 개수의 후보
  * 돌린 grid 를 전부 CSV 로 남김 (diverged 셀 포함)

capacity 는 exp3_final.CAPACITY 에 이미 있는 shift/diag/linear 를 그대로 쓰고,
bottleneck 폭만 LatentAdapter(latent_dim, bottleneck=r) 로 등록한다.

  shift      128 params (0.015%)   diag      256 (0.031%)
  bneck8   2,208 (0.27%)           bneck16 4,240 (0.51%)
  bneck32  8,352 (1.01%)  <-현행    bneck64 16,576 (2.01%)
  bneck128 33,024 (3.99%)          linear  16,512 (2.00%)
  full_ft 826,773 (100%)

실행:
  python final_70_sweep.py --mode tune   --outdir jobs_out\\final\\SWEEP_TUNE ^
      --ckpt-from jobs_out\\final\\S2_main --seeds 0 1 2
  python final_70_sweep.py --mode report --outdir jobs_out\\final\\SWEEP_REPORT ^
      --ckpt-from jobs_out\\final\\S1_main --register-boiler --seeds 0 1 2 ^
      --config bneck32_lr1e-3_e40 shift_lr1e-3_e40 ...
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, copy, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TUNE_PAIRS = ["Boiler_1to2", "SMD_1-1to1-2"]
REPORT_PAIRS = ["Boiler_1to3", "Boiler_2to1", "Boiler_2to3", "Boiler_3to1",
                "Boiler_3to2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2",
                "MSL_P-11toP-10", "MSL_P-10toP-11", "MSL_G-6toE-13", "MSL_G-6toE-1"]

ARGS = SimpleNamespace(indicator="rms", k=3.0, persistence=5,
                       epochs_pretrain=120, patience=15,
                       force_pretrain=False, msl_max=4, smoke=False)


def register_bottlenecks(widths, log=print):
    import exp3_final as E3
    from run_cross_domain_adapter_v2 import LatentAdapter
    for r in widths:
        E3.CAPACITY[f"bneck{r}"] = (lambda rr: (lambda d: LatentAdapter(d, rr)))(r)
    log(f"[capacity] 등록: {['bneck%d' % r for r in widths]}")


def build_grid(args):
    g = []
    # full_finetune 은 patience=20, ReduceLROnPlateau(10) 이 하드코딩돼 있다.
    # adapter 의 train_adapter 는 patience=8, 스케줄러 없음.
    # 예산을 맞추기 위해 adapter 쪽에 patience 축을 넣는다.
    for lr in [3e-4, 1e-3, 3e-3]:
        for ep, pa in [(40, 8), (120, 20)]:
            g.append((f"bneck32_lr{lr:g}_e{ep}_p{pa}",
                      dict(kind="adapter", cap="bneck32", lr=lr,
                           epochs=ep, patience=pa)))
    for cap in ["shift", "diag", "bneck8", "bneck16", "bneck64", "bneck128", "linear"]:
        g.append((f"{cap}_lr1e-3_e40_p8",
                  dict(kind="adapter", cap=cap, lr=1e-3, epochs=40, patience=8)))
    for lr in [3e-5, 1e-4, 3e-4]:
        for ep in [40, 120]:
            g.append((f"fullft_lr{lr:g}_e{ep}",
                      dict(kind="full_ft", cap=None, lr=lr, epochs=ep, patience=20)))
    g.append(("source_only",
              dict(kind="source_only", cap=None, lr=0.0, epochs=0, patience=0)))
    if args.config:
        g = [(n, c) for n, c in g if n in args.config]
        if not g:
            raise SystemExit(f"--config 에 맞는 설정 없음: {args.config}")
    if args.smoke:
        g = [g[0], ("source_only",
                    dict(kind="source_only", cap=None, lr=0.0, epochs=0, patience=0))]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "report"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt-from", default=None)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--contam", type=float, default=0.0)
    ap.add_argument("--clip-eval", action="store_true",
                    help="평가 입력도 [0,1] 로 clip. --clip-pool 과 함께 쓰세요")
    ap.add_argument("--clip-pool", action="store_true",
                    help="adaptation pool 을 [0,1] 로 clip (full_ft 발산 방지 부록용). "
                         "주 결과에는 쓰지 말 것 — 반드시 별도 outdir 로 분리")
    ap.add_argument("--no-leakfix", action="store_true")
    ap.add_argument("--register-boiler", action="store_true")
    ap.add_argument("--boiler-spec", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["ckpt", "scores", "figures"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    LF = None
    if not args.no_leakfix:
        try:
            import final_10_leakfix as LF_mod
            LF_mod.install(outdir=str(outdir), hash_round=None,
                           abort_after_masks=False, dump_scores=True)
            LF = LF_mod; print("[sweep] leakfix 설치 완료")
        except Exception as e:
            print(f"[sweep] leakfix 설치 실패(계속): {e!r}")

    import final_20_run as F20
    F20.register_msl(outdir, msl_max=4)
    if args.register_boiler:
        F20.register_boiler(args.boiler_spec)
    if args.ckpt_from:
        F20.reuse_ckpt(Path(args.ckpt_from), outdir)

    import torch
    import exp3_final as E3
    from config import train_cfg, model_cfg
    from models import create_model
    from exp9_unified import prepare, pretrain, MODEL_NAME
    from exp2_robust import scores as tf_scores, evaluate_all
    from exp1_efficiency import AdaptTimer
    from run_cross_domain_adapter_v2 import get_latent_dim, full_finetune

    register_bottlenecks([8, 16, 32, 64, 128])
    N_HEADS, LB = 5, 0.1
    device, bs = train_cfg.device, train_cfg.batch_size
    pairs = args.pairs or (TUNE_PAIRS if args.mode == "tune" else REPORT_PAIRS)
    grid = build_grid(args)

    tot = len(grid) * len(args.seeds) * len(pairs)
    print("=" * 70)
    print(f"mode={args.mode} pairs={pairs}")
    print(f"설정 {len(grid)} x seed {len(args.seeds)} x pair {len(pairs)} = {tot} cell")
    if args.clip_pool:
        print("!! --clip-pool 활성화: 주 결과와 섞지 마세요")
    print("=" * 70)

    res_path = outdir / "results.csv"
    rows, done = [], set()
    if res_path.exists():
        try:
            rows = pd.read_csv(res_path).to_dict("records")
            done = {(str(r.get("pair")), str(r.get("config")), str(r.get("seed")),
                     str(r.get("contam")), str(r.get("clip_pool")))
                    for r in rows}
            print(f"[재개] 기존 {len(rows)}행")
        except Exception:
            pass

    t0, n = time.time(), 0
    for pair in pairs:
        try:
            D = prepare(pair, ARGS)
        except Exception as e:
            print(f"[skip] {pair} prepare 실패: {type(e).__name__}: {e}")
            continue
        if args.clip_eval:
            D["eval_x"] = np.clip(np.asarray(D["eval_x"], dtype=np.float32), 0.0, 1.0)
        print(f"\n--- {pair} eval={len(D['eval_x'])}w 이상={D['eval_y'].mean():.4f} ---")

        for seed in args.seeds:
            ck = outdir / "ckpt" / \
                f"{pair}__s{seed}__h{N_HEADS}__lb{LB:g}__w{D['W']}.pt"
            try:
                tf = pretrain(D, seed, N_HEADS, LB, ARGS, device, ck)
            except Exception as e:
                print(f"  [skip] pretrain 실패 s{seed}: {e!r}"); continue
            model_cfg.n_heads = N_HEADS

            rng = np.random.default_rng(seed * 991 + int(args.contam * 100))
            pool, _ = E3.make_pool(D, args.contam,
                                   "real" if len(D["anom"]) else "synth", rng)
            if args.clip_pool:
                pool = np.clip(pool, 0.0, 1.0)
            cut = max(int(len(pool) * .8), 1)
            tr = pool[:cut]
            va = pool[cut:] if len(pool) - cut >= 5 else pool[-5:]
            ld = get_latent_dim(MODEL_NAME)

            for cname, spec in grid:
                key = (str(pair), str(cname), str(seed),
                       str(args.contam), str(int(args.clip_pool)))
                if key in done:
                    continue
                try:
                    E3.set_all_seeds(seed)
                    with AdaptTimer() as _t:
                        if spec["kind"] == "source_only":
                            m, npar = tf, 0
                        elif spec["kind"] == "adapter":
                            m = E3.fresh_adapted(tf, spec["cap"], ld, device)
                            E3.train_adapter(m, tr, va, bs, spec["epochs"],
                                             spec["lr"], device,
                                             patience=spec.get("patience", 8))
                            npar = sum(p.numel() for p in m.adapter.parameters())
                        else:
                            m = copy.deepcopy(tf)
                            full_finetune(
                                m,
                                torch.utils.data.DataLoader(torch.FloatTensor(tr),
                                                            batch_size=bs, shuffle=True),
                                torch.utils.data.DataLoader(torch.FloatTensor(va),
                                                            batch_size=bs, shuffle=False),
                                spec["epochs"], spec["lr"], device)
                            npar = sum(p.numel() for p in m.parameters())
                    s_te = tf_scores(m, D["eval_x"], bs, device)
                    if not np.isfinite(s_te).all():
                        raise ValueError("점수 NaN — 학습 발산")
                    r = evaluate_all(s_te, D["eval_y"], tf_scores(m, pool, bs, device))
                    np.save(outdir / "scores" /
                            f"{pair}__{cname}__c{args.contam:.2f}__a0__s{seed}.npy", s_te)
                    row = dict(stage="SW", mode=args.mode, pair=pair, seed=seed,
                               config=cname, method=cname, kind=spec["kind"],
                               capacity=spec["cap"], adapt_lr=spec["lr"],
                               epochs=spec["epochs"], patience=spec.get("patience", 8),
                               contam=args.contam, alpha=0.0,
                               clip_pool=int(args.clip_pool),
                               trainable_params=npar,
                               deploy_storage_kb=npar * 4 / 1024,
                               adapt_time_s=_t.elapsed, peak_mem_mb=_t.peak_mem_mb,
                               n_eval_windows=len(D["eval_x"]),
                               n_pool_windows=len(pool), diverged=0, **r)
                except Exception as e:
                    row = dict(stage="SW", mode=args.mode, pair=pair, seed=seed,
                               config=cname, method=cname, kind=spec["kind"],
                               capacity=spec["cap"], adapt_lr=spec["lr"],
                               epochs=spec["epochs"], patience=spec.get("patience", 8),
                               contam=args.contam, alpha=0.0,
                               clip_pool=int(args.clip_pool), diverged=1,
                               error=f"{type(e).__name__}: {e}")
                    print(f"    [FAIL] {cname}: {type(e).__name__}: {e}")
                    with open(outdir / "sweep_errors.log", "a", encoding="utf-8") as f:
                        f.write(f"\n=== {pair} s{seed} {cname}\n{traceback.format_exc()}")

                rows.append(row); done.add(key); n += 1
                pd.DataFrame(rows).to_csv(res_path, index=False)
                el = time.time() - t0
                print(f"  [{n}/{tot}] {pair} s{seed} {cname:22s} "
                      f"AUROC={row.get('auroc', float('nan')):.4f} "
                      f"p={row.get('trainable_params', 0)} "
                      f"({el/60:.0f}분, 잔여 {el/max(n,1)*(tot-n)/60:.0f}분)")

    if LF is not None:
        try:
            LF.dump_summary(outdir / "leakfix_summary.json")
        except Exception:
            pass

    print(f"\n[done] {res_path}  {len(rows)}행  {(time.time()-t0)/60:.1f}분")
    if args.mode == "tune":
        try:
            d = pd.DataFrame(rows)
            col = "auroc_nopool" if "auroc_nopool" in d.columns else "auroc"
            print("\n튜닝 pair 평균 (이 표로 설정 1개를 고르세요):")
            print(d.pivot_table(index="config", values=col, aggfunc="mean")
                   .sort_values(col, ascending=False).round(4).to_string())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

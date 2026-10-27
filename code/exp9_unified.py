# -*- coding: utf-8 -*-
"""
exp9_unified.py — 최종 통합 실험 (논문 결과표 확정용)
=======================================================
■ 왜 통합이 필요한가
  지금까지 exp1·exp3·exp5·exp8이 각각 다른 사전학습 루틴·epoch·stride를 써서
  같은 Pair의 절대값이 서로 다릅니다(예: XB1→XB2 adapter AUROC 0.998 vs 0.861).
  논문에는 단일 파이프라인의 결과표만 실어야 하므로, 모든 Pair·방법·Ablation을
  ★하나의 설정★으로 다시 계산합니다.

■ 포함 범위 (전부 한 번에)
  데이터  : 배관 Sim→Real · 베어링 cross-condition 4 · cross-machine 4
            · SMD 4 · Boiler 3 · MSL/SMAP (옵션)          → 최대 16+ Pair
  방법    : source_only / adapter_latent(제안) / adapter_output(TARD식)
            / adapter_input / full_ft / usad / tranad(옵션)
  Ablation: --nheads (1/3/5) · --lb (head 균형화) · --contam (오염 주입)
  지표    : AUROC · AUPR · AUPR-lift · F1@FAR(5/1/0.1%) · 효율(params·time)
  통계    : Pair 단위 집계 + Wilcoxon + Holm 보정 (다중비교 대응)

■ 사전 고정 (튜닝 금지)
  window: bench 100 / field 20 · FAR 1% · lb 0.1 · trim(ISCA) keep (1.0,0.7,0.5)
  정렬 손실 미포함 (4변형 반증 결과 반영)

실행
  python exp9_unified.py --quick                       # 파일럿(seed 1, 핵심 방법만)
  python exp9_unified.py --seeds 0 1 2                 # 본실험
  python exp9_unified.py --ablation nheads --nheads 1 3 5
  python exp9_unified.py --ablation contam --contam 0 0.1 0.2 0.4
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, copy, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats as sps

sys.path.insert(0, ".")
from config import model_cfg, train_cfg
from models import create_model
from run_cross_domain_adapter_v2 import (LatentAdapter, AdaptedModel, full_finetune,
                                         get_latent_dim, load_and_normalize, FEATURE_COLS)
from exp2_robust import sample_errors, scores as tf_scores, evaluate_all
from exp3_final import train_adapter, fresh_adapted, make_pool, set_all_seeds, MODEL_NAME, ISCA_KEEP
from exp4_final import _usage_entropy_loss, head_assign, register_msl_pairs
from exp1_efficiency import AdaptTimer, measure_inference_latency
from exp8_journal import OutputAdapted, InputAdapted, CROSS_MACHINE, pseudo_label_by_indicator
import exp1_benchmark as B
import exp1_labeling as L

# ── 사전 고정 ──────────────────────────────────────────────────────────────
W_BENCH, W_FIELD, STRIDE = 100, 20, 5
LB, FAR = 0.1, 0.01

FIELD_PAIRS = {                      # (kind, src, tgt)
    "Sim2Real":  ("pipe", "data/pipe/simCase1_pfei.csv", "data/pipe/realCase1_pfei.csv"),
    "XB1toXB2":  ("bearing", "data/xjtu_sy/XB1_combined.csv", "data/xjtu_sy/XB2_combined.csv"),
    "XB2toXB1":  ("bearing", "data/xjtu_sy/XB2_combined.csv", "data/xjtu_sy/XB1_combined.csv"),
    "PB1toPB2":  ("bearing", "data/phm2012/PB1_combined.csv", "data/phm2012/PB2_combined.csv"),
    "PB2toPB1":  ("bearing", "data/phm2012/PB2_combined.csv", "data/phm2012/PB1_combined.csv"),
}
for k, (s, t) in CROSS_MACHINE.items():
    FIELD_PAIRS[k] = ("bearing", s, t)
BENCH_DEFAULT = ["Boiler_1to2", "Boiler_2to3", "Boiler_3to1",
                 "SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2"]

from pipe_columns import SIM_COLS, REAL_COLS  # NDA: names read from work/data/pipe/columns.json


def _win(x, w, s=1):
    n = (len(x) - w) // s + 1
    idx = np.arange(w)[None, :] + s * np.arange(n)[:, None]
    return x[idx]


# ===========================================================================
# 통합 데이터 준비 — 모든 데이터셋이 동일한 구조를 반환
# ===========================================================================
def prepare(pair, args):
    if pair in FIELD_PAIRS:
        kind, s_csv, t_csv = FIELD_PAIRS[pair]
        W = W_FIELD
        if kind == "pipe":
            from sklearn.preprocessing import MinMaxScaler
            sd, td_ = pd.read_csv(s_csv), pd.read_csv(t_csv)
            src = MinMaxScaler().fit_transform(sd[SIM_COLS].values.astype(np.float32))
            tgt = MinMaxScaler().fit_transform(td_[REAL_COLS].values.astype(np.float32))
            pl = L.pipe_labels(len(tgt), L.PIPE_FAULT_IDX_DEFAULT)["labels"]
        else:
            src, tgt = load_and_normalize(s_csv, t_csv, FEATURE_COLS)
            pl = pseudo_label_by_indicator(t_csv, args.indicator, args.k, args.persistence)["labels"]
        lab_w = np.array([pl[i:i+W].max() for i in range(0, len(tgt)-W+1, STRIDE)])
        t_all = _win(tgt, W, STRIDE)
        return dict(pair=pair, family="field", W=W, n_feat=src.shape[1],
                    src=_win(src, W, STRIDE), clean=t_all[lab_w == 0], anom=t_all[lab_w == 1],
                    eval_x=_win(tgt, W, 1),
                    eval_y=L.point_to_window_labels(pl, W, rule="any")[:len(tgt)-W+1])
    # bench
    W = W_BENCH
    cfg = B.BENCH_PAIRS[pair]
    S, T = B.load_pair(cfg)
    sc = B.MinMax().fit(S["train"])
    s_tr, t_tr, t_te = sc(S["train"]), sc(T["train"]), sc(T["test"])
    src = _win(s_tr, W, STRIDE)
    keep = B.win_healthy(S["train_label"] == 0, W, STRIDE)
    if keep.any():
        src = src[keep]
    t_all = _win(t_tr, W, STRIDE)
    t_lab = B.win_labels(T["train_label"], W, STRIDE)
    return dict(pair=pair, family="bench", W=W, n_feat=s_tr.shape[1], src=src,
                clean=t_all[t_lab == 0], anom=t_all[t_lab == 1],
                eval_x=_win(t_te, W, 1), eval_y=B.win_labels(T["test_label"], W, 1))


# ===========================================================================
# 통합 사전학습 (단일 루틴 — 파이프라인 일관성의 핵심)
# ===========================================================================
def pretrain(D, seed, n_heads, lb, args, device, ck: Path):
    model_cfg.n_heads = n_heads; model_cfg.input_size = D["n_feat"]
    model_cfg.window_size = D["W"]; model_cfg.d_model = 128; model_cfg.dim_ff = 512
    if ck.exists() and not args.force_pretrain:
        m = create_model(MODEL_NAME, model_cfg)
        m.load_state_dict(torch.load(ck, map_location=device)); return m.to(device)
    set_all_seeds(seed)
    m = create_model(MODEL_NAME, model_cfg).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=train_cfg.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=.5)
    n = int(len(D["src"]) * .8); tr, va = D["src"][:n], D["src"][n:]
    bs = train_cfg.batch_size; best, st, wait = float("inf"), None, 0
    for _ in range(args.epochs_pretrain):
        m.train()
        for i in range(0, len(tr), bs):
            idx = np.random.permutation(len(tr))[i:i+bs] if i == 0 else None
            xb = tr[np.random.choice(len(tr), min(bs, len(tr)), replace=False)]
            x = torch.FloatTensor(xb).to(device)
            if len(x) < 2:
                continue
            h = m(x); per, _ = sample_errors(h, x)
            loss = per.mean() + (lb * _usage_entropy_loss(h, x, n_heads) if lb > 0 else 0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            v = np.mean([sample_errors(m(torch.FloatTensor(va[i:i+bs]).to(device)),
                                       torch.FloatTensor(va[i:i+bs]).to(device))[0].mean().item()
                         for i in range(0, len(va), bs)]) if len(va) else np.inf
        sch.step(v)
        if v < best - 1e-7:
            best, wait, st = v, 0, copy.deepcopy(m.state_dict())
        else:
            wait += 1
            if wait >= args.patience:
                break
    if st:
        m.load_state_dict(st)
    ck.parent.mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), ck)
    return m


# ===========================================================================
def build_method(name, tf, D, pool, args, device):
    bs = train_cfg.batch_size; ld = get_latent_dim(MODEL_NAME)
    cut = max(int(len(pool) * .8), 1)
    tr, va = pool[:cut], pool[cut:] if len(pool)-cut >= 5 else pool[-5:]
    if name == "source_only":
        return tf, 0
    if name == "adapter_latent":
        m = fresh_adapted(tf, "bottleneck", ld, device)
    elif name == "adapter_output":
        m = OutputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
    elif name == "adapter_input":
        m = InputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
    elif name == "adapter_isca":
        cur, m = pool, None
        for keep in ISCA_KEEP:
            if keep < 1.0 and m is not None:
                sc = tf_scores(m, pool, bs, device)
                cur = pool[np.argsort(sc)[:max(int(len(pool)*keep), 10)]]
            c2 = max(int(len(cur)*.8), 1)
            m = fresh_adapted(tf, "bottleneck", ld, device)
            train_adapter(m, cur[:c2], cur[c2:] if len(cur)-c2 >= 5 else cur[-5:],
                          bs, args.epochs_adapt, args.adapt_lr, device)
        return m, sum(p.numel() for p in m.adapter.parameters())
    elif name == "full_ft":
        m = copy.deepcopy(tf)
        full_finetune(m, torch.utils.data.DataLoader(torch.FloatTensor(tr), batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(va), batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
        return m, sum(p.numel() for p in m.parameters())
    else:
        raise ValueError(name)
    train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
    return m, sum(p.numel() for p in m.adapter.parameters())


def run_cell(D, seed, meth, nh, lb, contam, args, device, out):
    ck = out / "ckpt" / f"{D['pair']}__s{seed}__h{nh}__lb{lb:g}__w{D['W']}.pt"
    tf = pretrain(D, seed, nh, lb, args, device, ck)
    model_cfg.n_heads = nh
    rng = np.random.default_rng(seed*991 + int(contam*100))
    pool, pgt = make_pool(D, contam, "real" if len(D["anom"]) else "synth", rng)
    set_all_seeds(seed)
    with AdaptTimer() as _t:
        m, npar = build_method(meth, tf, D, pool, args, device)
    el, peak = _t.elapsed, _t.peak_mem_mb
    bs = train_cfg.batch_size
    s_te = tf_scores(m, D["eval_x"], bs, device)
    s_cal = tf_scores(m, pool, bs, device)
    r = evaluate_all(s_te, D["eval_y"], s_cal)
    h_ev, _ = head_assign(m, D["eval_x"], bs, device)
    # ── 배포 효율: 추론 지연 · 저장 용량 ──────────────────────────────
    from models import select_best_head as _sbh
    @torch.no_grad()
    def _fn(x):
        hh = m(x); b, _, _ = _sbh(hh, x)
        return torch.mean(torch.abs(b - x), dim=(1, 2))
    _ld = torch.utils.data.DataLoader(torch.FloatTensor(D["eval_x"]),
                                      batch_size=bs, shuffle=False)
    lat_ms = measure_inference_latency(_fn, _ld, device)
    store_kb = npar * 4 / 1024.0        # fp32 기준 배포 저장량
    np.save(out / "scores" / f"{D['pair']}__{meth}__h{nh}__lb{lb:g}__c{contam:.2f}__s{seed}.npy", s_te)
    print(f"    [{meth:15s} h={nh} lb={lb:g} c={contam:.2f}] AUROC={r['auroc']:.4f} "
          f"lift={r['aupr_lift']:.2f} F1={r['f1_far0.01']:.4f} "
          f"p={npar} {store_kb:.0f}KB {lat_ms:.3f}ms ({el:.0f}s)")
    return dict(pair=D["pair"], family=D["family"], seed=seed, method=meth,
                n_heads=nh, lb=lb, contam=contam, contam_actual=float(pgt.mean()),
                trainable_params=npar, adapt_time_s=el, peak_mem_mb=peak,
                infer_ms_per_window=lat_ms, deploy_storage_kb=store_kb,
                heads_used=int((np.bincount(h_ev, minlength=nh)/max(len(h_ev),1) > .01).sum()),
                **r)


# ===========================================================================
def analyze(f):
    df = pd.read_csv(f)
    print(f"\n{'='*78}\n=== Pair × Method 요약 (AUROC) ===")
    print(df[df.contam == 0].pivot_table(index="pair", columns="method",
                                         values="auroc").round(4).to_string())
    base = df[df.method == "source_only"].groupby(["pair", "seed", "contam", "n_heads"])["auroc"].mean()
    g = df[df.method != "source_only"].copy()
    g["d"] = g.apply(lambda r: r["auroc"] - base.get(
        (r["pair"], r["seed"], r["contam"], r["n_heads"]), np.nan), axis=1)
    print("\n=== source_only 대비 (Pair 단위 · Wilcoxon · Holm 보정) ===")
    rows = []
    for (meth, c), gg in g[g.contam == 0].groupby(["method", "contam"]):
        v = gg.groupby("pair")["d"].mean().dropna()
        if len(v) < 3:
            continue
        try:
            pv = sps.wilcoxon(v)[1]
        except Exception:
            pv = np.nan
        rows.append(dict(method=meth, n_pairs=len(v), mean=v.mean(), median=v.median(),
                         win=f"{int((v>0).sum())}/{len(v)}", p=pv))
    if rows:
        s = pd.DataFrame(rows).sort_values("p")
        m = len(s)
        s["p_holm"] = np.minimum.accumulate((s["p"].values*(m-np.arange(m)))[::-1])[::-1].clip(max=1)
        s["sig"] = s["p_holm"] < .05
        print(s.round(4).to_string(index=False))
        print(f"\n  ※ Wilcoxon 최소 p: n=4→0.125, n=6→0.031, n=8→0.008 "
              f"(Pair 수가 검정력을 결정)")
    if df.contam.nunique() > 1:
        print("\n=== 오염 dose-response (AUROC) ===")
        print(df.pivot_table(index="contam", columns="method", values="auroc").round(4).to_string())
    if df.n_heads.nunique() > 1:
        print("\n=== Head 수 Ablation ===")
        print(df.pivot_table(index="n_heads", columns="method",
                             values=["auroc", "f1_far0.01"]).round(4).to_string())
    print("\n=== 배포 효율 (전체 평균) ===")
    cols = [c for c in ["trainable_params", "deploy_storage_kb", "adapt_time_s",
                        "infer_ms_per_window", "peak_mem_mb"] if c in df]
    print(df.groupby("method")[cols].mean().round(3).to_string())
    if "infer_ms_per_window" in df:
        print("\n=== 시나리오별 효율 (family = 데이터 성격) ===")
        print("  field  = 배관·베어링 (열화형, 설비 수가 많음 → 저장·배포 비용 지배)")
        print("  bench  = SMD·Boiler (반복 고장형, 상시 모니터링 → 추론 지연 지배)")
        t = df.groupby(["family", "method"])[cols].mean().round(3)
        print(t.to_string())
        base = df[df.method == "source_only"].groupby("family")["infer_ms_per_window"].mean()
        ov = df[df.method != "source_only"].groupby(["family", "method"])["infer_ms_per_window"].mean()
        print("\n  추론 오버헤드 (source_only 대비 %):")
        for (fam, meth), v in ov.items():
            b = base.get(fam, np.nan)
            print(f"    {fam:6s} {meth:16s} {100*(v-b)/max(b,1e-9):+6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--field", action="store_true", help="배관·베어링만")
    ap.add_argument("--bench", action="store_true", help="공개 benchmark만")
    ap.add_argument("--msl_pairs", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="*",
                    default=["source_only", "adapter_latent", "adapter_output",
                             "adapter_input", "adapter_isca", "full_ft"])
    ap.add_argument("--ablation", choices=["none", "nheads", "contam"], default="none")
    ap.add_argument("--nheads", nargs="*", type=int, default=[5])
    ap.add_argument("--contam", nargs="*", type=float, default=[0.0])
    ap.add_argument("--indicator", default="rms")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--epochs_pretrain", type=int, default=120)
    ap.add_argument("--epochs_adapt", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp9_out")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds = [0]
        args.methods = ["source_only", "adapter_latent", "adapter_output", "full_ft"]
        args.epochs_pretrain, args.epochs_adapt = 40, 20

    out = Path(args.out); (out/"ckpt").mkdir(parents=True, exist_ok=True); (out/"scores").mkdir(exist_ok=True)
    device = train_cfg.device
    if args.msl_pairs:
        register_msl_pairs([tuple(x.split(":")) for x in args.msl_pairs])
    if args.pairs:
        pairs = args.pairs
    else:
        pairs = []
        if not args.bench:
            pairs += list(FIELD_PAIRS)
        if not args.field:
            pairs += BENCH_DEFAULT + [f"MSL_{s}to{t}" for s, t in
                                      [tuple(x.split(":")) for x in args.msl_pairs]]
    nhs = args.nheads if args.ablation == "nheads" else [args.nheads[0]]
    cts = args.contam if args.ablation == "contam" else [args.contam[0]]
    print(f"[exp9] pairs={len(pairs)}개 methods={args.methods} seeds={args.seeds} "
          f"nheads={nhs} contam={cts}\n       {pairs}")

    f = out / "results.csv"
    if not args.analyze_only:
        rows = pd.read_csv(f).to_dict("records") if f.exists() else []
        for pair in pairs:
            try:
                D = prepare(pair, args)
            except Exception as e:
                print(f"  [skip] {pair}: {type(e).__name__}: {e}"); continue
            print(f"\n{'━'*64}\n  {pair} [{D['family']}] clean={len(D['clean'])}w "
                  f"eval={len(D['eval_x'])}w 이상={D['eval_y'].mean():.3f}")
            for seed in args.seeds:
                for nh in nhs:
                    lb = 0.0 if nh == 1 else LB
                    for c in cts:
                        for meth in args.methods:
                            try:
                                rows.append(run_cell(D, seed, meth, nh, lb, c, args, device, out))
                                pd.DataFrame(rows).to_csv(f, index=False)
                            except Exception as e:
                                print(f"    [ERR {meth}] {type(e).__name__}: {e}")
    analyze(f)
    print(f"\n[exp9] → {f.resolve()}")


if __name__ == "__main__":
    main()

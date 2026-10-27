# -*- coding: utf-8 -*-
"""
exp5_final.py — 최종 제안 방법 + 전체 Ablation (벤치마크 + 배관/베어링 통합)
============================================================================
최종 방법 RCA (Robust Cross-domain Adaptation)
  (1) Source 사전학습 + head load-balancing  (collapse 방지, lb 고정)
  (2) Target 적응: latent adapter(1%) + ISCA 반복 자가정제 (오염 내성)
  (3) 탐지: 운전모드별 FAR 제어 threshold (calibration 실패 구제)
  ※ 정렬(MMD)은 포함하지 않음 — exp3/exp4에서 4변형 모두 반증됨

전 데이터셋 지원: SMD · Boiler · MSL/SMAP · 배관(Sim→Real) · 베어링(XJTU/PHM)

Ablation (--ablate):
  full          최종 방법 전체
  no_balance    (1) 제거 → head collapse 상태
  no_isca       (2)의 자가정제 제거 (단순 adapter)
  no_permode    (3) 제거 → 전역 threshold
  single_head   n_heads=1 (multi-head 자체의 필요성 검증) ★교수님 지적
  nhead_K       n_heads=K 스윕 (K=1,3,5,7) ★"왜 5개인가"
  source_only   적응 없음
  full_ft       전체 미세조정

라벨 누수 방지: 모든 하이퍼파라미터 사전 고정. GT는 평가·오염조건 구성·진단에만 사용.

실행:
  python exp5_final.py --ablate full no_balance no_isca no_permode source_only full_ft
  python exp5_final.py --family field --ablate full no_isca source_only
  python exp5_final.py --ablate nhead --nheads 1 3 5 7
  python exp5_final.py --contam 0.0 0.2 0.4 --ablate full no_isca   # 오염 내성
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, copy, time, random, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats as sps
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from config import model_cfg, train_cfg
from models import create_model
from run_cross_domain_adapter_v2 import LatentAdapter, AdaptedModel, full_finetune, get_latent_dim
import exp1_benchmark as B
from exp2_robust import sample_errors, scores, evaluate_all
from exp3_final import (STRIDE, CAPACITY, train_adapter, fresh_adapted, make_pool,
                        set_all_seeds, MODEL_NAME)
from exp4_final import _usage_entropy_loss, head_assign, register_msl_pairs

# ── 사전 고정 하이퍼파라미터 ────────────────────────────────────────────────
LB = 0.1                     # head 균형화 계수 (balance 실험에서 head 4.67/5, val loss 최소 구간)
ISCA_KEEP = (1.0, 0.7, 0.5)
FAR = 0.01
MIN_MODE_N = 30
W_BENCH, W_FIELD = 100, 20   # 프로토콜 고정 (DACAD / 본 연구 선행설정)

FIELD_PAIRS = ["Sim2Real", "XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1"]


# ===========================================================================
def prepare_any(pair, family, k=3.0, persistence=5):
    """bench/field 통합 데이터 준비 → clean/anom/eval 구조로 반환."""
    if family == "bench":
        W = W_BENCH
        cfg = B.BENCH_PAIRS[pair]
        S, T = B.load_pair(cfg)
        sc = B.MinMax().fit(S["train"])
        s_tr, t_tr, t_te = sc(S["train"]), sc(T["train"]), sc(T["test"])
        src = B.make_win(s_tr, W, STRIDE)
        keep = B.win_healthy(S["train_label"] == 0, W, STRIDE)
        if keep.any():
            src = src[keep]
        t_all = B.make_win(t_tr, W, STRIDE)
        t_lab = B.win_labels(T["train_label"], W, STRIDE)
        ev_x = B.make_win(t_te, W, 1)
        ev_y = B.win_labels(T["test_label"], W, 1)
        return dict(src=src, clean=t_all[t_lab == 0], anom=t_all[t_lab == 1],
                    eval_x=ev_x, eval_y=ev_y, n_feat=s_tr.shape[1], W=W)
    # ── field (배관 / 베어링) ──
    W = W_FIELD
    import run_experiment1 as R
    import exp1_labeling as L
    cfg = R.DOMAIN_PAIRS[pair]
    src_x, tgt_x, n_feat = R.load_pair(cfg)
    if cfg["kind"] == "pipe":
        res = L.pipe_labels(len(tgt_x), L.PIPE_FAULT_IDX_DEFAULT)
    else:
        res = L.bearing_pseudo_labels(cfg["tgt"], k=k, persistence=persistence)
    pl = res["labels"]

    def win(x, s=STRIDE):
        n = (len(x) - W) // s + 1
        idx = np.arange(W)[None, :] + s * np.arange(n)[:, None]
        return x[idx]

    lab_w = np.array([pl[i:i + W].max() for i in range(0, len(tgt_x) - W + 1, STRIDE)])
    t_all = win(tgt_x)
    ev_x = win(tgt_x, 1)
    ev_y = L.point_to_window_labels(pl, W, rule="any")[:len(ev_x)]
    return dict(src=win(src_x), clean=t_all[lab_w == 0], anom=t_all[lab_w == 1],
                eval_x=ev_x, eval_y=ev_y, n_feat=n_feat, W=W)


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
    n = int(len(D["src"]) * 0.8); tr, va = D["src"][:n], D["src"][n:]
    bs = train_cfg.batch_size
    best, state, wait = float("inf"), None, 0
    for _ in range(args.epochs_pretrain):
        m.train()
        for i in np.array_split(np.random.permutation(len(tr)), max(len(tr)//bs, 1)):
            if len(i) < 2:
                continue
            x = torch.FloatTensor(tr[i]).to(device)
            h = m(x); per, _ = sample_errors(h, x)
            loss = per.mean() + (lb * _usage_entropy_loss(h, x, n_heads) if lb > 0 else 0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            v = np.mean([sample_errors(m(torch.FloatTensor(va[i:i+bs]).to(device)),
                                       torch.FloatTensor(va[i:i+bs]).to(device))[0].mean().item()
                         for i in range(0, len(va), bs)]) if len(va) else np.inf
        if v < best - 1e-6:
            best, wait, state = v, 0, copy.deepcopy(m.state_dict())
        else:
            wait += 1
            if wait >= 10:
                break
    if state:
        m.load_state_dict(state)
    ck.parent.mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), ck)
    return m


def adapt(model, pool, args, device, use_isca):
    ld = get_latent_dim(MODEL_NAME); bs = train_cfg.batch_size
    if not use_isca:
        cut = max(int(len(pool) * .8), 1)
        ad = fresh_adapted(model, "bottleneck", ld, device)
        train_adapter(ad, pool[:cut], pool[cut:] if len(pool)-cut >= 5 else pool[-5:],
                      bs, args.epochs_adapt, args.adapt_lr, device)
        return ad
    cur, ad = pool, None
    for keep in ISCA_KEEP:
        if keep < 1.0 and ad is not None:
            s = scores(ad, pool, bs, device)
            cur = pool[np.argsort(s)[:max(int(len(pool)*keep), 10)]]
        cut = max(int(len(cur) * .8), 1)
        ad = fresh_adapted(model, "bottleneck", ld, device)
        train_adapter(ad, cur[:cut], cur[cut:] if len(cur)-cut >= 5 else cur[-5:],
                      bs, args.epochs_adapt, args.adapt_lr, device)
    return ad


def detect(ad, cal_pool, D, device, per_mode, n_heads):
    bs = train_cfg.batch_size
    ch, cs = head_assign(ad, cal_pool, bs, device)
    eh, es = head_assign(ad, D["eval_x"], bs, device)
    thr_g = float(np.quantile(cs, 1 - FAR))
    if not per_mode:
        pred = (es >= thr_g).astype(int)
    else:
        pred = np.zeros(len(es), int)
        for k in range(n_heads):
            sk = cs[ch == k]
            t = float(np.quantile(sk, 1 - FAR)) if len(sk) >= MIN_MODE_N else thr_g
            pred[eh == k] = (es[eh == k] >= t).astype(int)
    y = D["eval_y"]
    r = evaluate_all(es, y, cs)
    r.update(f1_used=f1_score(y, pred, zero_division=0),
             prec_used=precision_score(y, pred, zero_division=0),
             rec_used=recall_score(y, pred, zero_division=0),
             n_heads_used=int((np.bincount(eh, minlength=n_heads) /
                               max(len(eh), 1) > 0.01).sum()))
    return r


# ===========================================================================
def run(pair, family, seed, ab, contam, args, device, out):
    D = prepare_any(pair, family, args.k, args.persistence)
    nh = args.nheads[0] if ab.startswith("nhead") else (1 if ab == "single_head" else 5)
    if ab.startswith("nhead_"):
        nh = int(ab.split("_")[1])
    lb = 0.0 if ab in ("no_balance", "single_head") or nh == 1 else LB
    ck = Path(args.out) / "ckpt" / f"{pair}__s{seed}__h{nh}__lb{lb:g}__w{D['W']}.pt"
    model = pretrain(D, seed, nh, lb, args, device, ck)
    model_cfg.n_heads = nh

    rng = np.random.default_rng(seed * 991 + int(contam * 100))
    pool, pgt = make_pool(D, contam, "real" if len(D["anom"]) else "synth", rng)
    t0 = time.perf_counter()
    if ab == "source_only":
        ad = model
    elif ab == "full_ft":
        ad = copy.deepcopy(model); bs = train_cfg.batch_size
        cut = max(int(len(pool)*.8), 1)
        full_finetune(ad, torch.utils.data.DataLoader(torch.FloatTensor(pool[:cut]), batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(pool[cut:]), batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
    else:
        ad = adapt(model, pool, args, device, use_isca=(ab not in ("no_isca",)))
    el = time.perf_counter() - t0
    per_mode = ab not in ("no_permode", "single_head") and nh > 1
    r = detect(ad, pool, D, device, per_mode, nh)
    tot = sum(p.numel() for p in ad.parameters())
    tra = sum(p.numel() for p in ad.parameters() if p.requires_grad) if ab != "source_only" else 0
    rec = dict(family=family, pair=pair, seed=seed, ablation=ab, n_heads=nh, lb=lb,
               contam=contam, contam_actual=float(pgt.mean()), per_mode=per_mode,
               trainable_params=tra, trainable_pct=100*tra/max(tot, 1),
               adapt_time_s=el, **r)
    print(f"    [{ab:12s} h={nh} c={contam:.2f}] AUROC={r['auroc']:.4f} "
          f"AUPR-lift={r['aupr_lift']:.2f} F1={r['f1_used']:.4f} heads={r['n_heads_used']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["bench", "field", "both"], default="bench")
    ap.add_argument("--datasets", nargs="*", default=["boiler", "smd"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--msl_pairs", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--ablate", nargs="*", default=["full", "no_balance", "no_isca",
                                                    "no_permode", "source_only", "full_ft"])
    ap.add_argument("--nheads", nargs="*", type=int, default=[5])
    ap.add_argument("--contam", nargs="*", type=float, default=[0.0])
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--epochs_pretrain", type=int, default=60)
    ap.add_argument("--epochs_adapt", type=int, default=30)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp5_out")
    ap.add_argument("--force_pretrain", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    device = train_cfg.device
    if args.msl_pairs:
        register_msl_pairs([tuple(p.split(":")) for p in args.msl_pairs])

    tasks = []
    if args.pairs:
        for p in args.pairs:
            tasks.append((p, "field" if p in FIELD_PAIRS else "bench"))
    else:
        if args.family in ("bench", "both"):
            from exp3_final import PAIRS as BP
            tasks += [(p, "bench") for d in args.datasets if d in BP for p in BP[d]]
            tasks += [(f"MSL_{s}to{t}", "bench") for s, t in
                      [tuple(x.split(":")) for x in args.msl_pairs]]
        if args.family in ("field", "both"):
            tasks += [(p, "field") for p in FIELD_PAIRS]

    ablist = []
    for a in args.ablate:
        ablist += [f"nhead_{k}" for k in args.nheads] if a == "nhead" else [a]
    print(f"[exp5] tasks={[t[0] for t in tasks]} ablate={ablist} "
          f"contam={args.contam} seeds={args.seeds} (lb={LB}, FAR={FAR} 고정)")

    rows, f = [], out / "results.csv"
    if f.exists():
        rows = pd.read_csv(f).to_dict("records")
    for pair, fam in tasks:
        print(f"\n{'━'*62}\n  {pair} [{fam}]")
        for seed in args.seeds:
            print(f"  ── seed {seed}")
            for c in args.contam:
                for ab in ablist:
                    try:
                        rows.append(run(pair, fam, seed, ab, c, args, device, out))
                        pd.DataFrame(rows).to_csv(f, index=False)
                    except Exception as e:
                        print(f"    [ERR {ab}] {e}")
    analyze(f)


def analyze(f):
    if not Path(f).exists():
        return
    df = pd.read_csv(f)
    print(f"\n{'='*72}\n=== 요약 (pair 평균) ===")
    print(df.groupby(["family", "ablation", "contam"])[
        ["auroc", "aupr_lift", "f1_used", "trainable_pct", "n_heads_used"]]
        .mean().round(4).to_string())
    base = df[df.ablation == "source_only"].groupby(["pair", "seed", "contam"])["auroc"].mean()
    g = df[df.ablation != "source_only"].copy()
    g["d"] = g.apply(lambda r: r["auroc"] - base.get((r["pair"], r["seed"], r["contam"]), np.nan), axis=1)
    print("\n=== source_only 대비 개선 (pair 단위 주분석) ===")
    for (ab, c), gg in g.groupby(["ablation", "contam"]):
        v = gg.groupby("pair")["d"].mean().dropna()
        if len(v) >= 3:
            try:
                p = sps.wilcoxon(v)[1]
            except Exception:
                p = np.nan
            print(f"  {ab:12s} c={c:.2f} mean={v.mean():+.4f} med={v.median():+.4f} "
                  f"win={int((v>0).sum())}/{len(v)} p={p:.4f}")
    print(f"\n[exp5] → {Path(f).resolve()}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
exp8_journal.py — 저널 투고용 잔여 실험 3종
=============================================
지금까지 미실시였던, 그러나 심사에서 반드시 요구되는 항목만 모았다.

 [stage crossmachine]  Domain Shift 3수준 완성
        cross-condition(완료) · sim-to-real(완료) · ★cross-machine(미실시)
        XJTU-SY ↔ PHM2012 (feature 8개 동일 → 구조 변경 불필요)
        "동일 Adapter가 3수준 전이에서 일관되게 작동하는가"

 [stage position]      Adapter 위치 ablation  ★교수님 지적
        latent(제안) vs output(TARD 방식의 출력 보정) vs input
        → TARD는 '출력단 Δx 보정', 본 연구는 'latent 보정'.
          같은 파이프라인에서 직접 비교해야 차별성이 입증된다.

 [stage indicator]     Pseudo Label 상태지표 ablation  ★교수님 지적
        RMS만 쓴 이유는? kurtosis·peak·복합 지표로 라벨을 다시 만들어
        method 순위가 유지되는지 확인 (저장된 score 재사용 → 재학습 불필요, 수분)

실행
  python exp8_journal.py --stage indicator                    # 가장 빠름(수분)
  python exp8_journal.py --stage position --seeds 0 1 2
  python exp8_journal.py --stage crossmachine --seeds 0 1 2
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
from models import create_model, select_best_head
from run_cross_domain_adapter_v2 import (LatentAdapter, AdaptedModel, full_finetune,
                                         get_latent_dim, load_and_normalize, FEATURE_COLS)
from exp2_robust import sample_errors, scores as tf_scores, evaluate_all
from exp3_final import train_adapter, fresh_adapted, set_all_seeds, MODEL_NAME, STRIDE
from exp5_final import pretrain as tf_pretrain, W_FIELD
import exp1_labeling as L

W = W_FIELD          # 배관·베어링 프로토콜 고정
FAR = 0.01

# ── cross-machine pair 정의 (feature 8개 공통) ─────────────────────────────
CROSS_MACHINE = {
    "XB1toPB1": ("data/xjtu_sy/XB1_combined.csv", "data/phm2012/PB1_combined.csv"),
    "XB2toPB2": ("data/xjtu_sy/XB2_combined.csv", "data/phm2012/PB2_combined.csv"),
    "PB1toXB1": ("data/phm2012/PB1_combined.csv", "data/xjtu_sy/XB1_combined.csv"),
    "PB2toXB2": ("data/phm2012/PB2_combined.csv", "data/xjtu_sy/XB2_combined.csv"),
}


def win(x, w=W, s=STRIDE):
    n = (len(x) - w) // s + 1
    idx = np.arange(w)[None, :] + s * np.arange(n)[:, None]
    return x[idx]


def bearing_data(src_csv, tgt_csv, k=3.0, persistence=5, indicator="rms"):
    src, tgt = load_and_normalize(src_csv, tgt_csv, FEATURE_COLS)
    res = pseudo_label_by_indicator(tgt_csv, indicator, k, persistence)
    pl = res["labels"]
    lab_w = np.array([pl[i:i + W].max() for i in range(0, len(tgt) - W + 1, STRIDE)])
    t_all = win(tgt)
    return dict(src=win(src), clean=t_all[lab_w == 0], anom=t_all[lab_w == 1],
                eval_x=win(tgt, W, 1),
                eval_y=L.point_to_window_labels(pl, W, rule="any")[:len(tgt) - W + 1],
                n_feat=src.shape[1], W=W, onsets=res.get("onsets"))


# ===========================================================================
# [indicator] 상태지표별 pseudo label
# ===========================================================================
IND_COLS = {"rms": ["rms_h", "rms_v"],
            "kurtosis": ["kurtosis_h", "kurtosis_v"],
            "peak": ["peak_h", "peak_v"],
            "std": ["std_h", "std_v"]}


def pseudo_label_by_indicator(csv, indicator="rms", k=3.0, persistence=5):
    """RMS 외 지표(kurtosis/peak/std)나 복합 지표로 onset 산출."""
    if indicator == "combined":
        # 여러 지표의 정규화 z-score 최대값을 상태지표로 사용
        df = pd.read_csv(csv)
        segs, _ = L.resolve_segments(len(df))
        labels = np.zeros(len(df), int); pos = 0; onsets = []
        for Lg in segs:
            sub = df.iloc[pos:pos + Lg]
            h = max(int(Lg * 0.1), 10)
            hi = None
            for cols in IND_COLS.values():
                v = sub[cols].to_numpy(float)
                mu, sd = v[:h].mean(0), v[:h].std(0, ddof=1) + 1e-9
                z = ((v - mu) / sd).max(1)
                hi = z if hi is None else np.maximum(hi, z)
            m, s = hi[:h].mean(), hi[:h].std(ddof=1) + 1e-9
            ex = hi > m + k * s
            run = np.convolve(ex[h:].astype(int), np.ones(persistence, int), "valid")
            idx = np.flatnonzero(run == persistence)
            on = int(h + idx[0]) if len(idx) else None
            if on is not None:
                labels[pos + on:pos + Lg] = 1
            onsets.append(on); pos += Lg
        return {"labels": labels, "onsets": onsets}
    r = L.bearing_pseudo_labels(csv, k=k, persistence=persistence,
                                rms_cols=IND_COLS[indicator])
    return {"labels": r["labels"], "onsets": [m["onset_local"] for m in r["segment_meta"]]}


# ===========================================================================
# [position] Adapter 위치 변형
# ===========================================================================
class OutputAdapted(nn.Module):
    """TARD식 출력단 보정: 재구성 출력에 Δx를 더한다 (window 요약으로부터 예측)."""
    def __init__(self, base, n_feat, hidden=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.g = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(),
                               nn.Linear(hidden, n_feat))
        nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)
        self.adapter = self.g      # train_adapter 호환

    def forward(self, x):
        heads = self.base(x)                        # (N,B,W,F)
        d = self.g(x.mean(dim=1))                   # [B,F]
        return heads + d.unsqueeze(0).unsqueeze(2)

    def encode_to_latent(self, x):
        return self.base.transformer_enc(self.base.enc_pos(self.base.enc_proj(x))).mean(1)


class InputAdapted(nn.Module):
    """입력단 보정: 입력 신호 자체를 보정한 뒤 frozen 모델 통과."""
    def __init__(self, base, n_feat, hidden=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.g = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(),
                               nn.Linear(hidden, n_feat))
        nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)
        self.adapter = self.g

    def forward(self, x):
        return self.base(x + self.g(x))

    def encode_to_latent(self, x):
        z = x + self.g(x)
        return self.base.transformer_enc(self.base.enc_pos(self.base.enc_proj(z))).mean(1)


# ===========================================================================
def evaluate_model(m, D, pool, device):
    bs = train_cfg.batch_size
    s_te = tf_scores(m, D["eval_x"], bs, device)
    s_cal = tf_scores(m, pool, bs, device)
    r = evaluate_all(s_te, D["eval_y"], s_cal)
    return r, s_te


def run_one(pair, D, seed, method, args, device, out):
    bs = train_cfg.batch_size
    ck = out / "ckpt" / f"{pair}__s{seed}.pt"
    model_cfg.n_heads = 5
    tf = tf_pretrain(D, seed, 5, 0.0, args, device, ck)
    ld = get_latent_dim(MODEL_NAME)
    pool = D["clean"]
    cut = max(int(len(pool) * 0.8), 1)
    tr, va = pool[:cut], pool[cut:] if len(pool) - cut >= 5 else pool[-5:]

    t0 = time.perf_counter()
    set_all_seeds(seed)
    if method == "source_only":
        m = tf
    elif method == "adapter_latent":
        m = fresh_adapted(tf, "bottleneck", ld, device)
        train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
    elif method == "adapter_output":
        m = OutputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
        train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
    elif method == "adapter_input":
        m = InputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
        train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
    elif method == "full_ft":
        m = copy.deepcopy(tf)
        full_finetune(m, torch.utils.data.DataLoader(torch.FloatTensor(tr), batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(va), batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
    else:
        raise ValueError(method)
    el = time.perf_counter() - t0
    r, s_te = evaluate_model(m, D, pool, device)
    tra = (0 if method == "source_only" else
           sum(p.numel() for p in m.parameters() if p.requires_grad))
    np.save(out / "scores" / f"{pair}__{method}__s{seed}.npy", s_te)
    print(f"    [{method:15s}] AUROC={r['auroc']:.4f} AUPR-lift={r['aupr_lift']:.2f} "
          f"F1={r['f1_far0.01']:.4f} params={tra} ({el:.0f}s)")
    return dict(pair=pair, seed=seed, method=method, trainable_params=tra,
                adapt_time_s=el, **r)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["crossmachine", "position", "indicator"],
                    default="indicator")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--indicators", nargs="*",
                    default=["rms", "kurtosis", "peak", "std", "combined"])
    ap.add_argument("--score_dir", default="exp8_out/scores",
                    help="indicator stage: 재사용할 저장 score 위치")
    ap.add_argument("--epochs_pretrain", type=int, default=60)
    ap.add_argument("--epochs_adapt", type=int, default=30)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp8_out")
    ap.add_argument("--force_pretrain", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    device = train_cfg.device

    # ── indicator: 재학습 없이 라벨만 바꿔 재평가 ──────────────────────────
    if args.stage == "indicator":
        rows = []
        import run_experiment1 as R
        pairs = args.pairs or ["XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1"]
        for pair in pairs:
            cfg = R.DOMAIN_PAIRS.get(pair)
            if cfg is None or cfg["kind"] != "bearing":
                continue
            print(f"\n  {pair}")
            for ind in args.indicators:
                res = pseudo_label_by_indicator(cfg["tgt"], ind)
                wl = L.point_to_window_labels(res["labels"], W, rule="any")
                for f in sorted((Path(args.score_dir)).glob(f"{pair}__*__s*.npy")):
                    s = np.load(f)
                    y = wl[:len(s)]
                    if len(np.unique(y)) < 2:
                        continue
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    meth = f.stem.split("__")[1]
                    rows.append(dict(pair=pair, indicator=ind, method=meth,
                                     seed=int(f.stem.split("s")[-1]),
                                     auroc=roc_auc_score(y, s),
                                     aupr=average_precision_score(y, s),
                                     anom_ratio=float(y.mean()),
                                     onsets=str(res["onsets"])))
                print(f"    {ind:9s} 이상비율={wl.mean():.3f} onsets={res['onsets']}")
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / "indicator_results.csv", index=False)
            print("\n=== 상태지표별 AUROC (method 순위 유지 여부) ===")
            print(df.pivot_table(index=["pair", "indicator"], columns="method",
                                 values="auroc").round(4).to_string())
        else:
            print("\n저장된 score가 없습니다. --score_dir 로 exp1/exp5의 scores 폴더를 지정하세요.")
        return

    # ── crossmachine / position ────────────────────────────────────────────
    if args.stage == "crossmachine":
        pairs = args.pairs or list(CROSS_MACHINE)
        methods = ["source_only", "adapter_latent", "full_ft"]
        srcs = CROSS_MACHINE
    else:
        import run_experiment1 as R
        pairs = args.pairs or ["XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1"]
        methods = ["source_only", "adapter_latent", "adapter_output",
                   "adapter_input", "full_ft"]
        srcs = {p: (R.DOMAIN_PAIRS[p]["src"], R.DOMAIN_PAIRS[p]["tgt"]) for p in pairs}

    print(f"[exp8-{args.stage}] pairs={pairs} methods={methods} seeds={args.seeds}")
    rows, f = [], out / f"{args.stage}_results.csv"
    if f.exists():
        rows = pd.read_csv(f).to_dict("records")
    for pair in pairs:
        s_csv, t_csv = srcs[pair]
        if not (Path(s_csv).exists() and Path(t_csv).exists()):
            print(f"  [skip] {pair}: CSV 없음"); continue
        D = bearing_data(s_csv, t_csv)
        print(f"\n{'━'*60}\n  {pair}  clean={len(D['clean'])}w eval={len(D['eval_x'])}w "
              f"이상={D['eval_y'].mean():.3f}")
        for seed in args.seeds:
            print(f"  ── seed {seed}")
            for meth in methods:
                try:
                    rows.append(run_one(pair, D, seed, meth, args, device, out))
                    pd.DataFrame(rows).to_csv(f, index=False)
                except Exception as e:
                    print(f"    [ERR {meth}] {type(e).__name__}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*70}\n=== {args.stage} 요약 ===")
        print(df.groupby(["pair", "method"])[["auroc", "aupr_lift", "f1_far0.01",
                                              "trainable_params"]].mean().round(4).to_string())
        base = df[df.method == "source_only"].groupby(["pair", "seed"])["auroc"].mean()
        g = df[df.method != "source_only"].copy()
        g["d"] = g.apply(lambda r: r["auroc"] - base.get((r["pair"], r["seed"]), np.nan), axis=1)
        print("\n=== source_only 대비 (pair 단위) ===")
        for m, gg in g.groupby("method"):
            v = gg.groupby("pair")["d"].mean().dropna()
            if len(v) >= 3:
                try:
                    p = sps.wilcoxon(v)[1]
                except Exception:
                    p = np.nan
                print(f"  {m:16s} mean={v.mean():+.4f} win={int((v>0).sum())}/{len(v)} p={p:.4f}")
    print(f"\n[exp8] → {out.resolve()}")


if __name__ == "__main__":
    main()

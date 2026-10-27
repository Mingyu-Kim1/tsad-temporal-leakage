# -*- coding: utf-8 -*-
"""
exp6_baselines.py — 발표된 방법들의 '오염 내성' 비교 (논문 핵심 그림)
======================================================================
핵심 질문: 기존 TSAD/적응 방법들은 target 학습 데이터가 오염되면 어떻게 되는가?
  → 저희 full FT는 오염 5%에서 이미 무적응보다 나빠졌음.
  → USAD·TranAD처럼 전체 학습하는 방법도 같은 붕괴를 보이는가?

비교 대상
  source_only   적응 없음 (하한)
  ours          TF-AE(frozen) + latent adapter(1%) + ISCA 자가정제  ← 제안
  full_ft       TF-AE 전체 미세조정
  usad_target   USAD를 target pool로만 학습        (USAD 표준 사용법)
  usad_ft       USAD source 사전학습 → target 미세조정 (동일 데이터 조건 비교)
  tranad_target TranAD를 target pool로만 학습
  tranad_ft     TranAD source 사전학습 → target 미세조정

※ AdaBN은 제외 — 본 백본이 LayerNorm 기반이라 BatchNorm 통계 치환이 정의되지 않음.
  동등한 latent 통계 정합(featnorm)은 exp3 align stage에서 이미 평가함.

■ 데이터 처리 정확성 (신경 쓴 부분)
  - 전 방법이 '완전히 동일한' 오염 pool과 동일한 평가 window를 사용
  - source-fit MinMax 후 USAD 입력만 [0,1] clip (USAD는 sigmoid 출력이라 필수)
  - TranAD는 시간축 우선 [W,B,F] 텐서 + 마지막 시점 예측이라 전용 어댑터 사용
  - 라벨 규칙이 방법마다 유불리가 갈리므로 'any'와 'last' 두 규칙 모두 계산·보고
    (window 전체 점수를 쓰는 방법 vs 마지막 시점 점수를 쓰는 방법의 공정성)

■ 사전 준비 (TranAD)
    git clone --depth 1 https://github.com/imperial-qore/TranAD.git reference_TranAD
  없으면 TranAD 관련 설정은 자동으로 건너뜁니다.

실행 (파일럿: Boiler 3쌍 × seed 1개 × 오염 3수준, 약 2~3시간)
    python exp6_baselines.py --quick

전체
    python exp6_baselines.py --seeds 0 1 2 --contam 0 0.05 0.1 0.2 0.4
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, copy, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, ".")
from config import model_cfg, train_cfg
from models import create_model
from run_cross_domain_adapter_v2 import full_finetune, get_latent_dim
from exp2_robust import sample_errors, scores as tf_scores
from exp3_final import (STRIDE, train_adapter, fresh_adapted, make_pool, set_all_seeds,
                        MODEL_NAME, ISCA_KEEP)
from exp5_final import prepare_any, pretrain as tf_pretrain, W_BENCH, W_FIELD, FIELD_PAIRS
import exp1_benchmark as B

FAR = 0.01
LB = 0.1


# ===========================================================================
# USAD 재구현 (Audibert et al., KDD 2020)
#   encoder 1개 + decoder 2개, 2단계 적대적 학습
#   L_AE1 = 1/n ||W-AE1(W)||² + (1-1/n)||W-AE2(AE1(W))||²
#   L_AE2 = 1/n ||W-AE2(W)||² - (1-1/n)||W-AE2(AE1(W))||²
#   score = α||W-AE1(W)||² + β||W-AE2(AE1(W))||²   (α=β=0.5)
# ===========================================================================
class USAD(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        h1, h2 = max(w_size // 2, 8), max(w_size // 4, 4)
        self.enc = nn.Sequential(nn.Linear(w_size, h1), nn.ReLU(True),
                                 nn.Linear(h1, h2), nn.ReLU(True),
                                 nn.Linear(h2, z_size), nn.ReLU(True))
        def dec():
            return nn.Sequential(nn.Linear(z_size, h2), nn.ReLU(True),
                                 nn.Linear(h2, h1), nn.ReLU(True),
                                 nn.Linear(h1, w_size), nn.Sigmoid())
        self.dec1, self.dec2 = dec(), dec()

    def forward(self, w):
        z = self.enc(w)
        return self.dec1(z), self.dec2(z), self.dec2(self.enc(self.dec1(z)))


def usad_fit_minmax(w_arr, eps=1e-6):
    """Fit a feature-wise Min-Max scaler using training windows only."""
    x = np.asarray(w_arr, dtype=np.float32)

    if x.ndim != 3 or len(x) == 0:
        raise ValueError(
            f"Expected non-empty [N, W, F] input, received shape={x.shape}"
        )

    x_min = x.min(axis=(0, 1), keepdims=True)
    x_max = x.max(axis=(0, 1), keepdims=True)
    x_range = np.maximum(x_max - x_min, eps)

    return x_min.astype(np.float32), x_range.astype(np.float32)


def usad_apply_minmax(w_arr, x_min, x_range):
    """Apply the training-fitted scaler and clip extrapolated values."""
    x = np.asarray(w_arr, dtype=np.float32)
    z = (x - x_min) / x_range
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def usad_flat(w_arr, sub_w=None):
    """Convert [N, W, F] windows into non-overlapping USAD sub-windows."""
    x = np.asarray(w_arr)

    if x.ndim != 3:
        raise ValueError(
            f"usad_flat expects [N, W, F], but received shape={x.shape}"
        )

    n_win, full_w, n_feat = x.shape

    if sub_w is None:
        sub_w = full_w

    sub_w = int(sub_w)

    if sub_w < 1 or sub_w > full_w:
        raise ValueError(
            f"sub_w must satisfy 1 <= sub_w <= full_w, "
            f"but received sub_w={sub_w}, full_w={full_w}"
        )

    # Match the aggregation protocol used by usad_agg.
    n_sub = max(full_w // sub_w, 1)

    x = x[:, :n_sub * sub_w, :]
    x = x.reshape(n_win * n_sub, sub_w, n_feat)

    return np.clip(
        x.reshape(len(x), -1),
        0.0,
        1.0
    ).astype(np.float32)


def usad_agg(sub_scores, n_win, sub_w, full_w):
    """Aggregate sub-window scores into one score per original window."""
    n_sub = max(int(full_w) // int(sub_w), 1)
    scores = np.asarray(sub_scores)

    expected = int(n_win) * n_sub
    if len(scores) != expected:
        raise ValueError(
            f"Expected {expected} sub-window scores, "
            f"but received {len(scores)}."
        )

    return scores.reshape(int(n_win), n_sub).max(axis=1)



def usad_train(model, tr, va, epochs, lr, device, bs=128, patience=8, init_epoch=1):
    o1 = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec1.parameters()), lr=lr)
    o2 = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec2.parameters()), lr=lr)
    best, state, wait = float("inf"), None, 0
    for ep in range(init_epoch, init_epoch + epochs):
        model.train()
        n = float(ep)
        for idx in np.array_split(np.random.permutation(len(tr)), max(len(tr) // bs, 1)):
            if len(idx) < 2:
                continue
            w = torch.from_numpy(tr[idx]).to(device)
            w1, w2, w3 = model(w)
            l1 = (1/n) * torch.mean((w - w1)**2) + (1 - 1/n) * torch.mean((w - w3)**2)
            o1.zero_grad(); l1.backward(); o1.step()
            w1, w2, w3 = model(w)
            l2 = (1/n) * torch.mean((w - w2)**2) - (1 - 1/n) * torch.mean((w - w3)**2)
            o2.zero_grad(); l2.backward(); o2.step()
        model.eval()
        with torch.no_grad():
            v = []
            for i in range(0, len(va), bs):
                w = torch.from_numpy(va[i:i+bs]).to(device)
                w1, _, w3 = model(w)
                v.append((0.5*torch.mean((w-w1)**2) + 0.5*torch.mean((w-w3)**2)).item())
            v = float(np.mean(v)) if v else float("inf")
        if v < best - 1e-7:
            best, wait, state = v, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if state:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def usad_score(model, arr, device, bs=256, alpha=0.5, beta=0.5):
    model.eval(); out = []
    for i in range(0, len(arr), bs):
        w = torch.from_numpy(arr[i:i+bs]).to(device)
        w1, _, w3 = model(w)
        s = alpha*torch.mean((w-w1)**2, dim=1) + beta*torch.mean((w-w3)**2, dim=1)
        out.append(s.cpu().numpy())
    return np.concatenate(out)


# ===========================================================================
# TranAD 어댑터 (공개 저장소 사용)
# ===========================================================================
def load_tranad(feats, device, repo="reference_TranAD"):
    p = Path(repo)
    if not p.exists():
        return None, f"{repo} 없음 (git clone https://github.com/imperial-qore/TranAD.git {repo})"
    sys.path.insert(0, str(p.resolve()))
    try:
        from src.models import TranAD          # noqa
    except Exception as e:
        return None, f"TranAD import 실패: {e}"
    try:
        m = TranAD(feats).double().to(device)
        return m, None
    except Exception as e:
        return None, f"TranAD 생성 실패: {e}"


def tranad_train(model, tr, va, epochs, lr, device, bs=128, patience=8):
    """tr: [N,W,F]. TranAD는 [W,B,F] 시간축 우선 + 마지막 시점 예측."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss(reduction="none")
    best, state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train(); n = float(ep)
        for idx in np.array_split(np.random.permutation(len(tr)), max(len(tr)//bs, 1)):
            if len(idx) < 2:
                continue
            w = torch.from_numpy(tr[idx]).double().to(device)      # [B,W,F]
            win = w.permute(1, 0, 2)                               # [W,B,F]
            elem = win[-1, :, :].view(1, len(idx), -1)             # [1,B,F]
            z = model(win, elem)
            if isinstance(z, tuple):
                l = (1/n) * mse(z[0], elem).mean() + (1 - 1/n) * mse(z[1], elem).mean()
            else:
                l = mse(z, elem).mean()
            opt.zero_grad(); l.backward(retain_graph=True); opt.step()
        model.eval()
        with torch.no_grad():
            v = []
            for i in range(0, len(va), bs):
                w = torch.from_numpy(va[i:i+bs]).double().to(device)
                win = w.permute(1, 0, 2); elem = win[-1].view(1, len(w), -1)
                z = model(win, elem)
                z = z[1] if isinstance(z, tuple) else z
                v.append(mse(z, elem).mean().item())
            v = float(np.mean(v)) if v else float("inf")
        if v < best - 1e-9:
            best, wait, state = v, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if state:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def tranad_score(model, arr, device, bs=256):
    model.eval(); mse = nn.MSELoss(reduction="none"); out = []
    for i in range(0, len(arr), bs):
        w = torch.from_numpy(arr[i:i+bs]).double().to(device)
        win = w.permute(1, 0, 2); elem = win[-1].view(1, len(w), -1)
        z = model(win, elem)
        z = z[1] if isinstance(z, tuple) else z
        out.append(mse(z, elem).mean(dim=2).squeeze(0).cpu().numpy())
    return np.concatenate(out)


# ===========================================================================
def eval_scores(s_te, s_cal, y_any, y_last):
    r = {}
    thr = float(np.quantile(s_cal, 1 - FAR))
    pred = (s_te >= thr).astype(int)
    for tag, y in (("any", y_any), ("last", y_last)):
        if len(np.unique(y)) > 1:
            r[f"auroc_{tag}"] = roc_auc_score(y, s_te)
            r[f"aupr_{tag}"] = average_precision_score(y, s_te)
            r[f"lift_{tag}"] = r[f"aupr_{tag}"] / max(float(y.mean()), 1e-9)
        else:
            r[f"auroc_{tag}"] = r[f"aupr_{tag}"] = r[f"lift_{tag}"] = np.nan
        r[f"f1_{tag}"] = f1_score(y, pred, zero_division=0)
    r["anom_ratio_any"] = float(y_any.mean())
    return r


def last_labels(pair, family, n_eval, W):
    """평가 window의 '마지막 시점' 라벨 (TranAD류 공정성용)."""
    if family == "bench":
        cfg = B.BENCH_PAIRS[pair]
        _, T = B.load_pair(cfg)
        pl = T["test_label"]
    else:
        import run_experiment1 as R, exp1_labeling as L
        cfg = R.DOMAIN_PAIRS[pair]
        pl = (L.pipe_labels(len(pd.read_csv(cfg["tgt"])), L.PIPE_FAULT_IDX_DEFAULT)["labels"]
              if cfg["kind"] == "pipe" else L.bearing_pseudo_labels(cfg["tgt"])["labels"])
    return np.asarray(pl[W - 1: W - 1 + n_eval], dtype=int)


# ===========================================================================
def run_pair(pair, family, seed, args, device, out, rows, f):
    D = prepare_any(pair, family, 3.0, 5)
    W, F = D["W"], D["n_feat"]
    y_any = D["eval_y"]
    y_last = last_labels(pair, family, len(D["eval_x"]), W)
    y_last = y_last[:len(y_any)] if len(y_last) >= len(y_any) else np.pad(
        y_last, (0, len(y_any) - len(y_last)))
    print(f"  eval={len(D['eval_x'])}w  이상비율 any={y_any.mean():.3f} last={y_last.mean():.3f}")

    # 공통 source 사전학습 (TF-AE)
    ck = out / "ckpt" / f"{pair}__s{seed}__tf.pt"
    model_cfg.n_heads = 5
    tf = tf_pretrain(D, seed, 5, LB, args, device, ck)

    # USAD / TranAD source 사전학습 (동일 source 데이터)
    src_f = usad_flat(D["src"]); ns = int(len(src_f)*0.8)
    set_all_seeds(seed)
    usad_src = USAD(W*F, args.z_size).to(device)
    usad_src = usad_train(usad_src, src_f[:ns], src_f[ns:], args.epochs_pretrain,
                          args.lr_base, device)
    tran_src, tmsg = (None, None)
    if not args.no_tranad:
        tran_src, tmsg = load_tranad(F, device, args.tranad_repo)
        if tran_src is None:
            print(f"  [TranAD 건너뜀] {tmsg}")
        else:
            set_all_seeds(seed)
            tran_src = tranad_train(tran_src, D["src"][:ns], D["src"][ns:],
                                    args.epochs_pretrain, args.lr_base, device)

    for c in args.contam:
        rng = np.random.default_rng(seed*991 + int(c*100))
        pool, pgt = make_pool(D, c, "real" if len(D["anom"]) else "synth", rng)
        cut = max(int(len(pool)*0.8), 1)
        tr_w, va_w = pool[:cut], pool[cut:] if len(pool)-cut >= 5 else pool[-5:]
        pf, vf = usad_flat(tr_w), usad_flat(va_w)
        base = dict(pair=pair, family=family, seed=seed, contam=c,
                    contam_actual=float(pgt.mean()))
        print(f"  ── 오염 {c:.0%} (실제 {pgt.mean():.3f}) pool={len(pool)}w")

        for name in args.methods:
            t0 = time.perf_counter()
            try:
                if name == "source_only":
                    s_te = tf_scores(tf, D["eval_x"], train_cfg.batch_size, device)
                    s_cal = tf_scores(tf, pool, train_cfg.batch_size, device)
                    npar = 0
                elif name == "ours":
                    ld = get_latent_dim(MODEL_NAME); cur, ad = pool, None
                    for keep in ISCA_KEEP:
                        if keep < 1.0 and ad is not None:
                            sc = tf_scores(ad, pool, train_cfg.batch_size, device)
                            cur = pool[np.argsort(sc)[:max(int(len(pool)*keep), 10)]]
                        k = max(int(len(cur)*0.8), 1)
                        ad = fresh_adapted(tf, "bottleneck", ld, device)
                        train_adapter(ad, cur[:k], cur[k:] if len(cur)-k >= 5 else cur[-5:],
                                      train_cfg.batch_size, args.epochs_adapt,
                                      args.adapt_lr, device)
                    s_te = tf_scores(ad, D["eval_x"], train_cfg.batch_size, device)
                    # ★ threshold는 ISCA가 정제한 pool로 산출 (라벨 미사용)
                    s_cal = tf_scores(ad, cur, train_cfg.batch_size, device)
                    npar = sum(p.numel() for p in ad.adapter.parameters())
                elif name == "full_ft":
                    ft = copy.deepcopy(tf)
                    full_finetune(ft, torch.utils.data.DataLoader(torch.FloatTensor(tr_w),
                                  batch_size=train_cfg.batch_size, shuffle=True),
                                  torch.utils.data.DataLoader(torch.FloatTensor(va_w),
                                  batch_size=train_cfg.batch_size, shuffle=False),
                                  args.epochs_adapt, args.ft_lr, device)
                    s_te = tf_scores(ft, D["eval_x"], train_cfg.batch_size, device)
                    s_cal = tf_scores(ft, pool, train_cfg.batch_size, device)
                    npar = sum(p.numel() for p in ft.parameters())
                elif name in ("usad_target", "usad_ft", "usad_isca"):
                    set_all_seeds(seed)
                    m = (USAD(W*F, args.z_size).to(device) if name == "usad_target"
                         else copy.deepcopy(usad_src))
                    if name != "usad_isca":
                        m = usad_train(m, pf, vf, args.epochs_adapt, args.lr_base, device)
                        cal = usad_flat(pool)
                    else:
                        # 공정 비교: USAD에도 동일한 반복 자가정제를 적용
                        cur_f = usad_flat(pool)
                        for keep in ISCA_KEEP:
                            if keep < 1.0:
                                sc = usad_score(m, usad_flat(pool), device)
                                cur_f = usad_flat(pool)[np.argsort(sc)[
                                    :max(int(len(pool)*keep), 10)]]
                            k = max(int(len(cur_f)*0.8), 1)
                            m = copy.deepcopy(usad_src)
                            m = usad_train(m, cur_f[:k], cur_f[k:] if len(cur_f)-k >= 5
                                           else cur_f[-5:], args.epochs_adapt,
                                           args.lr_base, device)
                        cal = cur_f
                    s_te = usad_score(m, usad_flat(D["eval_x"]), device)
                    s_cal = usad_score(m, cal, device)
                    npar = sum(p.numel() for p in m.parameters())
                elif name in ("tranad_target", "tranad_ft"):
                    if tran_src is None and name == "tranad_ft":
                        continue
                    set_all_seeds(seed)
                    if name == "tranad_target":
                        m, msg = load_tranad(F, device, args.tranad_repo)
                        if m is None:
                            continue
                    else:
                        m = copy.deepcopy(tran_src)
                    m = tranad_train(m, tr_w, va_w, args.epochs_adapt, args.lr_base, device)
                    s_te = tranad_score(m, D["eval_x"], device)
                    s_cal = tranad_score(m, pool, device)
                    npar = sum(p.numel() for p in m.parameters())
                else:
                    continue
                r = eval_scores(s_te, s_cal, y_any, y_last)
                rows.append({**base, "method": name, "trainable_params": npar,
                             "time_s": time.perf_counter()-t0, **r})
                pd.DataFrame(rows).to_csv(f, index=False)
                print(f"      [{name:14s}] AUROC(any)={r['auroc_any']:.4f} "
                      f"AUROC(last)={r['auroc_last']:.4f} F1={r['f1_any']:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)")
            except Exception as e:
                print(f"      [ERR {name}] {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=["boiler"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--contam", nargs="*", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--methods", nargs="*",
                    default=["source_only", "ours", "full_ft", "usad_ft",
                             "usad_target", "usad_isca", "tranad_ft", "tranad_target"])
    ap.add_argument("--z_size", type=int, default=64)
    ap.add_argument("--lr_base", type=float, default=1e-3)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--epochs_pretrain", type=int, default=40)
    ap.add_argument("--epochs_adapt", type=int, default=20)
    ap.add_argument("--tranad_repo", default="reference_TranAD")
    ap.add_argument("--no_tranad", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="파일럿: seed 1개·오염 3수준·epoch 축소 (2~3시간)")
    ap.add_argument("--out", default="exp6_out")
    ap.add_argument("--force_pretrain", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.contam = [0], [0.0, 0.2, 0.4]
        args.epochs_pretrain, args.epochs_adapt = 25, 12

    out = Path(args.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    device = train_cfg.device
    from exp3_final import PAIRS as BP
    if args.pairs:
        tasks = [(p, "field" if p in FIELD_PAIRS else "bench") for p in args.pairs]
    else:
        tasks = [(p, "bench") for d in args.datasets if d in BP for p in BP[d]]
    print(f"[exp6] pairs={[t[0] for t in tasks]} methods={args.methods} "
          f"contam={args.contam} seeds={args.seeds}")

    f = out / "results.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    for pair, fam in tasks:
        print(f"\n{'━'*64}\n  {pair} [{fam}]")
        for seed in args.seeds:
            run_pair(pair, fam, seed, args, device, out, rows, f)

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*72}\n=== 오염 수준별 AUROC(any) ===")
        print(df.pivot_table(index="contam", columns="method",
                             values="auroc_any").round(4).to_string())
        print("\n=== 오염 0→최대 하락폭 (붕괴 정도) ===")
        p = df.pivot_table(index="method", columns="contam", values="auroc_any")
        if p.shape[1] >= 2:
            p["drop"] = p[p.columns[0]] - p[p.columns[-1]]
            print(p.round(4).to_string())
    print(f"\n[exp6] → {f.resolve()}")


if __name__ == "__main__":
    main()

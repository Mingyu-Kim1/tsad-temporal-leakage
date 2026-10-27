# -*- coding: utf-8 -*-
"""
exp4_final.py — 리뷰어 공격 대응 실험 (+ MSL/SMAP 추가)
=========================================================
exp3 결과에 대해 예상되는 리뷰어 공격을 실험으로 방어한다.

  [Stage alpha]  정렬 α 곡선 — "α=0.001은 너무 작아서 MMD가 안 먹힌 것 아닌가"
                 α ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1} 전 구간에서 α=0을 이기는
                 지점이 없음을 보여야 '정렬 반증'이 성립.
                 + full FT에도 정렬 적용 (frozen decoder 때문인지 분리)

  [Stage capdim] 순수 용량 축 — shift/diag/bottleneck/linear는 '함수 형태'가
                 달라 용량만의 효과가 아님. 동일 구조에서 bottleneck 차원만
                 {2,8,32,128} 변화시켜 용량 효과를 분리.

  [Stage mode]   Multimode 해석 — 각 head가 어떤 운전 모드를 담당하는가.
                 (a) head 사용 분포 및 source-target 모드 분포 shift(JS)
                 (b) Boiler 운전상태 컬럼과의 상호정보량 → 물리적 검증
                 (c) head별 센서 프로파일
                 (d) 이상의 head 편중
                 (e) ★ head별 FAR threshold vs 전역 threshold (F1=0 문제 해결)
                 → 학습 불필요(캐시된 모델 사용)이라 매우 빠름

  [Stage guard]  ISCA 안전장치 — Boiler_3to1처럼 탐지기가 역전(AUROC<0.5)이면
                 자가정제가 오염을 오히려 증가시킴(5%→5.8%, 40%→48.4% 실측).
                 라벨 없이 이 상황을 감지하는 진단 지표를 계산·검증.

실행:
  python exp4_final.py --stage mode  --seeds 0 1 2          # 가장 빠름(수분)
  python exp4_final.py --stage alpha --seeds 0 1 2
  python exp4_final.py --stage capdim --seeds 0 1 2
  python exp4_final.py --stage guard --seeds 0 1 2
  python exp4_final.py --stage mode --datasets msl          # MSL/SMAP 추가 후
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, ast, copy, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sps
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score, f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from config import model_cfg, train_cfg
from models import create_model, select_best_head
from trainer import train_model
from run_cross_domain_adapter_v2 import LatentAdapter, AdaptedModel, full_finetune, get_latent_dim
import exp1_benchmark as B
from exp2_robust import sample_errors, trimmed_mean, scores, evaluate_all, latents
from exp3_final import (WINDOW, STRIDE, ALPHA, CAPACITY, ShiftAdapter, DiagAffineAdapter,
                        LinearAdapter, train_adapter, fresh_adapted, prepare,
                        source_latents_modes, make_pool, set_all_seeds, MODEL_NAME, PAIRS)

# ===========================================================================
# MSL / SMAP 로더 (train=정상만 → 오염 주입은 합성으로만 가능)
# ===========================================================================
MSL_DIRS = ["data/benchmark/MSL_SMAP/data/data", "data/benchmark/MSL_SMAP/data",
            "data/benchmark/MSL_SMAP", "data/benchmark/nasa", "data/benchmark",
            "data/benchmark/data"]


def find_msl_root():
    """train/ test/ 가 있는 폴더를 우선 반환. labeled_anomalies.csv는 상위에 있어도 됨."""
    for c in MSL_DIRS:
        p = Path(c)
        if (p / "train").is_dir() and (p / "test").is_dir():
            return p
    for p in Path("data").rglob("labeled_anomalies.csv"):
        for c in [p.parent, p.parent / "data", p.parent / "data" / "data"]:
            if (c / "train").is_dir() and (c / "test").is_dir():
                return c
    return None


def find_msl_csv(root):
    """labeled_anomalies.csv 위치 탐색 (root 또는 상위 폴더)."""
    for c in [root, root.parent, root.parent.parent,
              Path("data/benchmark/MSL_SMAP")]:
        f = c / "labeled_anomalies.csv"
        if f.exists():
            return f
    hits = list(Path("data").rglob("labeled_anomalies.csv"))
    return hits[0] if hits else None


def load_msl_entity(entity: str):
    root = find_msl_root()
    if root is None:
        raise FileNotFoundError(
            "labeled_anomalies.csv를 찾지 못했습니다. MSL/SMAP을 "
            "data/benchmark/MSL_SMAP/ 아래에 train/, test/, labeled_anomalies.csv "
            "구조로 배치하세요.")
    csvp = find_msl_csv(root)
    if csvp is None:
        raise FileNotFoundError("labeled_anomalies.csv 없음")
    la = pd.read_csv(csvp)
    row = la[la["chan_id"] == entity]
    if len(row) == 0:
        raise KeyError(f"{entity} 없음. 가용: {la['chan_id'].tolist()[:15]} ...")
    row = row.iloc[0]
    tr = np.load(root / "train" / f"{entity}.npy").astype(np.float32)
    te = np.load(root / "test" / f"{entity}.npy").astype(np.float32)
    lab = np.zeros(len(te), int)
    for s, e in ast.literal_eval(row["anomaly_sequences"]):
        lab[int(s):int(e) + 1] = 1
    return dict(train=np.nan_to_num(tr), train_label=np.zeros(len(tr), int),
                test=np.nan_to_num(te), test_label=lab)


def register_msl_pairs(pairs_spec):
    """pairs_spec: [(src,tgt), ...] — 같은 spacecraft 내 entity 간 전이"""
    added = []
    for s, t in pairs_spec:
        name = f"MSL_{s}to{t}"
        B.BENCH_PAIRS[name] = dict(kind="msl", src=s, tgt=t)
        added.append(name)
    return added


_orig_load_pair = B.load_pair


def _load_pair_ext(cfg):
    if cfg.get("kind") == "msl":
        return load_msl_entity(cfg["src"]), load_msl_entity(cfg["tgt"])
    return _orig_load_pair(cfg)


B.load_pair = _load_pair_ext          # exp3.prepare()가 이걸 사용

# ===========================================================================
# 공통: 캐시된 사전학습 모델
# ===========================================================================

def get_model(pair, seed, D, args, device, out):
    set_all_seeds(seed)
    model_cfg.n_heads = 5; model_cfg.input_size = D["n_feat"]
    model_cfg.window_size = WINDOW; model_cfg.d_model = 128; model_cfg.dim_ff = 512
    for ck in [Path(args.ckpt_dir) / f"{pair}__s{seed}__w{WINDOW}.pt",
               out / "ckpt" / f"{pair}__s{seed}__w{WINDOW}.pt"]:
        if ck.exists() and not args.force_pretrain:
            m = create_model(MODEL_NAME, model_cfg)
            m.load_state_dict(torch.load(ck, map_location=device))
            return m.to(device)
    n = int(len(D["src"]) * 0.8)
    train_cfg.seed = seed; train_cfg.epochs = args.epochs_pretrain
    m, _, _ = train_model(MODEL_NAME, model_cfg, train_cfg,
                          torch.utils.data.DataLoader(torch.FloatTensor(D["src"][:n]),
                              batch_size=train_cfg.batch_size, shuffle=True),
                          torch.utils.data.DataLoader(torch.FloatTensor(D["src"][n:]),
                              batch_size=train_cfg.batch_size, shuffle=False), verbose=False)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out / "ckpt" / f"{pair}__s{seed}__w{WINDOW}.pt")
    return m.to(device)


@torch.no_grad()
def head_assign(model, w_arr, bs, device):
    """window별 지배 head와 best-head score 반환."""
    model.eval(); H, S = [], []
    for i in range(0, len(w_arr), bs):
        x = torch.FloatTensor(w_arr[i:i + bs]).to(device)
        per, mode = sample_errors(model(x), x)
        H.append(mode.cpu().numpy()); S.append(per.cpu().numpy())
    return np.concatenate(H), np.concatenate(S)


# ===========================================================================
# Stage: mode — multimode 해석 + head별 threshold
# ===========================================================================
BOILER_STATE_COLS = ["Operating Status", "Operating Code", "Input Status"]


def boiler_state_per_window(pair, n_win, window, stride=1):
    """test 구간의 운전상태 컬럼을 window 단위(최빈값)로 변환."""
    cfg = B.BENCH_PAIRS[pair]
    if cfg["kind"] != "boiler":
        return None
    df = pd.read_csv(B.BOILER_DIR / cfg["tgt"] / "test.csv")
    out = {}
    for c in BOILER_STATE_COLS:
        if c not in df.columns:
            continue
        v = df[c].to_numpy()
        idx = np.arange(window)[None, :] + stride * np.arange(n_win)[:, None]
        seg = v[idx]
        out[c] = sps.mode(seg, axis=1, keepdims=False).mode
    return out


def run_mode(pair, seed, D, model, args, device, out):
    bs = train_cfg.batch_size
    h_src, _ = head_assign(model, D["src"], bs, device)
    h_tgt, _ = head_assign(model, D["clean"], bs, device)
    h_te, s_te = head_assign(model, D["eval_x"], bs, device)
    y = D["eval_y"]
    N = model_cfg.n_heads

    p_s = np.bincount(h_src, minlength=N) / max(len(h_src), 1)
    p_t = np.bincount(h_tgt, minlength=N) / max(len(h_tgt), 1)
    js = float(jensenshannon(p_s + 1e-12, p_t + 1e-12, base=2) ** 2)

    rec = {"pair": pair, "seed": seed, "js_mode_shift": js,
           "n_heads_used_src": int((p_s > 0.01).sum()),
           "n_heads_used_tgt": int((p_t > 0.01).sum())}
    for k in range(N):
        rec[f"src_head{k}"] = float(p_s[k]); rec[f"tgt_head{k}"] = float(p_t[k])
        m = h_te == k
        rec[f"test_head{k}_frac"] = float(m.mean())
        rec[f"test_head{k}_anomrate"] = float(y[m].mean()) if m.any() else np.nan
        rec[f"test_head{k}_meanscore"] = float(s_te[m].mean()) if m.any() else np.nan

    # 물리적 검증 (Boiler): head ↔ 운전상태 상호정보량
    st = boiler_state_per_window(pair, len(h_te), WINDOW, 1)
    if st:
        for c, v in st.items():
            v = v[:len(h_te)]
            mi = mutual_info_score(v, h_te[:len(v)])
            hx = sps.entropy(np.bincount(h_te[:len(v)]) + 1e-12, base=np.e)
            rec[f"MI_head_{c}"] = float(mi)
            rec[f"NMI_head_{c}"] = float(mi / max(hx, 1e-12))

    # ★ head별 FAR threshold vs 전역 threshold
    clean_h, clean_s = head_assign(model, D["clean"], bs, device)
    for far in (0.05, 0.01):
        thr_g = float(np.quantile(clean_s, 1 - far))
        pred_g = (s_te >= thr_g).astype(int)
        pred_m = np.zeros_like(pred_g)
        for k in range(N):
            sk = clean_s[clean_h == k]
            t = float(np.quantile(sk, 1 - far)) if len(sk) >= 30 else thr_g
            pred_m[h_te == k] = (s_te[h_te == k] >= t).astype(int)
        for tag, pr in [("global", pred_g), ("permode", pred_m)]:
            rec[f"f1_{tag}_far{far:g}"] = f1_score(y, pr, zero_division=0)
            rec[f"prec_{tag}_far{far:g}"] = precision_score(y, pr, zero_division=0)
            rec[f"rec_{tag}_far{far:g}"] = recall_score(y, pr, zero_division=0)
    print(f"    JS(mode shift)={js:.4f} heads(src/tgt)={rec['n_heads_used_src']}/"
          f"{rec['n_heads_used_tgt']} F1@1% global={rec['f1_global_far0.01']:.4f} "
          f"per-mode={rec['f1_permode_far0.01']:.4f}"
          + (f" NMI(OpCode)={rec.get('NMI_head_Operating Code', float('nan')):.3f}" if st else ""))
    return rec


# ===========================================================================
# Stage: alpha / capdim / guard
# ===========================================================================

def run_alpha(pair, seed, D, model, args, device, out):
    bs = train_cfg.batch_size
    ld = get_latent_dim(MODEL_NAME)
    src_lat, src_mode = source_latents_modes(model, D["src"][:4000], bs, device)
    pool = D["clean"]; cut = max(int(len(pool) * .8), 1)
    tr, va = pool[:cut], pool[cut:]
    rows = []
    for a in args.alphas:
        for al in ["mmd", "mmd_mode"]:
            if a == 0 and al != "mmd":
                continue
            set_all_seeds(seed)
            ad = fresh_adapted(model, "bottleneck", ld, device)
            train_adapter(ad, tr, va, bs, args.epochs_adapt, args.adapt_lr, device,
                          trim_q=0.0, align=("none" if a == 0 else al), alpha=a,
                          src_lat=src_lat, src_mode=src_mode)
            m = evaluate_all(scores(ad, D["eval_x"], bs, device), D["eval_y"],
                             scores(ad, va, bs, device))
            rows.append({"pair": pair, "seed": seed, "config": f"a{a:g}_{al}",
                         "alpha": a, "align": al if a > 0 else "none", "arch": "adapter",
                         **m})
            print(f"    [adapter a={a:g} {al:9s}] AUROC={m['auroc']:.4f}")
    # full FT + 정렬 (frozen decoder 때문인지 분리)
    for a in [0.0, args.alphas[-1]]:
        set_all_seeds(seed)
        ft = copy.deepcopy(model)
        full_finetune(ft, torch.utils.data.DataLoader(torch.FloatTensor(tr), batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(va), batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
        m = evaluate_all(scores(ft, D["eval_x"], bs, device), D["eval_y"],
                         scores(ft, va, bs, device))
        rows.append({"pair": pair, "seed": seed, "config": f"fullft_a{a:g}",
                     "alpha": a, "align": "none", "arch": "full_ft", **m})
    return rows


def run_capdim(pair, seed, D, model, args, device, out):
    bs = train_cfg.batch_size; ld = get_latent_dim(MODEL_NAME)
    pool = D["clean"]; cut = max(int(len(pool) * .8), 1)
    tr, va = pool[:cut], pool[cut:]
    rows = []
    for dim in args.dims:
        set_all_seeds(seed)
        base = copy.deepcopy(model)
        for p in base.parameters():
            p.requires_grad = False
        ad = AdaptedModel(base, LatentAdapter(ld, bottleneck=dim).to(device),
                          MODEL_NAME).to(device)
        train_adapter(ad, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
        tra = sum(p.numel() for p in ad.adapter.parameters() if p.requires_grad)
        m = evaluate_all(scores(ad, D["eval_x"], bs, device), D["eval_y"],
                         scores(ad, va, bs, device))
        rows.append({"pair": pair, "seed": seed, "config": f"dim{dim}",
                     "bottleneck_dim": dim, "trainable_params": tra, **m})
        print(f"    [bottleneck dim={dim:3d} ({tra}p)] AUROC={m['auroc']:.4f}")
    return rows


def run_guard(pair, seed, D, model, args, device, out):
    """ISCA 안전장치: 라벨 없이 '정제가 역효과일 위험'을 감지할 수 있는가."""
    bs = train_cfg.batch_size; ld = get_latent_dim(MODEL_NAME)
    rows = []
    for c in args.contam:
        rng = np.random.default_rng(seed * 977 + int(c * 100))
        pool, pgt = make_pool(D, c, "real" if len(D["anom"]) else "synth", rng)
        cut = max(int(len(pool) * .8), 1)
        set_all_seeds(seed)
        ad = fresh_adapted(model, "bottleneck", ld, device)
        train_adapter(ad, pool[:cut], pool[cut:], bs, args.epochs_adapt,
                      args.adapt_lr, device)
        s = scores(ad, pool, bs, device)
        keep = np.argsort(s)[:int(len(pool) * .5)]
        # 라벨 없는 진단 지표들
        bimod = float(sps.skew(s))                       # 점수 분포 비대칭도
        kurt = float(sps.kurtosis(s))
        # 유지/제거 집합의 latent 이동 차이 (정제가 의미 있으면 커야 함)
        lat = latents(ad, pool, bs, device, apply_adapter=True)
        drop = np.setdiff1d(np.arange(len(pool)), keep)
        sep = float(np.linalg.norm(lat[keep].mean(0) - lat[drop].mean(0)))
        rows.append({"pair": pair, "seed": seed, "config": f"guard_c{c:.2f}",
                     "contam_target": c, "contam_kept": float(pgt[keep].mean()),
                     "contam_pool": float(pgt.mean()),
                     "cleaning_works": bool(pgt[keep].mean() < pgt.mean()),
                     "score_skew": bimod, "score_kurtosis": kurt,
                     "latent_sep_keep_drop": sep,
                     "auroc_adapted": evaluate_all(
                         scores(ad, D["eval_x"], bs, device), D["eval_y"],
                         scores(ad, pool[cut:], bs, device))["auroc"]})
        print(f"    [guard c={c:.2f}] pool오염={pgt.mean():.3f}→유지집합 "
              f"{pgt[keep].mean():.3f} ({'정상' if pgt[keep].mean()<pgt.mean() else '★역효과'})"
              f" skew={bimod:.2f} sep={sep:.2f}")
    return rows




# ===========================================================================
# Stage: balance — head collapse 진단 및 완화 (load-balancing 정규화)
# ===========================================================================
def _usage_entropy_loss(heads, x, n_head):
    """배치 내 head 사용률을 균등에 가깝게: CV^2 페널티 (MoE load-balancing)."""
    err = (heads - x.unsqueeze(0)).abs().mean(-1)          # (N,B,W)
    soft = torch.softmax(-err / (err.std() + 1e-8), dim=0)  # (N,B,W) 미분가능 배정
    usage = soft.mean(dim=(1, 2))                           # (N,)
    return (usage.std() / (usage.mean() + 1e-8)) ** 2       # CV^2


def run_balance(pair, seed, D, model, args, device, out):
    """
    source 재학습 시 head 균형 정규화를 적용해 collapse가 풀리는지 검증.
    적응이 아니라 '사전학습'에 개입해야 하므로 여기서 source 모델을 다시 학습한다.
    """
    bs = train_cfg.batch_size
    rows = []
    n = int(len(D["src"]) * 0.8)
    tr, va = D["src"][:n], D["src"][n:]
    for lb in args.lb:
        set_all_seeds(seed)
        model_cfg.n_heads = 5; model_cfg.input_size = D["n_feat"]
        model_cfg.window_size = WINDOW; model_cfg.d_model = 128; model_cfg.dim_ff = 512
        m = create_model(MODEL_NAME, model_cfg).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=train_cfg.lr)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(args.epochs_pretrain):
            m.train()
            perm = np.random.permutation(len(tr))
            for i in range(0, len(tr), bs):
                x = torch.FloatTensor(tr[perm[i:i + bs]]).to(device)
                if len(x) < 2:
                    continue
                h = m(x)
                per, _ = sample_errors(h, x)
                loss = per.mean()
                if lb > 0:
                    loss = loss + lb * _usage_entropy_loss(h, x, model_cfg.n_heads)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval()
            with torch.no_grad():
                v = []
                for i in range(0, len(va), bs):
                    x = torch.FloatTensor(va[i:i + bs]).to(device)
                    p, _ = sample_errors(m(x), x); v.append(p.mean().item())
                v = float(np.mean(v)) if v else float("inf")
            if v < best - 1e-6:
                best, wait, best_state = v, 0, copy.deepcopy(m.state_dict())
            else:
                wait += 1
                if wait >= 10:
                    break
        if best_state:
            m.load_state_dict(best_state)

        h_s, _ = head_assign(m, D["src"], bs, device)
        h_t, _ = head_assign(m, D["clean"], bs, device)
        h_e, s_e = head_assign(m, D["eval_x"], bs, device)
        N = model_cfg.n_heads
        p_s = np.bincount(h_s, minlength=N) / max(len(h_s), 1)
        ent = float(sps.entropy(p_s + 1e-12, base=2))
        # 균형화된 모델로 적응 후 성능
        ld = get_latent_dim(MODEL_NAME)
        pool = D["clean"]; cut = max(int(len(pool) * .8), 1)
        set_all_seeds(seed)
        ad = fresh_adapted(m, "bottleneck", ld, device)
        train_adapter(ad, pool[:cut], pool[cut:], bs, args.epochs_adapt,
                      args.adapt_lr, device)
        va_s = scores(ad, pool[cut:], bs, device)
        te_s = scores(ad, D["eval_x"], bs, device)
        met = evaluate_all(te_s, D["eval_y"], va_s)
        # head별 threshold (균형화 후에는 의미가 생기는지)
        ch, cs = head_assign(ad, pool, bs, device)
        eh, es = head_assign(ad, D["eval_x"], bs, device)
        far = 0.01
        thr_g = float(np.quantile(cs, 1 - far))
        pm = np.zeros(len(es), int)
        for k in range(N):
            sk = cs[ch == k]
            t = float(np.quantile(sk, 1 - far)) if len(sk) >= 30 else thr_g
            pm[eh == k] = (es[eh == k] >= t).astype(int)
        rows.append({"pair": pair, "seed": seed, "config": f"lb{lb:g}", "lb": lb,
                     "n_heads_used": int((p_s > 0.01).sum()),
                     "head_entropy_bits": ent, "src_val_loss": best,
                     "f1_global": met["f1_far0.01"],
                     "f1_permode": f1_score(D["eval_y"], pm, zero_division=0),
                     **{k: v for k, v in met.items()
                        if k in ("auroc", "aupr", "aupr_lift")}})
        print(f"    [lb={lb:g}] heads={int((p_s>0.01).sum())} entropy={ent:.3f}bit "
              f"AUROC={met['auroc']:.4f} F1(global)={met['f1_far0.01']:.4f} "
              f"F1(per-mode)={rows[-1]['f1_permode']:.4f}")
    return rows


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mode", "alpha", "capdim", "guard", "balance"],
                    default="mode")
    ap.add_argument("--datasets", nargs="*", default=["boiler", "smd"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--msl_pairs", nargs="*", default=[],
                    help="예: --msl_pairs M-6:M-1 T-4:T-5  (src:tgt)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--dims", nargs="*", type=int, default=[2, 8, 32, 128])
    ap.add_argument("--contam", nargs="*", type=float, default=[0.05, 0.20, 0.40])
    ap.add_argument("--lb", nargs="*", type=float, default=[0.0, 0.01, 0.1, 1.0],
                    help="head 균형화 계수 (0=현행). balance stage 전용")
    ap.add_argument("--epochs_pretrain", type=int, default=60)
    ap.add_argument("--epochs_adapt", type=int, default=30)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_dir", default="exp3_out/ckpt", help="exp3 체크포인트 재사용")
    ap.add_argument("--out", default="exp4_out")
    ap.add_argument("--force_pretrain", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); (out / "ckpt").mkdir(parents=True, exist_ok=True)
    device = train_cfg.device
    msl_added = []
    if args.msl_pairs:
        msl_added = register_msl_pairs([tuple(p.split(":")) for p in args.msl_pairs])
    pairs = args.pairs or ([p for d in args.datasets if d in PAIRS for p in PAIRS[d]] + msl_added)
    print(f"[exp4] stage={args.stage} pairs={pairs} seeds={args.seeds}")

    fn = {"mode": run_mode, "alpha": run_alpha, "capdim": run_capdim,
          "guard": run_guard, "balance": run_balance}[args.stage]
    rows = []
    f = out / f"{args.stage}_results.csv"
    for pair in pairs:
        try:
            D = prepare(pair, args, np.random.default_rng(1234))
        except Exception as e:
            print(f"  [skip] {pair}: {e}"); continue
        print(f"\n{'━'*60}\n  {pair} (clean={len(D['clean'])}w eval={len(D['eval_x'])}w)")
        for seed in args.seeds:
            model = get_model(pair, seed, D, args, device, out)
            print(f"  ── seed {seed}")
            r = fn(pair, seed, D, model, args, device, out)
            rows += (r if isinstance(r, list) else [r])
            pd.DataFrame(rows).to_csv(f, index=False)
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*70}\n[{args.stage}] 요약")
        if args.stage == "mode":
            cols = [c for c in ["js_mode_shift", "n_heads_used_src", "n_heads_used_tgt",
                                "f1_global_far0.01", "f1_permode_far0.01",
                                "NMI_head_Operating Code"] if c in df]
            print(df.groupby("pair")[cols].mean().round(4).to_string())
            if "f1_permode_far0.01" in df:
                d = df["f1_permode_far0.01"] - df["f1_global_far0.01"]
                try:
                    p = sps.wilcoxon(d)[1]
                except Exception:
                    p = np.nan
                print(f"\n  per-mode threshold 이득: mean={d.mean():+.4f} "
                      f"win={int((d>0).sum())}/{len(d)} p={p:.4f}")
        else:
            print(df.groupby(["pair", "config"])["auroc"].mean().round(4).to_string())
    print(f"\n[exp4] → {f.resolve()}")


if __name__ == "__main__":
    main()

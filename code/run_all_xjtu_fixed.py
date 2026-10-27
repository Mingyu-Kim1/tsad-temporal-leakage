# -*- coding: utf-8 -*-
"""
run_all_xjtu_fixed.py — XJTU 시간순 정렬 수정본 전용 재실행 러너
================================================================
■ 설계 원칙 (무인 실행 안정성)
  1) 재개 가능      : 셀 단위 고유키를 CSV에 기록. 재실행 시 이미 끝난 셀은 건너뜀
  2) 오류 격리      : 셀 하나가 실패해도 전체 중단 없음. 실패 내역은 errors.log에 기록
  3) 즉시 저장      : 매 셀마다 CSV 갱신 (정전·강제종료 시에도 손실 최소)
  4) 진행 로그      : run_log.txt에 타임스탬프와 진행률 기록
  5) 데이터 부재 대응: MSL/SMAP 등이 없으면 자동 skip (전체 중단 안 함)
  6) 단계별 시간 상한: --budget_h 로 stage별 최대 시간 지정 가능

■ 포함 실험 (교수님 2026-07-28 피드백 반영)
  S1 main        20 Pair × 6 method × 3 seed — 주 결과표
  S2 robustness  오염 0/5/10/20/40% × 4 method × 3 seed — ★Robustness curve
                 (오염 정의: 적응 window 중 anomaly window의 비율)
  S3 domaingap   Wasserstein(sliced)·MMD·CORAL·Energy distance 측정 + 개선폭 상관
                 ★"Domain gap이 클수록 개선폭이 큰가"의 정량 검증 (cross-machine 포함)
  S4 alpha       ★MMD strength curve — α ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}
                 "MMD를 뺀 것이 단순히 성능 때문인가"에 대한 근거
  S5 baselines   USAD(재구현) · TranAD(공개 저장소)
  S6 indicator   Pseudo Label 지표 변형(RMS/Kurtosis/Peak/Std/복합) — 재학습 불필요
  S7 selfrefine  ★Self-refinement를 라벨 없이 언제 적용할지 판단 가능한가
  S8 figures     Robustness curve, Domain gap 산점도, Pareto 등 그림 생성
  S9 analyze     전체 통계 (pair 단위 Wilcoxon + Holm), 결과표 CSV

■ 실행
    python run_all.py                     # 전체 (권장, 5일 무인)
    python run_all.py --stages S1 S2      # 일부만
    python run_all.py --stages S9          # 분석·그림만 재생성
    python run_all.py --smoke             # 5분 점검 (epoch 축소, pair 2개)
"""
from __future__ import annotations
import os, sys, time, json, traceback, argparse, copy, hashlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from config import model_cfg, train_cfg
import paths  # noqa: E402  seed 규약 (SEEDS_MAIN / SEEDS_HEAVY)
from models import create_model, select_best_head
from run_cross_domain_adapter_v2 import (LatentAdapter, AdaptedModel, full_finetune,
                                         get_latent_dim)
from exp2_robust import sample_errors, scores as tf_scores, evaluate_all
from exp1_efficiency import AdaptTimer, measure_inference_latency
from exp3_final import train_adapter, fresh_adapted, make_pool, set_all_seeds, MODEL_NAME, ISCA_KEEP
from exp4_final import _usage_entropy_loss, head_assign, register_msl_pairs, find_msl_root
from exp8_journal import OutputAdapted, InputAdapted, pseudo_label_by_indicator
from exp9_unified import prepare, pretrain, FIELD_PAIRS, BENCH_DEFAULT
import exp1_benchmark as B
import exp1_labeling as L

# ── 사전 고정 하이퍼파라미터 (튜닝 금지) ───────────────────────────────────
N_HEADS, LB, FAR = 5, 0.1, 0.01
CONTAM_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.40]
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MAIN_METHODS = ["source_only", "adapter_latent", "adapter_output",
                "adapter_input", "adapter_isca", "full_ft"]
ROBUST_METHODS = ["source_only", "adapter_latent", "adapter_isca", "full_ft"]

DEFAULT_OUT = Path("run_all_out_xjtu_fixed")
OUT = DEFAULT_OUT
RES = OUT / "results.csv"
GAP = OUT / "domaingap.csv"
LOG = OUT / "run_log.txt"
ERR = OUT / "errors.log"

XJTU_FIXED_PAIRS = [
    "XB1toXB2", "XB2toXB1",
    "XB1toPB1", "XB2toPB2",
    "PB1toXB1", "PB2toXB2",
]

EXPECTED_XJTU_SHA256 = {
    Path("data/xjtu_sy/XB1_combined.csv"): "a552fe89d6bedb39f32d1803bf8f3cce060c72e6154f59879d4b8916a113adb9",
    Path("data/xjtu_sy/XB2_combined.csv"): "c5467891c05714e41998a13e94e3622844622e3761e67c1c2fc1b2c2a3956bb5",
}

def configure_output(outdir):
    """출력 폴더를 변경해 기존 결과·체크포인트와 완전히 분리한다."""
    global OUT, RES, GAP, LOG, ERR
    OUT = Path(outdir)
    RES = OUT / "results.csv"
    GAP = OUT / "domaingap.csv"
    LOG = OUT / "run_log.txt"
    ERR = OUT / "errors.log"

def _sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_xjtu_fixed_files(allow_hash_mismatch=False):
    """정렬 수정본 CSV가 실제 데이터 경로에 배치됐는지 확인한다."""
    problems = []
    for path, expected in EXPECTED_XJTU_SHA256.items():
        if not path.exists():
            problems.append(f"파일 없음: {path}")
            continue
        actual = _sha256(path)
        if actual != expected:
            msg = f"해시 불일치: {path}\n  expected={expected}\n  actual  ={actual}"
            if allow_hash_mismatch:
                print(f"[warn] {msg}", flush=True)
            else:
                problems.append(msg)
    if problems:
        detail = "\n".join(problems)
        raise RuntimeError(
            "XJTU 정렬 수정본 확인에 실패했습니다.\n"
            + detail
            + "\nZIP의 수정된 XB1_combined.csv와 XB2_combined.csv를 "
              "data/xjtu_sy 폴더에 복사한 뒤 다시 실행하세요. "
              "파일을 Excel 등으로 다시 저장해 내용은 맞지만 해시만 달라졌다면 "
              "--allow_hash_mismatch 옵션을 사용하세요."
        )

def holm_adjust(pvalues):
    """정렬된 p-value에 대한 올바른 Holm–Bonferroni 보정."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


# ===========================================================================
# 로깅 · 재개
# ===========================================================================
def log(msg, p=LOG):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def logerr(key, e):
    log(f"  [ERROR] {key}: {type(e).__name__}: {e}")
    try:
        with open(ERR, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now()} | {key}\n{traceback.format_exc()}")
    except Exception:
        pass


def load_done(path, keycols):
    """이미 완료된 셀의 키 집합 (재개용)."""
    if not Path(path).exists():
        return set(), []
    try:
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        keys = set()
        for r in rows:
            try:
                keys.add(tuple(str(r.get(c, "")) for c in keycols))
            except Exception:
                pass
        return keys, rows
    except Exception:
        return set(), []


def save(rows, path):
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
    except Exception as e:
        logerr("save", e)


CROSS_MACHINE_PAIRS = {"XB1toPB1", "XB2toPB2", "PB1toXB1", "PB2toXB2"}
CROSS_COND_PAIRS = {"XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1"}


def transfer_level(pair):
    """교수님 요청: 전이 수준별 domain gap 비교 (특히 cross-machine)."""
    if pair in CROSS_MACHINE_PAIRS:
        return "cross-machine"
    if pair in CROSS_COND_PAIRS:
        return "cross-condition"
    if pair == "Sim2Real":
        return "sim-to-real"
    if pair.startswith("SMD"):
        return "benchmark-SMD"
    if pair.startswith("Boiler"):
        return "benchmark-Boiler"
    if pair.startswith("MSL"):
        return "benchmark-MSL/SMAP"
    return "other"


# ===========================================================================
# 데이터 준비 (캐시)
# ===========================================================================
_DCACHE = {}


def get_data(pair, args):
    if pair not in _DCACHE:
        _DCACHE[pair] = prepare(pair, args)
    return _DCACHE[pair]


def all_pairs(args):
    pairs = list(FIELD_PAIRS) + BENCH_DEFAULT
    root = find_msl_root()
    if root is None:
        log("  [MSL] train/test 폴더를 찾지 못해 MSL/SMAP은 건너뜁니다.")
        return pairs
    try:
        csvp = None
        for c in [root, root.parent, root.parent.parent]:
            if (c / "labeled_anomalies.csv").exists():
                csvp = c / "labeled_anomalies.csv"; break
        if csvp is None:
            hits = list(Path("data").rglob("labeled_anomalies.csv"))
            csvp = hits[0] if hits else None
        if csvp is None:
            log("  [MSL] labeled_anomalies.csv 없음 → 건너뜀"); return pairs
        la = pd.read_csv(csvp)
        import ast
        rows = []
        for _, r in la.iterrows():
            f1, f2 = root / "train" / f"{r.chan_id}.npy", root / "test" / f"{r.chan_id}.npy"
            if not (f1.exists() and f2.exists()):
                continue
            a = np.load(f1)
            seq = ast.literal_eval(r.anomaly_sequences)
            ratio = sum(e - s + 1 for s, e in seq) / max(r.num_values, 1)
            rows.append(dict(chan=r.chan_id, sc=r.spacecraft, dim=a.shape[1],
                             train=len(a), ratio=ratio))
        df = pd.DataFrame(rows)
        if df.empty:
            log("  [MSL] npy 파일 없음 → 건너뜀"); return pairs
        picked = []
        for (sc, dim), g in df.groupby(["sc", "dim"]):
            ok = g[(g.train >= 2000) & (g.ratio.between(0.02, 0.25))]
            src = g.sort_values("train", ascending=False)
            for _, t in ok.sort_values("train", ascending=False).head(2).iterrows():
                s = src[src.chan != t.chan].head(1)
                if len(s):
                    picked.append((s.iloc[0].chan, t.chan))
        picked = picked[:args.msl_max]
        if picked:
            added = register_msl_pairs(picked)
            log(f"  [MSL] 자동 선정 pair: {added}")
            pairs += added
        else:
            log("  [MSL] 조건(train≥2000, 이상비율 2~25%)을 만족하는 조합 없음")
    except Exception as e:
        logerr("msl_setup", e)
    return pairs


# ===========================================================================
# 모델 구성
# ===========================================================================
def build_method(name, tf, D, pool, args, device, alpha=0.0):
    bs = train_cfg.batch_size
    ld = get_latent_dim(MODEL_NAME)
    cut = max(int(len(pool) * .8), 1)
    tr = pool[:cut]
    va = pool[cut:] if len(pool) - cut >= 5 else pool[-5:]
    if name == "source_only":
        return tf, 0
    if name == "adapter_isca":
        cur, m = pool, None
        for keep in ISCA_KEEP:
            if keep < 1.0 and m is not None:
                sc = tf_scores(m, pool, bs, device)
                cur = pool[np.argsort(sc)[:max(int(len(pool) * keep), 10)]]
            c2 = max(int(len(cur) * .8), 1)
            m = fresh_adapted(tf, "bottleneck", ld, device)
            train_adapter(m, cur[:c2], cur[c2:] if len(cur) - c2 >= 5 else cur[-5:],
                          bs, args.epochs_adapt, args.adapt_lr, device)
        return m, sum(p.numel() for p in m.adapter.parameters())
    if name == "full_ft":
        m = copy.deepcopy(tf)
        full_finetune(m, torch.utils.data.DataLoader(torch.FloatTensor(tr), batch_size=bs, shuffle=True),
                      torch.utils.data.DataLoader(torch.FloatTensor(va), batch_size=bs, shuffle=False),
                      args.epochs_adapt, args.ft_lr, device)
        return m, sum(p.numel() for p in m.parameters())
    if name == "adapter_output":
        m = OutputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
    elif name == "adapter_input":
        m = InputAdapted(copy.deepcopy(tf), D["n_feat"]).to(device)
    elif name in ("adapter_latent", "adapter_mmd"):
        m = fresh_adapted(tf, "bottleneck", ld, device)
    else:
        raise ValueError(name)
    if name == "adapter_mmd" and alpha > 0:
        from exp3_final import source_latents_modes
        sl, sm = source_latents_modes(tf, D["src"][:4000], bs, device)
        train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device,
                      trim_q=0.0, align="mmd", alpha=alpha, src_lat=sl, src_mode=sm)
    else:
        train_adapter(m, tr, va, bs, args.epochs_adapt, args.adapt_lr, device)
    return m, sum(p.numel() for p in m.adapter.parameters())


def run_cell(D, seed, meth, contam, args, device, stage, alpha=0.0):
    ck = OUT / "ckpt" / f"{D['pair']}__s{seed}__h{N_HEADS}__lb{LB:g}__w{D['W']}.pt"
    tf = pretrain(D, seed, N_HEADS, LB, args, device, ck)
    model_cfg.n_heads = N_HEADS
    rng = np.random.default_rng(seed * 991 + int(contam * 100))
    pool, pgt = make_pool(D, contam, "real" if len(D["anom"]) else "synth", rng)
    set_all_seeds(seed)
    with AdaptTimer() as _t:
        m, npar = build_method(meth, tf, D, pool, args, device, alpha)
    el, peak = _t.elapsed, _t.peak_mem_mb
    bs = train_cfg.batch_size
    s_te = tf_scores(m, D["eval_x"], bs, device)
    r = evaluate_all(s_te, D["eval_y"], tf_scores(m, pool, bs, device))
    # 배포 효율: 추론 지연 (window당 ms)
    try:
        @torch.no_grad()
        def _fn(x):
            hh = m(x); b_, _, _ = select_best_head(hh, x)
            return torch.mean(torch.abs(b_ - x), dim=(1, 2))
        _ld = torch.utils.data.DataLoader(torch.FloatTensor(D["eval_x"]),
                                          batch_size=bs, shuffle=False)
        lat = measure_inference_latency(_fn, _ld, device)
    except Exception:
        lat = np.nan
    np.save(OUT / "scores" / f"{D['pair']}__{meth}__c{contam:.2f}__a{alpha:g}__s{seed}.npy", s_te)
    return dict(stage=stage, pair=D["pair"], family=D["family"], seed=seed,
                method=meth, contam=contam, alpha=alpha,
                contam_actual=float(pgt.mean()), n_heads=N_HEADS, lb=LB,
                trainable_params=npar, deploy_storage_kb=npar * 4 / 1024,
                adapt_time_s=el, peak_mem_mb=peak, infer_ms_per_window=lat,
                n_eval_windows=len(D["eval_x"]), n_pool_windows=len(pool), **r)


# ===========================================================================
# S3 Domain gap 지표 (라벨 불필요 · 입력공간과 잠재공간 모두)
# ===========================================================================
def sliced_wasserstein(X, Y, n_proj=128, seed=0):
    """다변량 Wasserstein-1 근사 (랜덤 1D 투영 평균)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    P = rng.normal(size=(d, n_proj))
    P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
    A, Bm = X @ P, Y @ P
    return float(np.mean([sps.wasserstein_distance(A[:, i], Bm[:, i])
                          for i in range(n_proj)]))


def gap_metrics_raw(X, Y, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(len(X), n), replace=False)]
    Y = Y[rng.choice(len(Y), min(len(Y), n), replace=False)]
    Xf, Yf = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    out = {"wasserstein": sliced_wasserstein(Xf, Yf, seed=seed),
           "mean_l2": float(np.linalg.norm(Xf.mean(0) - Yf.mean(0)))}
    try:
        cs, ct = np.cov(Xf.T), np.cov(Yf.T)
        out["coral"] = float(((cs - ct) ** 2).sum() / (4 * Xf.shape[1] ** 2))
    except Exception:
        out["coral"] = np.nan
    xt, yt = torch.tensor(Xf[:600]), torch.tensor(Yf[:600])
    Z = torch.cat([xt, yt]); dm = torch.cdist(Z, Z)
    med = torch.median(dm[dm > 0]).clamp(min=1e-6)
    k = lambda a, b: torch.exp(-torch.cdist(a, b) ** 2 / (2 * med ** 2))
    out["mmd_rbf"] = float(k(xt, xt).mean() + k(yt, yt).mean() - 2 * k(xt, yt).mean())
    # energy distance (1D 투영 평균)
    rng2 = np.random.default_rng(seed + 1)
    P = rng2.normal(size=(Xf.shape[1], 32)); P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
    out["energy"] = float(np.mean([sps.energy_distance((Xf @ P)[:, i], (Yf @ P)[:, i])
                                   for i in range(32)]))
    return out


# ===========================================================================
# 스테이지
# ===========================================================================
def stage_main(pairs, args, device, rows, done):
    KEY = ["stage", "pair", "seed", "method", "contam", "alpha"]
    tot = len(pairs) * len(args.seeds) * len(MAIN_METHODS); i = 0
    for pair in pairs:
        try:
            D = get_data(pair, args)
        except Exception as e:
            logerr(f"prepare:{pair}", e); continue
        log(f"[S1] {pair} (eval={len(D['eval_x'])}w 이상={D['eval_y'].mean():.4f})")
        for seed in args.seeds:
            for meth in MAIN_METHODS:
                i += 1
                k = ("S1", pair, str(seed), meth, "0.0", "0.0")
                if k in done:
                    continue
                try:
                    rows.append(run_cell(D, seed, meth, 0.0, args, device, "S1"))
                    done.add(k); save(rows, RES)
                    log(f"  [S1 {i}/{tot}] {pair} s{seed} {meth} "
                        f"AUROC={rows[-1]['auroc']:.4f}")
                except Exception as e:
                    logerr(f"S1:{pair}:{seed}:{meth}", e)


def stage_robustness(pairs, args, device, rows, done):
    tot = len(pairs) * len(args.seeds) * len(CONTAM_LEVELS) * len(ROBUST_METHODS); i = 0
    for pair in pairs:
        try:
            D = get_data(pair, args)
        except Exception as e:
            logerr(f"prepare:{pair}", e); continue
        inj = "real" if len(D["anom"]) else "synth"
        log(f"[S2] {pair} (오염 주입 방식={inj})")
        for seed in args.seeds:
            for c in CONTAM_LEVELS:
                for meth in ROBUST_METHODS:
                    i += 1
                    k = ("S2", pair, str(seed), meth, f"{c}", "0.0")
                    if k in done:
                        continue
                    try:
                        rows.append(run_cell(D, seed, meth, c, args, device, "S2"))
                        done.add(k); save(rows, RES)
                        if i % 10 == 0:
                            log(f"  [S2 {i}/{tot}] {pair} c={c:.2f} {meth} "
                                f"AUROC={rows[-1]['auroc']:.4f}")
                    except Exception as e:
                        logerr(f"S2:{pair}:{seed}:{meth}:{c}", e)


def stage_domaingap(pairs, args, device, grows, gdone):
    for pair in pairs:
        k = ("S3", pair)
        if k in gdone:
            continue
        try:
            D = get_data(pair, args)
            raw = gap_metrics_raw(D["src"], D["clean"])
            ck = OUT / "ckpt" / f"{pair}__s0__h{N_HEADS}__lb{LB:g}__w{D['W']}.pt"
            lat = {}
            if ck.exists():
                tf = pretrain(D, 0, N_HEADS, LB, args, device, ck)
                bs = train_cfg.batch_size
                from exp2_robust import latents
                zs = latents(tf, D["src"][:2000], bs, device, apply_adapter=False)
                zt = latents(tf, D["clean"][:2000], bs, device, apply_adapter=False)
                lat = {f"lat_{a}": b for a, b in gap_metrics_raw(zs, zt).items()}
            grows.append(dict(stage="S3", pair=pair, family=D["family"],
                              transfer_level=transfer_level(pair),
                              n_feat=D["n_feat"], window=D["W"],
                              **{f"raw_{a}": b for a, b in raw.items()}, **lat))
            gdone.add(k); save(grows, GAP)
            log(f"[S3] {pair} Wasserstein={raw['wasserstein']:.4f} MMD={raw['mmd_rbf']:.4f}")
        except Exception as e:
            logerr(f"S3:{pair}", e)


def stage_alpha(pairs, args, device, rows, done):
    sub = pairs[:args.alpha_pairs]
    for pair in sub:
        try:
            D = get_data(pair, args)
        except Exception as e:
            logerr(f"prepare:{pair}", e); continue
        log(f"[S4] {pair} — MMD strength curve")
        for seed in args.seeds:
            for a in ALPHAS:
                meth = "adapter_latent" if a == 0 else "adapter_mmd"
                k = ("S4", pair, str(seed), meth, "0.0", f"{a}")
                if k in done:
                    continue
                try:
                    rows.append(run_cell(D, seed, meth, 0.0, args, device, "S4", alpha=a))
                    done.add(k); save(rows, RES)
                    log(f"  [S4] {pair} s{seed} α={a:g} AUROC={rows[-1]['auroc']:.4f}")
                except Exception as e:
                    logerr(f"S4:{pair}:{seed}:{a}", e)


def stage_baselines(pairs, args, device, rows, done):
    try:
        import exp6_baselines as X
    except Exception as e:
        logerr("import exp6", e); return
    for pair in pairs:
        try:
            D = get_data(pair, args)
        except Exception as e:
            logerr(f"prepare:{pair}", e); continue
        log(f"[S5] {pair} — USAD / TranAD")
        sw = args.usad_window
        for seed in args.seeds:
            rng = np.random.default_rng(seed * 991)
            pool, _ = make_pool(D, 0.0, "real" if len(D["anom"]) else "synth", rng)
            cut = max(int(len(pool) * .8), 1)
            for name in ["usad", "tranad"]:
                k = ("S5", pair, str(seed), name, "0.0", "0.0")
                if k in done:
                    continue
                if name == "tranad" and args.no_tranad:
                    continue
                try:
                    set_all_seeds(seed)
                    if name == "usad":
                        # Fit scaling parameters using only the USAD training split.
                        scale_min, scale_range = X.usad_fit_minmax(pool[:cut])

                        tr_usad = X.usad_apply_minmax(
                            pool[:cut], scale_min, scale_range
                        )
                        va_usad = X.usad_apply_minmax(
                            pool[cut:], scale_min, scale_range
                        )
                        eval_usad = X.usad_apply_minmax(
                            D["eval_x"], scale_min, scale_range
                        )
                        pool_usad = X.usad_apply_minmax(
                            pool, scale_min, scale_range
                        )

                        m = X.USAD(sw * D["n_feat"], args.z_size).to(device)
                        m = X.usad_train(
                            m,
                            X.usad_flat(tr_usad, sw),
                            X.usad_flat(va_usad, sw),
                            args.epochs_usad,
                            1e-3,
                            device
                        )
                        s_te = X.usad_agg(
                            X.usad_score(
                                m, X.usad_flat(eval_usad, sw), device
                            ),
                            len(D["eval_x"]),
                            sw,
                            D["W"]
                        )
                        s_cal = X.usad_agg(
                            X.usad_score(
                                m, X.usad_flat(pool_usad, sw), device
                            ),
                            len(pool),
                            sw,
                            D["W"]
                        )
                    else:
                        from tranad_model import (TranAD, tranad_train as _ttr,
                                                  tranad_score as _tsc)
                        m = TranAD(D["n_feat"], n_window=args.tranad_window).to(device)
                        m = _ttr(m, pool[:cut], pool[cut:], args.epochs_tranad,
                                 args.tranad_lr, device, sub_w=args.tranad_window)
                        s_te = _tsc(m, D["eval_x"], device, sub_w=args.tranad_window)
                        s_cal = _tsc(m, pool, device, sub_w=args.tranad_window)
                    r = evaluate_all(s_te, D["eval_y"], s_cal)
                    npar = sum(p.numel() for p in m.parameters())
                    rows.append(dict(stage="S5", pair=pair, family=D["family"], seed=seed,
                                     method=name, contam=0.0, alpha=0.0,
                                     contam_actual=0.0, n_heads=N_HEADS, lb=LB,
                                     trainable_params=npar,
                                     deploy_storage_kb=npar * 4 / 1024,
                                     adapt_time_s=np.nan, **r))
                    done.add(k); save(rows, RES)
                    log(f"  [S5] {pair} s{seed} {name} AUROC={r['auroc']:.4f}")
                except Exception as e:
                    logerr(f"S5:{pair}:{seed}:{name}", e)


def stage_indicator(pairs, args, rows_out):
    """저장된 score 재사용 — 재학습 없음."""
    inds = ["rms", "kurtosis", "peak", "std", "combined"]
    import run_experiment1 as R
    res = []
    for pair in pairs:
        cfg = R.DOMAIN_PAIRS.get(pair)
        if cfg is None or cfg.get("kind") != "bearing":
            continue
        for ind in inds:
            try:
                lab = pseudo_label_by_indicator(cfg["tgt"], ind)["labels"]
                wl = L.point_to_window_labels(lab, 20, rule="any")
                for f in sorted((OUT / "scores").glob(f"{pair}__*__c0.00__a0__s*.npy")):
                    s = np.load(f); y = wl[:len(s)]
                    if len(np.unique(y)) < 2:
                        continue
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    res.append(dict(stage="S6", pair=pair, indicator=ind,
                                    method=f.stem.split("__")[1],
                                    seed=int(f.stem.split("s")[-1]),
                                    auroc=roc_auc_score(y, s),
                                    aupr=average_precision_score(y, s),
                                    anomaly_ratio=float(y.mean())))
            except Exception as e:
                logerr(f"S6:{pair}:{ind}", e)
    if res:
        pd.DataFrame(res).to_csv(OUT / "indicator.csv", index=False)
        log(f"[S6] 지표 변형 {len(res)}행 저장")


def stage_selfrefine(pairs, args, device, rows, done):
    """★자가정제를 라벨 없이 언제 적용할지 판단 가능한가 (진단 지표 수집)."""
    res = []
    for pair in pairs:
        try:
            D = get_data(pair, args)
        except Exception as e:
            logerr(f"prepare:{pair}", e); continue
        for seed in args.seeds[:1]:
            for c in [0.0, 0.2, 0.4]:
                k = ("S7", pair, str(seed), "diag", f"{c}", "0.0")
                if k in done:
                    continue
                try:
                    ck = OUT / "ckpt" / f"{pair}__s{seed}__h{N_HEADS}__lb{LB:g}__w{D['W']}.pt"
                    tf = pretrain(D, seed, N_HEADS, LB, args, device, ck)
                    rng = np.random.default_rng(seed * 991 + int(c * 100))
                    pool, pgt = make_pool(D, c, "real" if len(D["anom"]) else "synth", rng)
                    bs = train_cfg.batch_size
                    cut = max(int(len(pool) * .8), 1)
                    set_all_seeds(seed)
                    ad = fresh_adapted(tf, "bottleneck", get_latent_dim(MODEL_NAME), device)
                    train_adapter(ad, pool[:cut], pool[cut:], bs,
                                  args.epochs_adapt, args.adapt_lr, device)
                    s = tf_scores(ad, pool, bs, device)
                    keep = np.argsort(s)[:int(len(pool) * .5)]
                    res.append(dict(stage="S7", pair=pair, seed=seed, contam=c,
                                    contam_pool=float(pgt.mean()),
                                    contam_kept=float(pgt[keep].mean()),
                                    cleaning_works=bool(pgt.mean() <= 1e-9 or
                                                        pgt[keep].mean() < pgt.mean()),
                                    score_skew=float(sps.skew(s)),
                                    score_kurtosis=float(sps.kurtosis(s)),
                                    score_gap_ratio=float(np.quantile(s, .95) /
                                                          max(np.median(s), 1e-9)),
                                    bimodality=float((sps.skew(s) ** 2 + 1) /
                                                     (sps.kurtosis(s) + 3 + 1e-9))))
                    done.add(k)
                    pd.DataFrame(res).to_csv(OUT / "selfrefine.csv", index=False)
                    log(f"  [S7] {pair} c={c:.2f} pool오염={pgt.mean():.3f}→"
                        f"유지 {pgt[keep].mean():.3f} "
                        f"{'해당없음' if pgt.mean()<=1e-9 else ('정상' if pgt[keep].mean()<pgt.mean() else '★역효과')}")
                except Exception as e:
                    logerr(f"S7:{pair}:{c}", e)


# ===========================================================================
# S8 그림 · S9 분석
# ===========================================================================
def make_figures():
    try:
        df = pd.read_csv(RES)
    except Exception as e:
        logerr("fig:read", e); return
    FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

    # (1) Robustness curve — pair 단위 평균±표준편차
    d = df[df.stage == "S2"]
    if len(d):
        pv = d.groupby(["method", "contam", "pair"])["auroc"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 5))
        style = {"source_only": ("gray", "s", "No adaptation"),
                 "adapter_latent": ("#1f77b4", "o", "Adapter (1%)"),
                 "adapter_isca": ("#1E6E43", "^", "Adapter + self-refine"),
                 "full_ft": ("#9E2B25", "D", "Full fine-tuning")}
        for m, (c, mk, lab) in style.items():
            g = pv[pv.method == m]
            if not len(g):
                continue
            st = g.groupby("contam")["auroc"].agg(["mean", "std", "count"])
            ax.errorbar(st.index * 100, st["mean"],
                        yerr=st["std"] / np.sqrt(st["count"].clip(lower=1)),
                        marker=mk, color=c, capsize=3, lw=1.8, label=lab)
        ax.set_xlabel("Contamination ratio of adaptation windows (%)")
        ax.set_ylabel("AUROC (pair-level mean ± s.e.)")
        ax.set_title("Robustness to contaminated target data")
        ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "fig_robustness_curve.png", dpi=200)
        plt.close(fig)
        # pair별 표
        pv.pivot_table(index="pair", columns=["method", "contam"], values="auroc"
                       ).round(4).to_csv(OUT / "table_robustness_pairlevel.csv")
        log("[S8] fig_robustness_curve.png 저장")

    # (2) Domain gap ↔ 개선폭 (S3 미실행 시 건너뜀)
    if not Path(GAP).exists():
        log("[S8] domaingap.csv 없음 → Domain gap 그림 건너뜀 (S3 먼저 실행)")
    else:
      try:
        gp = pd.read_csv(GAP)
        m = df[(df.stage == "S1")]
        base = m[m.method == "source_only"].groupby("pair")["auroc"].mean()
        ad = m[m.method == "adapter_latent"].groupby("pair")["auroc"].mean()
        imp = (ad - base).dropna().rename("improvement").reset_index()
        mg = gp.merge(imp, on="pair")
        if len(mg) > 3:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, col in zip(axes, ["raw_wasserstein", "raw_mmd_rbf", "raw_coral"]):
                if col not in mg:
                    continue
                s = mg[[col, "improvement", "family"]].dropna()
                for fam, gg in s.groupby("family"):
                    ax.scatter(gg[col], gg["improvement"], label=fam, s=45, alpha=.85)
                if len(s) > 3:
                    rho, pv_ = sps.spearmanr(s[col], s["improvement"])
                    ax.set_title(f"{col}\nSpearman ρ={rho:+.3f} (p={pv_:.3f})", fontsize=10)
                ax.set_xlabel(col); ax.set_ylabel("AUROC improvement"); ax.grid(alpha=.3)
            axes[0].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(FIG / "fig_domaingap_vs_improvement.png", dpi=200)
            plt.close(fig)
            mg.round(6).to_csv(OUT / "table_domaingap.csv", index=False)
            log("[S8] fig_domaingap_vs_improvement.png 저장")
      except Exception as e:
        logerr("fig:gap", e)

    # (2b) 전이 수준별 Domain gap (★cross-machine 강조)
    try:
        if Path(GAP).exists():
            gp = pd.read_csv(GAP)
            if "transfer_level" not in gp:
                gp["transfer_level"] = gp["pair"].map(transfer_level)
            order = ["cross-condition", "cross-machine", "sim-to-real",
                     "benchmark-SMD", "benchmark-Boiler", "benchmark-MSL/SMAP"]
            gp["transfer_level"] = pd.Categorical(gp["transfer_level"],
                                                  [o for o in order if o in set(gp.transfer_level)],
                                                  ordered=True)
            cols = [c for c in ["raw_wasserstein", "raw_mmd_rbf"] if c in gp]
            if cols and gp["transfer_level"].notna().any():
                fig, axes = plt.subplots(1, len(cols), figsize=(6.2 * len(cols), 4.3))
                axes = np.atleast_1d(axes)
                for ax, c in zip(axes, cols):
                    g = gp.dropna(subset=["transfer_level"])
                    labs = list(g["transfer_level"].cat.categories)
                    data = [g[g.transfer_level == L][c].dropna().values for L in labs]
                    xs = np.arange(len(labs))
                    means = [np.mean(v) if len(v) else np.nan for v in data]
                    ax.bar(xs, means, color="#9E2B25", alpha=.75)
                    for i, v in enumerate(data):
                        if len(v):
                            ax.scatter(np.full(len(v), i) + np.random.uniform(-.08, .08, len(v)),
                                       v, color="k", s=18, zorder=3)
                    ax.set_xticks(xs)
                    ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
                    ax.set_ylabel(c.replace("raw_", "").replace("_", " ").title())
                    ax.set_title(f"Domain gap by transfer level\n({c.replace('raw_','')})",
                                 fontsize=10)
                    ax.grid(alpha=.3, axis="y")
                fig.tight_layout()
                fig.savefig(FIG / "fig_domaingap_by_level.png", dpi=200)
                plt.close(fig)
                log("[S8] fig_domaingap_by_level.png 저장 (cross-machine 비교)")
    except Exception as e:
        logerr("fig:gap_level", e)

    # (3) MMD strength curve
    d4 = df[df.stage == "S4"]
    if len(d4):
        try:
            st = d4.groupby("alpha")["auroc"].agg(["mean", "std", "count"])
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            x = np.arange(len(st))
            ax.errorbar(x, st["mean"], yerr=st["std"] / np.sqrt(st["count"].clip(lower=1)),
                        marker="o", color="#9E2B25", capsize=3)
            ax.set_xticks(x); ax.set_xticklabels([f"{a:g}" for a in st.index])
            ax.axhline(st["mean"].iloc[0], ls="--", color="gray",
                       label="α = 0 (no alignment)")
            ax.set_xlabel("MMD weight α"); ax.set_ylabel("AUROC (mean ± s.e.)")
            ax.set_title("Effect of distribution-alignment strength")
            ax.grid(alpha=.3); ax.legend()
            fig.tight_layout(); fig.savefig(FIG / "fig_mmd_strength_curve.png", dpi=200)
            plt.close(fig)
            log("[S8] fig_mmd_strength_curve.png 저장")
        except Exception as e:
            logerr("fig:alpha", e)

    # (4) Pareto
    d1 = df[df.stage == "S1"]
    if len(d1):
        try:
            g = d1.groupby("method").agg(auroc=("auroc", "mean"),
                                         par=("trainable_params", "mean")).reset_index()
            fig, ax = plt.subplots(figsize=(6.5, 4.6))
            for _, r in g.iterrows():
                x = max(r["par"], 1)
                ax.scatter(x, r["auroc"], s=90)
                ax.annotate(r["method"], (x, r["auroc"]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
            ax.set_xscale("log"); ax.set_xlabel("Trainable parameters (log)")
            ax.set_ylabel("AUROC (mean over pairs)")
            ax.set_title("Performance–efficiency trade-off"); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(FIG / "fig_pareto.png", dpi=200)
            plt.close(fig)
            log("[S8] fig_pareto.png 저장")
        except Exception as e:
            logerr("fig:pareto", e)


def analyze():
    try:
        df = pd.read_csv(RES)
    except Exception as e:
        logerr("analyze:read", e); return
    out = []
    log("\n" + "=" * 74)
    for stage, d in df.groupby("stage"):
        npair = d.pair.nunique()
        log(f"[{stage}] pair {npair}개 · {len(d)}행")
        if stage in ("S1", "S5"):
            t = d.pivot_table(index="pair", columns="method", values="auroc").round(4)
            t.to_csv(OUT / f"table_{stage}_auroc.csv")
            base = d[d.method == "source_only"].groupby(["pair", "seed"])["auroc"].mean()
            g = d[d.method != "source_only"].copy()
            g["imp"] = g.apply(lambda r: r["auroc"] - base.get((r["pair"], r["seed"]), np.nan), axis=1)
            rr = []
            for m, gg in g.groupby("method"):
                v = gg.groupby("pair")["imp"].mean().dropna()
                if len(v) < 3:
                    continue
                try:
                    p = sps.wilcoxon(v)[1]
                except Exception:
                    p = np.nan
                rr.append(dict(stage=stage, method=m, n_pairs=len(v), mean=v.mean(),
                               median=v.median(), win=f"{int((v>0).sum())}/{len(v)}",
                               std=v.std(), wilcoxon_p=p))
            if rr:
                s = pd.DataFrame(rr).sort_values("wilcoxon_p")
                k = len(s)
                s["p_holm"] = holm_adjust(s["wilcoxon_p"].values)
                s["sig"] = s["p_holm"] < .05
                log(s.round(4).to_string(index=False))
                out.append(s)
        if stage == "S2":
            t = d.pivot_table(index="contam", columns="method", values="auroc").round(4)
            log("\n오염별 AUROC:\n" + t.to_string())
            t.to_csv(OUT / "table_S2_contamination.csv")
            drop = (t.iloc[0] - t.iloc[-1]).round(4)
            log("\n0→40% 하락폭:\n" + drop.to_string())
        if stage == "S4":
            t = d.groupby("alpha")["auroc"].agg(["mean", "std", "count"]).round(4)
            log("\nα 곡선:\n" + t.to_string())
            t.to_csv(OUT / "table_S4_alpha.csv")
    if out:
        pd.concat(out).to_csv(OUT / "table_significance.csv", index=False)
    log(f"\n결과: {OUT.resolve()}")




# ===========================================================================
# 논문용 표 생성 (그대로 붙여 쓸 수 있는 형태: "mean ± std")
# ===========================================================================
NICE = {"source_only": "Source-only", "adapter_latent": "Adapter (latent, ours)",
        "adapter_output": "Adapter (output)", "adapter_input": "Adapter (input)",
        "adapter_isca": "Adapter + self-refine", "full_ft": "Full fine-tuning",
        "adapter_mmd": "Adapter + MMD", "usad": "USAD", "tranad": "TranAD"}


def _ms(v, nd=4):
    v = pd.Series(v).astype(float).dropna()
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v.iloc[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def make_paper_tables():
    """논문 Table 1~7을 CSV와 LaTeX로 저장."""
    T = OUT / "paper_tables"; T.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(RES)
    except Exception as e:
        logerr("paper:read", e); return
    df["level"] = df["pair"].map(transfer_level)

    def dump(tab, name, idx=False):
        tab.to_csv(T / f"{name}.csv", index=idx)
        try:
            with open(T / f"{name}.tex", "w", encoding="utf-8") as f:
                f.write(tab.to_latex(index=idx, escape=True))
        except Exception:
            pass
        log(f"  [표] {name}.csv 저장 ({len(tab)}행)")

    # ── Table 1: 주 결과 (Pair × Method, AUROC/AUPR/F1) ──────────────────
    d1 = df[df.stage == "S1"]
    if len(d1):
        rows = []
        for (lvl, pair), g in d1.groupby(["level", "pair"]):
            r = {"Transfer level": lvl, "Pair": pair,
                 "#anomaly ratio": f"{g['anomaly_ratio'].mean():.4f}"}
            for m in MAIN_METHODS:
                gg = g[g.method == m]
                r[NICE.get(m, m)] = _ms(gg["auroc"])
            rows.append(r)
        dump(pd.DataFrame(rows).sort_values(["Transfer level", "Pair"]), "T1_main_AUROC")
        for metric, nm in [("aupr", "T1b_main_AUPR"), ("f1_far0.01", "T1c_main_F1_FAR1pct")]:
            if metric not in d1:
                continue
            rows = []
            for (lvl, pair), g in d1.groupby(["level", "pair"]):
                r = {"Transfer level": lvl, "Pair": pair}
                for m in MAIN_METHODS:
                    r[NICE.get(m, m)] = _ms(g[g.method == m][metric])
                rows.append(r)
            dump(pd.DataFrame(rows).sort_values(["Transfer level", "Pair"]), nm)

        # ── Table 2: 방법별 종합 + 통계 검정 ──────────────────────────────
        base = d1[d1.method == "source_only"].groupby(["pair", "seed"])["auroc"].mean()
        g2 = d1[d1.method != "source_only"].copy()
        g2["imp"] = g2.apply(lambda r: r["auroc"] - base.get((r["pair"], r["seed"]), np.nan), axis=1)
        rr = []
        for m, gg in g2.groupby("method"):
            v = gg.groupby("pair")["imp"].mean().dropna()
            if len(v) < 3:
                continue
            try:
                pw = sps.wilcoxon(v)[1]
            except Exception:
                pw = np.nan
            try:
                pt = sps.ttest_1samp(v, 0)[1]
            except Exception:
                pt = np.nan
            rr.append({"Method": NICE.get(m, m), "#pairs": len(v),
                       "Mean ΔAUROC": f"{v.mean():+.4f}",
                       "Median ΔAUROC": f"{v.median():+.4f}",
                       "Std": f"{v.std(ddof=1):.4f}",
                       "Improved pairs": f"{int((v>0).sum())}/{len(v)}",
                       "Wilcoxon p": f"{pw:.4f}", "t-test p": f"{pt:.4f}",
                       "_p": pw})
        if rr:
            s2 = pd.DataFrame(rr).sort_values("_p")
            k = len(s2)
            ph = holm_adjust(s2["_p"].values)
            s2["Holm-adj p"] = [f"{x:.4f}" for x in ph]
            s2["Significant (α=.05)"] = ["Yes" if x < .05 else "No" for x in ph]
            dump(s2.drop(columns=["_p"]), "T2_summary_statistics")

        # ── Table 3: 성능–효율 ────────────────────────────────────────────
        rows = []
        for m in MAIN_METHODS + ["usad", "tranad"]:
            g = df[(df.method == m) & (df.stage.isin(["S1", "S5"]))]
            if not len(g):
                continue
            rows.append({"Method": NICE.get(m, m),
                         "Trainable params": f"{g['trainable_params'].mean():,.0f}",
                         "Trainable (%)": f"{100*g['trainable_params'].mean()/815912:.2f}",
                         "Deploy size (KB)": f"{g['deploy_storage_kb'].mean():.1f}",
                         "Adaptation time (s)": _ms(g["adapt_time_s"], 2),
                         "Inference (ms/window)": _ms(g.get("infer_ms_per_window", []), 4),
                         "Peak memory (MB)": _ms(g.get("peak_mem_mb", []), 1),
                         "Mean AUROC": _ms(g["auroc"])})
        if rows:
            dump(pd.DataFrame(rows), "T3_performance_efficiency")

    # ── Table 4: 오염 강건성 ──────────────────────────────────────────────
    d2 = df[df.stage == "S2"]
    if len(d2):
        rows = []
        for c, g in d2.groupby("contam"):
            r = {"Contamination (%)": f"{c*100:.0f}"}
            for m in ROBUST_METHODS:
                pm = g[g.method == m].groupby("pair")["auroc"].mean()
                r[NICE.get(m, m)] = _ms(pm)
            rows.append(r)
        t4 = pd.DataFrame(rows)
        # 하락폭 행 추가
        drop = {"Contamination (%)": "Drop (0→40%)"}
        for m in ROBUST_METHODS:
            a = d2[(d2.method == m) & (d2.contam == 0.0)].groupby("pair")["auroc"].mean()
            b = d2[(d2.method == m) & (d2.contam == 0.4)].groupby("pair")["auroc"].mean()
            v = (b - a).dropna()
            drop[NICE.get(m, m)] = f"{v.mean():+.4f}" if len(v) else "—"
        t4 = pd.concat([t4, pd.DataFrame([drop])], ignore_index=True)
        dump(t4, "T4_contamination_robustness")

    # ── Table 5: Domain gap (전이 수준별, cross-machine 강조) ─────────────
    if Path(GAP).exists():
        try:
            gp = pd.read_csv(GAP)
            if "transfer_level" not in gp:
                gp["transfer_level"] = gp["pair"].map(transfer_level)
            cols = [c for c in ["raw_wasserstein", "raw_mmd_rbf", "raw_coral",
                                "raw_energy", "raw_mean_l2"] if c in gp]
            t5 = gp[["transfer_level", "pair"] + cols].copy()
            t5.columns = ["Transfer level", "Pair"] + \
                [c.replace("raw_", "").replace("_", " ").title() for c in cols]
            dump(t5.sort_values(["Transfer level", "Pair"]), "T5_domain_gap_per_pair")
            agg = gp.groupby("transfer_level")[cols].agg(["mean", "std"]).round(5)
            agg.to_csv(T / "T5b_domain_gap_by_level.csv")
            log("  [표] T5b_domain_gap_by_level.csv 저장")
            # gap ↔ 개선폭 상관
            if len(df[df.stage == "S1"]):
                d = df[df.stage == "S1"]
                b0 = d[d.method == "source_only"].groupby("pair")["auroc"].mean()
                a0 = d[d.method == "adapter_latent"].groupby("pair")["auroc"].mean()
                imp = (a0 - b0).dropna().rename("dAUROC").reset_index()
                mg = gp.merge(imp, on="pair")
                cr = []
                for c in cols:
                    s_ = mg[[c, "dAUROC"]].dropna()
                    if len(s_) > 3:
                        rho, pv = sps.spearmanr(s_[c], s_["dAUROC"])
                        pr, pp = sps.pearsonr(s_[c], s_["dAUROC"])
                        cr.append({"Gap metric": c.replace("raw_", ""), "n": len(s_),
                                   "Spearman ρ": f"{rho:+.3f}", "p (Spearman)": f"{pv:.4f}",
                                   "Pearson r": f"{pr:+.3f}", "p (Pearson)": f"{pp:.4f}"})
                if cr:
                    dump(pd.DataFrame(cr), "T5c_gap_vs_improvement_correlation")
                mg.round(6).to_csv(T / "T5d_gap_and_improvement_raw.csv", index=False)
        except Exception as e:
            logerr("paper:T5", e)

    # ── Table 6: 발표된 방법 비교 ─────────────────────────────────────────
    d5 = df[df.stage.isin(["S1", "S5"])]
    if len(d5) and (d5.method.isin(["usad", "tranad"]).any()):
        rows = []
        for pair, g in d5.groupby("pair"):
            r = {"Pair": pair}
            for m in ["source_only", "usad", "tranad", "full_ft", "adapter_latent"]:
                r[NICE.get(m, m)] = _ms(g[g.method == m]["auroc"])
            rows.append(r)
        dump(pd.DataFrame(rows), "T6_baseline_comparison")

    # ── Table 7: MMD strength curve ──────────────────────────────────────
    d4 = df[df.stage == "S4"]
    if len(d4):
        rows = []
        for a, g in d4.groupby("alpha"):
            pm = g.groupby("pair")["auroc"].mean()
            rows.append({"α (MMD weight)": f"{a:g}", "#pairs": len(pm),
                         "AUROC": _ms(pm),
                         "Δ vs α=0": ""})
        t7 = pd.DataFrame(rows)
        try:
            a0 = d4[d4.alpha == 0.0].groupby("pair")["auroc"].mean()
            ds = []
            for a, g in d4.groupby("alpha"):
                v = (g.groupby("pair")["auroc"].mean() - a0).dropna()
                ds.append(f"{v.mean():+.4f}" if len(v) else "—")
            t7["Δ vs α=0"] = ds
        except Exception:
            pass
        dump(t7, "T7_mmd_strength_curve")

    # ── Table 8: 자가정제 진단 ────────────────────────────────────────────
    f8 = OUT / "selfrefine.csv"
    if f8.exists():
        try:
            s8 = pd.read_csv(f8)
            t8 = s8.groupby("contam").agg(
                pool_contamination=("contam_pool", "mean"),
                kept_contamination=("contam_kept", "mean"),
                cleaning_success_rate=("cleaning_works", "mean"),
                score_skewness=("score_skew", "mean")).round(4).reset_index()
            dump(t8, "T8_self_refinement_diagnosis")
        except Exception as e:
            logerr("paper:T8", e)

    log(f"[표] 논문용 표 저장 완료 → {T.resolve()}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*",
                    default=["S1", "S3", "S2", "S4", "S5", "S7", "S6", "S8", "S9"])
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=paths.SEEDS_HEAVY,
                    help="무거운 stage(S2/S4/S5)용 seed")
    ap.add_argument("--seeds_main", nargs="*", type=int, default=paths.SEEDS_MAIN,
                    help="주 결과표(S1)용 seed — 논문 주 결과는 0–14 (15회)")
    ap.add_argument("--msl_max", type=int, default=4)
    ap.add_argument("--alpha_pairs", type=int, default=8, help="α 곡선을 돌릴 pair 수")
    ap.add_argument("--indicator", default="rms")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--epochs_pretrain", type=int, default=120)
    ap.add_argument("--epochs_adapt", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--usad_window", type=int, default=12)
    ap.add_argument("--epochs_usad", type=int, default=100)
    ap.add_argument("--z_size", type=int, default=40)
    ap.add_argument("--tranad_window", type=int, default=10,
                    help="TranAD 원 논문 설정(n_window=10). 하위 window로 분할해 사용")
    ap.add_argument("--tranad_lr", type=float, default=1e-3)
    ap.add_argument("--epochs_tranad", type=int, default=30)
    ap.add_argument("--no_tranad", action="store_true")
    ap.add_argument("--force_pretrain", action="store_true")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT),
                    help="기존 run_all_out과 분리할 출력 폴더")
    ap.add_argument("--allow_hash_mismatch", action="store_true",
                    help="XJTU CSV 해시가 다르더라도 경고 후 계속 실행")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [0]; args.seeds_main = [0]
        args.epochs_pretrain = 3; args.epochs_adapt = 2
        args.epochs_usad = 5; args.epochs_tranad = 2
        args.alpha_pairs = 1; args.msl_max = 1

    configure_output(args.outdir)
    verify_xjtu_fixed_files(args.allow_hash_mismatch)

    for sub in ["ckpt", "scores", "figures"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    device = train_cfg.device
    t_start = time.time()
    log("=" * 74)
    log(f"XJTU-fixed 재실행 시작 | device={device} | out={OUT} | stages={args.stages} | seeds={args.seeds}")

    pairs = args.pairs or list(XJTU_FIXED_PAIRS)
    if args.smoke:
        pairs = pairs[:2]
    log(f"대상 Pair {len(pairs)}개: {pairs}")

    KEY = ["stage", "pair", "seed", "method", "contam", "alpha"]
    done, rows = load_done(RES, KEY)
    gdone, grows = load_done(GAP, ["stage", "pair"])
    log(f"재개: 기존 결과 {len(rows)}행 / 이미 완료된 셀 {len(done)}개")

    try:
        if "S1" in args.stages:
            log(f"\n### S1 주 결과표 (seed {args.seeds_main}) ###")
            _sv = args.seeds; args.seeds = args.seeds_main
            stage_main(pairs, args, device, rows, done)
            args.seeds = _sv
        if "S3" in args.stages:
            log("\n### S3 Domain gap 측정 ###"); stage_domaingap(pairs, args, device, grows, gdone)
        if "S2" in args.stages:
            log("\n### S2 오염 강건성 곡선 ###"); stage_robustness(pairs, args, device, rows, done)
        if "S4" in args.stages:
            log("\n### S4 MMD strength curve ###"); stage_alpha(pairs, args, device, rows, done)
        if "S5" in args.stages:
            log("\n### S5 발표 방법 비교 ###"); stage_baselines(pairs, args, device, rows, done)
        if "S7" in args.stages:
            log("\n### S7 자가정제 적용 판단 ###"); stage_selfrefine(pairs, args, device, rows, done)
        if "S6" in args.stages:
            log("\n### S6 Pseudo Label 지표 변형 ###"); stage_indicator(pairs, args, rows)
    except KeyboardInterrupt:
        log("사용자 중단 — 여기까지 저장됨")
    except Exception as e:
        logerr("main-loop", e)

    if "S8" in args.stages:
        log("\n### S8 그림 생성 ###"); make_figures()
    if "S9" in args.stages:
        log("\n### S9 분석 ###"); analyze()
        log("\n### S9b 논문용 표 생성 ###"); make_paper_tables()
    log(f"\n총 소요 {(time.time()-t_start)/3600:.2f} 시간")
    log("=" * 74)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
final_95_final_analysis.py — 남은 분석 전부 (GPU 불필요, 2분)
=============================================================
입력 (전부 이미 있는 파일)
  results_rescored.csv     전체 stage 재채점본
  mechanism_rows.csv       IN/OUT 분해 원자료
  leakfix_summary.json     (선택) exact 마스크 요약

출력 (outdir)
  A1_leakage_audit.csv        Table 3 원본: pair별 누수 실측 + 평가가능 판정
  A2_negative_control.csv     데이터셋별 오염 기울기 (none/exact/temporal)
  A3_split_controlled.csv     rho 5->40 만 사용한 재분석 (split 조건 동일)
  A4_dose_response.csv        용량-반응 상관 (음성 결과로 보고할 것)
  A5_threshold_metrics.csv    AUPR / F1@1%FAR (temporal)
  A6_external_baseline.csv    TranAD/USAD 와 동일 pair 비교
  A7_reliability_ci.csv       주 결과 + 부트스트랩 CI
  FINAL_REPORT.txt            콘솔 출력 전체

실행
  python final_95_final_analysis.py --results results_rescored.csv ^
      --mechanism mechanism_rows.csv --outdir tables_final
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, spearmanr
except ImportError:
    print("scipy 필요: pip install scipy --break-system-packages"); raise

OUT_LINES = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT_LINES.append(s)

METRICS = ["auroc", "auroc_nopool_exact", "auroc_nopool_temporal"]


def ds_of(p):
    p = str(p)
    if p.startswith("Boiler"): return "Boiler"
    if p.startswith("SMD"):    return "SMD"
    if p.startswith("MSL"):    return "MSL"
    return "field"


def fill_zero(d):
    """제거율 0 pair 는 temporal/exact 지표가 원 지표와 같다."""
    if "frac_removed_temporal" in d.columns:
        zero = d.groupby("pair")["frac_removed_temporal"].max()
        zero = set(zero[zero.fillna(1) <= 1e-12].index)
    else:
        zero = set()
    if "frac_removed" in d.columns:
        c = d[d.get("frac_removed_temporal", pd.Series(np.nan, index=d.index)).isna()] \
            .groupby("pair")["frac_removed"].max()
        zero |= {p for p, v in c.items() if pd.notna(v) and v <= 1e-12}
    n = 0
    for p in zero:
        for col in ["auroc_nopool_temporal", "auroc_nopool_exact"]:
            if col not in d.columns: d[col] = np.nan
            m = (d.pair == p) & d[col].isna() & d.auroc.notna()
            n += int(m.sum()); d.loc[m, col] = d.loc[m, "auroc"]
    P(f"[fill] 제거율 0 pair {len(zero)}개, {n} cell 을 원 지표로 채움")
    return d


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(x):
    x = np.asarray(x, float)
    if len(x) < 5: return np.nan
    try: return float(wilcoxon(x).pvalue)
    except ValueError: return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_rescored.csv")
    ap.add_argument("--mechanism", default="mechanism_rows.csv")
    ap.add_argument("--leakfix", default="leakfix_summary.json")
    ap.add_argument("--gap-files", nargs="*", default=[],
                    help="purge gap 민감도용 results_rescored*.csv 목록 "
                         "(예: results_rescored.csv results_rescored_g2L.csv ...)")
    ap.add_argument("--outdir", default="tables_final")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(a.results)
    P(f"[load] {a.results}: {len(d)}행, stage={d.stage.value_counts().to_dict()}")
    d = fill_zero(d)
    d["ds"] = d.pair.map(ds_of)
    s2 = d[d.stage == "S2"].copy()

    # ---------------- A1 : 누수 감사표 ----------------
    if Path(a.mechanism).exists():
        mr = pd.read_csv(a.mechanism)
        g = (mr.groupby("pair")
               .agg(n_eval=("n_eval", "max"), n_in=("n_in", "max"),
                    n_out=("n_out", "max"), pos_in=("pos_in", "max"),
                    pos_out=("pos_out", "max")).reset_index())
        g["frac_win_leaked"] = g.n_in / g.n_eval
        g["frac_pos_leaked"] = g.pos_in / (g.pos_in + g.pos_out).replace(0, np.nan)
        g["pos_kept"] = g.pos_out
        g["neg_kept"] = g.n_out - g.pos_out
        g["evaluable"] = (g.pos_kept > 0) & (g.neg_kept > 0)
        g["dataset"] = g.pair.map(ds_of)
        if Path(a.leakfix).exists():
            lf = json.load(open(a.leakfix, encoding="utf-8"))
            ex = {v["pair"]: v["frac_removed"] for v in lf["masks"].values()}
            g["frac_win_exact"] = g.pair.map(ex)
        g.to_csv(out / "A1_leakage_audit.csv", index=False, encoding="utf-8-sig")
        P("\n===== A1 누수 감사 =====")
        cols = ["pair", "dataset", "n_eval"] + \
               (["frac_win_exact"] if "frac_win_exact" in g else []) + \
               ["frac_win_leaked", "frac_pos_leaked", "pos_kept", "neg_kept", "evaluable"]
        P(g[cols].round(4).to_string(index=False))
        P(f"\n총 {len(g)} pair | 시간중복 있음 {int((g.frac_win_leaked>0).sum())} "
          f"| 평가 가능 {int(g.evaluable.sum())} | 평가 불가 {int((~g.evaluable).sum())}")
    else:
        g = pd.DataFrame(); P(f"[A1] 건너뜀 (없음: {a.mechanism})")

    # ---------------- A2 : negative control ----------------
    rows = []
    for meth in sorted(s2.method.unique()):
        for ds in ["Boiler", "SMD", "MSL", "field"]:
            sub = s2[(s2.method == meth) & (s2.ds == ds)]
            if sub.empty: continue
            for col in METRICS:
                p = sub.pivot_table(index="pair", columns="contam", values=col, aggfunc="mean")
                if 0.0 in p.columns and 0.2 in p.columns:
                    dd = (p[0.2] - p[0.0]).dropna()
                    if len(dd):
                        rows.append(dict(metric=col, method=meth, dataset=ds,
                                         n_pairs=len(dd), delta_0_to_20=dd.mean()))
    a2 = pd.DataFrame(rows)
    a2.to_csv(out / "A2_negative_control.csv", index=False, encoding="utf-8-sig")
    P("\n===== A2 negative control : 데이터셋별 오염 기울기 (0→20%) =====")
    for col in METRICS:
        t = a2[a2.metric == col].pivot_table(index="method", columns="dataset",
                                             values="delta_0_to_20")
        if not t.empty:
            P(f"\n[{col}]"); P(t.round(4).to_string())

    # ---------------- A3 : split 조건 통일 ----------------
    rows = []
    for meth in sorted(s2.method.unique()):
        for col in METRICS:
            p = s2[s2.method == meth].pivot_table(index="pair", columns="contam",
                                                  values=col, aggfunc="mean")
            if 0.05 in p.columns and 0.4 in p.columns:
                dd = (p[0.4] - p[0.05]).dropna()
                if len(dd):
                    lo, hi = boot(dd.values)
                    rows.append(dict(metric=col, method=meth, n_pairs=len(dd),
                                     delta_5_to_40=dd.mean(), ci_lo=lo, ci_hi=hi,
                                     p_wilcoxon=wil(dd.values)))
    a3 = pd.DataFrame(rows)
    a3.to_csv(out / "A3_split_controlled.csv", index=False, encoding="utf-8-sig")
    P("\n===== A3 split 조건 통일 (rho 5→40, 전 구간 동일 순열 split) =====")
    P(a3.pivot_table(index="method", columns="metric",
                     values="delta_5_to_40").round(4).to_string())

    # ---------------- A4 : 용량-반응 (음성 결과) ----------------
    rows = []
    if not g.empty:
        gi = g.set_index("pair")
        for meth in sorted(s2.method.unique()):
            p = s2[s2.method == meth].pivot_table(index="pair", columns="contam",
                                                  values="auroc", aggfunc="mean")
            if not (0.0 in p.columns and 0.2 in p.columns): continue
            dd = (p[0.2] - p[0.0]).dropna()
            for dose in ["frac_win_leaked", "frac_pos_leaked"]:
                x = gi[dose].reindex(dd.index); ok = x.notna()
                if ok.sum() >= 5:
                    r, pv = spearmanr(x[ok], dd[ok])
                    rows.append(dict(method=meth, dose=dose, n=int(ok.sum()),
                                     spearman=r, p=pv))
    a4 = pd.DataFrame(rows)
    a4.to_csv(out / "A4_dose_response.csv", index=False, encoding="utf-8-sig")
    P("\n===== A4 용량-반응 상관 (★음성 결과 — 논문에 negative result 로 보고) =====")
    if not a4.empty: P(a4.round(4).to_string(index=False))
    P("  해석: pair 간 이질성(이상 기저율 0.6~67%, window 20/100, pool 13~13,738)이 커")
    P("        cross-pair 상관은 검정력이 없다. 메커니즘 근거는 pair 내 IN/OUT 분해다.")

    # ---------------- A5 : threshold 의존 지표 ----------------
    s1 = d[d.stage == "S1"]
    cand = [c for c in ["aupr_nopool", "aupr", "f1_far0.01_nopool", "f1_far0.01",
                        "aupr_lift"] if c in d.columns]
    rows = []
    ev = set(g[g.evaluable].pair) if not g.empty else set(s1.pair)
    for col in cand:
        p = s1[s1.pair.isin(ev)].pivot_table(index="pair", columns="method",
                                             values=col, aggfunc="mean")
        if "source_only" not in p.columns: continue
        base = p["source_only"]
        for meth in p.columns:
            if meth == "source_only": continue
            dd = (p[meth] - base).dropna()
            if len(dd) == 0: continue
            lo, hi = boot(dd.values)
            rows.append(dict(metric=col, method=meth, n_pairs=len(dd),
                             mean=dd.mean(), median=dd.median(),
                             wins=int((dd > 1e-4).sum()), losses=int((dd < -1e-4).sum()),
                             worst=dd.min(), ci_lo=lo, ci_hi=hi, p=wil(dd.values)))
    a5 = pd.DataFrame(rows)
    a5.to_csv(out / "A5_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    P("\n===== A5 threshold 의존 지표 (평가가능 pair) =====")
    if not a5.empty:
        P(a5.round(4).to_string(index=False))

    # ---------------- A6 : 외부 baseline ----------------
    s5 = d[(d.stage == "S5") & d.auroc_nopool_temporal.notna()]
    if not s5.empty:
        pairs = sorted(s5.pair.unique())
        b = s5.pivot_table(index="pair", columns="method",
                           values="auroc_nopool_temporal", aggfunc="mean")
        aa = d[(d.stage == "S1") & d.pair.isin(pairs)].pivot_table(
            index="pair", columns="method", values="auroc_nopool_temporal", aggfunc="mean")
        keep = [c for c in ["source_only", "adapter_latent", "full_ft"] if c in aa.columns]
        tab = pd.concat([aa[keep], b], axis=1)
        tab.loc["MEAN"] = tab.mean()
        tab.loc["WORST"] = tab.iloc[:-1].min()
        tab.to_csv(out / "A6_external_baseline.csv", encoding="utf-8-sig")
        P(f"\n===== A6 외부 baseline 비교 (temporal, {len(pairs)} pair) =====")
        P(tab.round(4).to_string())
        P("  ※ TranAD/USAD 는 설비마다 모델 전체를 학습·저장한다.")
        P("     표에 trainable params 와 storage 를 반드시 병기할 것.")
    else:
        P("\n[A6] S5 temporal 없음 — Boiler S5 재실행 필요")

    # ---------------- A7 : 주 결과 + CI ----------------
    p = d[(d.stage == "S1") & d.pair.isin(ev)].pivot_table(
        index="pair", columns="method", values="auroc_nopool_temporal", aggfunc="mean")
    rows = []
    if "source_only" in p.columns:
        base = p["source_only"]
        for meth in p.columns:
            if meth == "source_only": continue
            dd = (p[meth] - base).dropna()
            if dd.empty: continue
            lo, hi = boot(dd.values)
            wlo, whi = boot(dd.values, seed=1)
            rows.append(dict(method=meth, n_pairs=len(dd), mean=dd.mean(),
                             median=dd.median(), ci_lo=lo, ci_hi=hi,
                             wins=int((dd > 1e-4).sum()),
                             ties=int((dd.abs() <= 1e-4).sum()),
                             losses=int((dd < -1e-4).sum()),
                             worst=dd.min(), total_loss=dd[dd < 0].sum(),
                             p=wil(dd.values)))
    a7 = pd.DataFrame(rows)
    a7.to_csv(out / "A7_reliability_ci.csv", index=False, encoding="utf-8-sig")
    P("\n===== A7 주 결과 (temporal) =====")
    if not a7.empty: P(a7.round(4).to_string(index=False))

    # ---------------- A8 : purge gap 민감도 ----------------
    if a.gap_files:
        rows = []
        for f in a.gap_files:
            if not Path(f).exists():
                P(f"[A8] 없음: {f}"); continue
            g8 = fill_zero(pd.read_csv(f))
            gm = float(g8["gap_mult"].dropna().iloc[0]) if "gap_mult" in g8.columns else np.nan
            pv = g8[g8.stage == "S1"].pivot_table(
                index="pair", columns="method",
                values="auroc_nopool_temporal", aggfunc="mean").dropna(how="all")
            if "source_only" not in pv.columns: continue
            base = pv["source_only"]
            nkeep = g8.groupby("pair")["n_keep_temporal"].max().reindex(pv.index)
            for meth in ["adapter_latent", "full_ft"]:
                if meth not in pv.columns: continue
                dd = (pv[meth] - base).dropna()
                lo, hi = boot(dd.values)
                rows.append(dict(file=Path(f).name, gap_mult=gm, method=meth,
                                 n_pairs=len(dd), n_eval_kept=int(nkeep.sum()),
                                 mean=dd.mean(), median=dd.median(),
                                 wins=int((dd > 1e-4).sum()),
                                 losses=int((dd < -1e-4).sum()),
                                 worst=dd.min(), ci_lo=lo, ci_hi=hi,
                                 p=wil(dd.values)))
            # 오염 기울기도 함께
            s28 = g8[g8.stage == "S2"]
            for meth in ["adapter_latent", "full_ft"]:
                pp = s28[s28.method == meth].pivot_table(
                    index="pair", columns="contam",
                    values="auroc_nopool_temporal", aggfunc="mean")
                if 0.0 in pp.columns and 0.2 in pp.columns:
                    ddc = (pp[0.2] - pp[0.0]).dropna()
                    if len(ddc):
                        rows.append(dict(file=Path(f).name, gap_mult=gm,
                                         method=meth + " (contam 0→20%)",
                                         n_pairs=len(ddc), mean=ddc.mean(),
                                         median=ddc.median()))
        a8 = pd.DataFrame(rows)
        a8.to_csv(out / "A8_purge_gap.csv", index=False, encoding="utf-8-sig")
        P("\n===== A8 purge gap 민감도 =====")
        if not a8.empty: P(a8.round(4).to_string(index=False))
        P("  판정: gap 을 키워도 부호·유의성이 유지되면 결론은 gap 선택에 둔감하다.")

    (out / "FINAL_REPORT.txt").write_text("\n".join(OUT_LINES), encoding="utf-8")
    P(f"\n[done] {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

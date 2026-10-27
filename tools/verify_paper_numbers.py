#!/usr/bin/env python3
"""verify_paper_numbers.py
논문(RESS 투고본 v3) 핵심 수치를 저장된 결과 CSV에서 다시 계산해 대조한다.
재학습 없음. GPU 불필요.

실행 (저장소 루트에서):
    python tools/verify_paper_numbers.py
    python tools/verify_paper_numbers.py --root .   # 다른 위치일 때

집계 규칙 (논문과 동일하게 맞추는 데 필요한 부분)
  - 주 지표: auroc_nopool_temporal (E2)
  - 제거율 0 인 pair(SMD 4, MSL 3)는 E2 = E0 이므로 E2 가 비어 있으면 auroc 로 채운다.
  - 제거율 > 0 인 pair(Boiler 6, MSL_G-6toE-1)에서 E2 가 비어 있는 셀은 채우지 않고 제외한다.
    (auroc 로 채우면 누수가 섞여 Boiler_3to1 무적응 AUROC 가 0.343 -> 0.338 로 바뀌고
     Full FT 평균이 +0.093 -> +0.096 으로 바뀐다.)
  - seed 평균 -> pair 평균 -> source_only 대비 차이 -> pair 단위 Wilcoxon
  - |gain| <= 1e-4 는 tie
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FIELD = ("PB", "XB", "Sim")
METHODS = ["adapter_input", "adapter_latent", "adapter_isca", "adapter_output", "full_ft"]
TIE = 1e-4


def load(path):
    d = pd.read_csv(path, low_memory=False)
    return d[(d.stage == "S1") & ~d.pair.str.startswith(FIELD)].copy()


def pair_gain(d, metric):
    if metric == "auroc":
        d["m"] = d["auroc"]
    else:
        frac = "frac_removed_temporal" if metric == "auroc_nopool_temporal" else "frac_removed_exact"
        zero = d.groupby("pair")[frac].transform("first").fillna(0) == 0
        d["m"] = np.where(zero & d[metric].isna(), d["auroc"], d[metric])
    pm = d.groupby(["pair", "method"])["m"].mean().unstack()
    return pm.sub(pm["source_only"], axis=0), pm["source_only"]


def summary(g):
    return dict(mean=g.mean(), median=g.median(), win=int((g > TIE).sum()),
                tie=int((g.abs() <= TIE).sum()), loss=int((g < -TIE).sum()),
                worst=g.min(), p=wilcoxon(g).pvalue, n=len(g))


CHECKS = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    CHECKS.append(ok)
    print(f"  [{'OK ' if ok else 'XX '}] {label:48s} got {got:+.4f}  paper {want:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    R = Path(a.root) / "work"
    s15 = R / "core_results_seed15"

    print("== Table 3 / A.3  누수 감사 (tables_final/A1_leakage_audit.csv)")
    au = pd.read_csv(R / "tables_final" / "A1_leakage_audit.csv")
    overlap = int((au.frac_win_leaked > 0).sum())
    unscorable = int((~au.evaluable.astype(bool)).sum())
    check("overlap pairs (16/23)", overlap, 16, 0)
    check("unscorable pairs (9)", unscorable, 9, 0)
    b = au[au.dataset == "Boiler"].frac_win_leaked
    check("Boiler removal min (20.0%)", b.min(), 0.2002, 5e-4)
    check("Boiler removal max (80.1%)", b.max(), 0.8009, 5e-4)

    print("== Table 8  ΔAUROC E2(0), 14 pairs, 15 seeds")
    g, base = pair_gain(load(s15 / "results_rescored.csv"), "auroc_nopool_temporal")
    want = {"adapter_input": (0.008, -0.0003, 0.002), "adapter_latent": (0.037, -0.022, 0.007),
            "adapter_isca": (0.031, -0.008, 0.013), "adapter_output": (0.032, -0.352, 0.153),
            "full_ft": (0.093, -0.102, 0.035)}
    for m, (mu, wo, p) in want.items():
        s = summary(g[m])
        print(f"  {m:15s} {s['mean']:+.4f} [{s['median']:+.4f}] {s['win']}/{s['tie']}/{s['loss']} worst {s['worst']:+.4f} p {s['p']:.3f}")
        check(f"{m} mean", s["mean"], mu, 6e-4)
        check(f"{m} worst", s["worst"], wo, 6e-4)
        check(f"{m} p", s["p"], p, 6e-4)
    print("  pre-adaptation AUROC < 0.5:", base[base < 0.5].round(3).to_dict())

    print("== Table 14  detection-valid pairs (no-adapt AUROC >= 0.5)")
    v = base >= 0.5
    for m, (mu, p) in {"adapter_latent": (0.024, 0.084), "full_ft": (0.056, 0.432)}.items():
        s = summary(g[m][v])
        check(f"{m} mean (n={s['n']})", s["mean"], mu, 6e-4)
        check(f"{m} p", s["p"], p, 6e-4)

    print("== Table 13  target-level (Boiler pairs sharing a target averaged)")
    tgt = g.index.map(lambda p: "Boiler_t" + p[-1] if p.startswith("Boiler") else p)
    for m, (mu, p) in {"adapter_latent": (0.036, 0.024), "full_ft": (0.096, 0.148)}.items():
        gt = g[m].groupby(tgt).mean()
        s = summary(gt)
        check(f"{m} mean (n={s['n']})", s["mean"], mu, 6e-4)
        check(f"{m} p", s["p"], p, 2e-3)

    print("== Table 7  adaptation gain by leakage definition (15 seeds)")
    d15 = load(s15 / "results_rescored.csv")
    for metric, w in [("auroc", (0.061, 0.127)), ("auroc_nopool_exact", (0.060, 0.127)),
                      ("auroc_nopool_temporal", (0.037, 0.093))]:
        gg, _ = pair_gain(d15.copy(), metric)
        check(f"{metric} latent", gg["adapter_latent"].mean(), w[0], 6e-4)
        check(f"{metric} full_ft", gg["full_ft"].mean(), w[1], 6e-4)

    print("== Table 9  purge gap (results_rescored_g2L = g=L, g5L = g=4L)")
    for f, w in [("results_rescored_g2L.csv", (0.034, 0.087)), ("results_rescored_g5L.csv", (0.028, 0.060))]:
        gg, _ = pair_gain(load(s15 / f), "auroc_nopool_temporal")
        check(f"{f} latent", gg["adapter_latent"].mean(), w[0], 6e-4)
        check(f"{f} full_ft", gg["full_ft"].mean(), w[1], 6e-4)

    print("== Table 5 / 7  contamination slope rho 0->20 (5 seeds, tables_final/A2)")
    a2 = pd.read_csv(R / "tables_final" / "A2_negative_control.csv")
    def slope(metric, method, ds):
        r = a2[(a2.metric == metric) & (a2.method == method) & (a2.dataset == ds)]
        return float(r.delta_0_to_20.iloc[0])
    for metric, method, ds, want in [
            ("auroc", "full_ft", "Boiler", -0.310), ("auroc_nopool_exact", "full_ft", "Boiler", -0.299),
            ("auroc_nopool_temporal", "full_ft", "Boiler", 0.000), ("auroc", "adapter_latent", "Boiler", -0.107),
            ("auroc_nopool_exact", "adapter_latent", "Boiler", -0.102), ("auroc_nopool_temporal", "adapter_latent", "Boiler", 0.013),
            ("auroc", "full_ft", "SMD", -0.028), ("auroc", "full_ft", "MSL", -0.007)]:
        check(f"slope {metric} {method} {ds}", slope(metric, method, ds), want, 6e-4)

    print("== Section 5.6  matched subsample (results_s2/s2_slopes.csv, Boiler 6)")
    s2 = pd.read_csv(R / "results_s2" / "s2_slopes.csv")
    bb = s2[s2.pair.str.startswith("Boiler")].groupby("method")[["slope_E0", "slope_E2", "slope_E0_matched"]].mean()
    check("latent slope E0", bb.loc["adapter_latent", "slope_E0"], -0.311, 6e-4)
    check("latent slope E2", bb.loc["adapter_latent", "slope_E2"], 0.022, 6e-4)
    check("full_ft slope E0", bb.loc["full_ft", "slope_E0"], -0.670, 6e-4)

    print("== Table 2  target-train anomaly ratio of Boiler (1.44–33.94%)")
    ds = pd.read_csv(R / "dataset_stats.csv")
    b = ds[ds.pair.str.startswith("Boiler")].point_anom_ratio_tgt_train
    check("Boiler min", b.min(), 0.0144, 1e-4)
    check("Boiler max", b.max(), 0.3394, 1e-4)

    print("== Table 3  E1 removal / leaked anomaly share (rescore_summary, A1)")
    rs = pd.read_csv(R / "rescore_summary.csv").set_index("pair")
    a1 = au.set_index("pair")
    check("Boiler_1to2 E1 (16.0%)", rs.loc["Boiler_1to2", "frac_removed_exact"], 0.160, 5e-4)
    check("Boiler_1to3 E1 (4.0%)", rs.loc["Boiler_1to3", "frac_removed_exact"], 0.040, 5e-4)
    check("MSL_G-6toE-1 E1 (1.4%)", rs.loc["MSL_G-6toE-1", "frac_removed_exact"], 0.014, 6e-4)
    check("MSL_G-6toE-1 E2 (9.5%)", rs.loc["MSL_G-6toE-1", "frac_removed_temporal"], 0.095, 6e-4)
    check("field E1 (20.0%)", rs.loc["Sim2Real", "frac_removed_exact"], 0.200, 6e-4)
    big = a1.loc[["Boiler_1to2", "Boiler_2to1", "Boiler_3to1", "Boiler_3to2"], "frac_pos_leaked"]
    check("leaked anomaly min (64.0%)", big.min(), 0.640, 6e-4)
    check("leaked anomaly max (91.2%)", big.max(), 0.912, 6e-4)
    check("Boiler_1to3 leaked anomaly (84.9%)", a1.loc["Boiler_1to3", "frac_pos_leaked"], 0.849, 6e-4)
    check("MSL_G-6toE-1 leaked anomaly (32.7%)", a1.loc["MSL_G-6toE-1", "frac_pos_leaked"], 0.327, 6e-4)

    print("== Table 4  IN/OUT, Boiler 6, rho 0 -> 40 (cell-level means, S2)")
    mm = pd.read_csv(R / "mechanism_merged.csv", low_memory=False)
    mm = mm[mm.pair.str.startswith("Boiler") & (mm.stage == "S2")]
    t4 = {("adapter_latent", "auroc_in"): (0.706, 0.550), ("full_ft", "auroc_in"): (0.804, 0.440),
          ("adapter_isca", "auroc_in"): (0.689, 0.577), ("source_only", "auroc_in"): (0.590, 0.590),
          ("adapter_latent", "auroc_out"): (0.541, 0.555), ("full_ft", "auroc_out"): (0.587, 0.562),
          ("source_only", "auroc_out"): (0.507, 0.507)}
    for (m, col), (w0, w4) in t4.items():
        x = mm[mm.method == m].groupby("contam")[col].mean()
        check(f"{m} {col} rho=0", x[0.0], w0, 6e-4)
        check(f"{m} {col} rho=40", x[0.4], w4, 6e-4)

    print("== Table 6  same split, rho 5 -> 40 (A3)")
    a3 = pd.read_csv(R / "tables_final" / "A3_split_controlled.csv").set_index(["metric", "method"])
    for (metric, m), w in {("auroc", "adapter_latent"): -0.055, ("auroc_nopool_temporal", "adapter_latent"): -0.002,
                           ("auroc_nopool_temporal", "adapter_isca"): 0.003, ("auroc_nopool_temporal", "full_ft"): -0.006}.items():
        check(f"{metric} {m}", a3.loc[(metric, m), "delta_5_to_40"], w, 6e-4)

    print("== Table 10  capacity ladder (tables_final/R3, sweep)")
    r3 = pd.read_csv(R / "tables_final" / "R3_pareto_report.csv")
    r3 = r3[r3.subset == "sweep"].set_index("params")
    for prm, (mu, wo, p) in {128: (0.011, -0.013, 0.034), 2184: (0.038, -0.024, 0.021), 8352: (0.043, -0.022, 0.027),
                             33024: (0.048, -0.021, 0.021), 852243: (0.084, -0.123, 0.092)}.items():
        r = r3.loc[prm]
        check(f"params {prm} mean", r["mean"], mu, 6e-4)
        check(f"params {prm} worst", r["worst_drop"], wo, 6e-4)
        check(f"params {prm} p", r["p_wilcoxon"], p, 6e-4)

    print("== Table 11  external reference methods (A6, 7 pairs)")
    a6 = pd.read_csv(R / "tables_final" / "A6_external_baseline.csv").set_index("pair")
    for m, (mu, wo) in {"source_only": (0.679, 0.542), "adapter_latent": (0.705, 0.542), "full_ft": (0.753, 0.531),
                        "tranad": (0.710, 0.514), "usad": (0.769, 0.501)}.items():
        check(f"{m} mean", a6.loc["MEAN", m], mu, 6e-4)
        check(f"{m} worst", a6.loc["WORST", m], wo, 6e-4)

    print("== Table 12  5-seed row (work/results_rescored.csv)")
    g5, _ = pair_gain(load(R / "results_rescored.csv"), "auroc_nopool_temporal")
    for m, (mu, wo, p) in {"adapter_latent": (0.037, -0.009, 0.004), "full_ft": (0.096, -0.104, 0.025)}.items():
        s = summary(g5[m])
        check(f"{m} mean", s["mean"], mu, 6e-4)
        check(f"{m} worst", s["worst"], wo, 6e-4)
        check(f"{m} p", s["p"], p, 6e-4)

    print("== Table 15  ranking region (results_a4_new/a4_summary.csv)")
    a4 = pd.read_csv(R / "results_a4_new" / "a4_summary.csv").set_index("metric")
    for m, (mu, p) in {"AUROC": (0.041, 0.011), "pAUROC": (-0.001, 0.808), "R-precision": (0.001, 0.937),
                       "recall@1%FAR": (-0.001, 0.263), "AUPR": (0.007, 0.068)}.items():
        check(f"{m} mean", a4.loc[m, "mean"], mu, 6e-4)
        check(f"{m} p", a4.loc[m, "p"], p, 6e-4)

    print("== Table C.1  normalization intervention (cliptest_out/C1)")
    c1 = pd.read_csv(R / "cliptest_out" / "C1_normalization_variants.csv").pivot(index="pair", columns="variant", values="auroc_tail")
    d = c1["clip"] - c1["raw"]
    check("clip - raw mean (+0.0011)", d.mean(), 0.0011, 6e-5)
    check("clip - raw max (+0.0031)", d.max(), 0.0031, 6e-5)
    check("Boiler_2to1 raw (0.2910)", c1.loc["Boiler_2to1", "raw"], 0.2910, 6e-5)

    print(f"\n{sum(CHECKS)}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()

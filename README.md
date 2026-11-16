# Adaptation–evaluation temporal leakage in cross-domain time-series anomaly detection

Code and stored results for

> M. Kim, S. Song. *Adaptation–evaluation temporal leakage in cross-domain time-series anomaly detection: Reassessing contamination robustness and adaptation strategies for fleet condition monitoring.* Submitted to *Reliability Engineering & System Safety*.

When a sliding-window anomaly detector is adapted to a target unit with a small reference pool, the pool and the evaluation windows can share time spans even though they differ as arrays (e.g. adaptation stride 5, evaluation stride 1). This repository provides

1. **`leakage_audit/`** — a standalone, NumPy-only implementation of the audit and control used in the paper (exact-match control E1, time-span control E2(g), evaluability check). It does not depend on the experiment code and can be applied to any windowed experiment.
2. **`code/`** — the full experiment and analysis pipeline.
3. **`work/`** — the stored result tables from which every number in the paper is computed, and **`tools/verify_paper_numbers.py`**, which recomputes them (121 checks).

## Main results

Leakage-controlled adaptation gain, E2(0), 14 scorable pairs x 15 seeds (Table 8 of the paper; recomputed by `tools/verify_paper_numbers.py`).

| Method | Trainable params | Mean ΔAUROC | Win / tie / loss | Worst observed loss | Wilcoxon p |
|---|---|---|---|---|---|
| Adapter (input) | 357–3,607 | +0.008 | 11 / 2 / 1 | −0.0003 | 0.002 |
| **Adapter (latent, 1%)** | 8,352 | **+0.037** | 11 / 2 / 1 | **−0.022** | 0.007 |
| Adapter + self-refine | 8,352 | +0.031 | 10 / 2 / 2 | −0.008 | 0.013 |
| Adapter (output) | 357–3,607 | +0.032 | 10 / 0 / 4 | −0.352 | 0.153 |
| Full fine-tuning | 813,593–852,243 | +0.093 | 10 / 1 / 3 | −0.102 | 0.035 |

Leakage audit (Table 3): adaptation and evaluation windows share time spans in 16 of 23 pairs; in 9 pairs no scorable windows remain after control. The 0.312 AUROC contamination loss of full fine-tuning on Boiler (E0) is not observed under time-span control (E2).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # leakage_audit tests (reproduces the Boiler audit counts)
python tools/verify_paper_numbers.py       # recompute the paper's numbers from work/ (no GPU, seconds)
```

### Using the leakage audit on your own experiment

```python
import numpy as np
from leakage_audit import window_starts, audit, span_keep

L = 100
pool = window_starts(n_train, L, stride=5)     # adaptation windows (start indices)
ev   = window_starts(n_test,  L, stride=1)     # evaluation windows on the same time axis
r = audit(pool, ev, L, y_window=y_eval, gap=0) # E1 / E2(0) removal, leaked anomaly share, evaluability
keep = span_keep(pool, ev, L, gap=L)           # E2(g) mask with purge gap g = L
print(r.to_dict())
```

`span_keep` compares time spans directly and works for any strides and window lengths. `dilation_keep` is the fast path of Algorithm 1 (evaluation stride 1, pool windows on the same start grid); both give identical masks whenever both apply.

## Repository layout

```
run.py                  runs a script from code/ with work/ as the working directory
code/paths.py           all paths and the seed convention in one place
code/                   pipeline (kept flat because of module imports)
leakage_audit/          standalone audit (Algorithm 1)
tests/                  tests for leakage_audit
tools/verify_paper_numbers.py
work/                   working directory: stored results (data/ is not included)
```

## Reproducing the experiments

Download the public datasets into `work/data/` as described in `work/data/README.md` (XJTU-SY, PHM 2012 / PRONOSTIA, the preprocessed Boiler release of Cai et al. 2021, SMD, MSL), then

```bash
python run.py preprocess_xjtu_sy.py
python run.py final_20_run.py --plan            # list the jobs
python run.py --kmp final_20_run.py --all       # experiments (--kmp: Windows OpenMP workaround)
python run.py final_35_rescore.py --gap-mult 1  # E1 / E2 rescoring (2 -> g = L, 5 -> g = 4L)
python code/clean_rows.py                       # drop smoke / duplicate S2 rows -> results_rescored*_clean.csv
python run.py final_95_final_analysis.py --results results_rescored_clean.csv   # tables A1–A8
python run.py make_paper_figures.py             # Fig. 3–7 from the stored results
```

Adaptation gains (Tables 8, 13, 14) from the 15-seed results:

```bash
python run.py a0_prepare_gains.py --results core_results_seed15/results_rescored.csv --out gains.csv --stage S1
python run.py a1_target_sensitivity.py --gains gains.csv --out a1_out      # Table 8 CI, Table 13
python run.py a2_detector_valid_subset.py --gains gains.csv --out a2_out   # Table 14
```

| Paper | Script (`code/`) | Stored result (`work/`) |
|---|---|---|
| Table 3, A.1 (leakage audit) | `final_95_final_analysis.py`, `leakage_audit/` | `tables_final/A1_leakage_audit.csv`, `rescore_summary.csv` |
| Table 4, Fig. 3 (IN/OUT) | `final_45_leakage_mechanism.py`, `clean_rows.py` | `mechanism_rows.csv` joined to `results_rescored_clean.csv` |
| Table 5, 6, 7 (slopes) | `clean_rows.py`, `final_95_final_analysis.py` | `tables_final/A2_*.csv`, `A3_*.csv` (from `results_rescored_clean.csv`) |
| Table 7–9, 12–14 (gains) | `final_35_rescore.py`, `a0`–`a2_*.py`, `seed_expansion_full.py` | `core_results_seed15/` |
| Table 10, Fig. 6 (capacity) | `final_70_sweep.py`, `final_90_reliability.py` | `tables_final/R3_pareto_report.csv` |
| Table 11 (TranAD, USAD) | `run_all_xjtu_fixed.py` (stage S5), `tranad_model.py`, `final_95_final_analysis.py` | `tables_final/A6_external_baseline.csv` |
| Table 15 (ranking region) | `run_analysis.py`, `a4_*.py` | `results_a4_new/` |
| Section 5.6 (matched subsample) | `s2_matched.py` | `results_s2/` |
| Fig. 7 (deciles) | `final_65_diagnose.py` | `diagnose_out/` |
| Fig. 3–7 (drawing) | `make_paper_figures.py` | `figures_paper/` (generated) |
| Table C.1, C.2 | `final_66_cliptest.py` | `cliptest_out/` |

### Conventions needed to reproduce the reported values

- Primary metric `auroc_nopool_temporal` = E2(0). For the seven pairs with zero removal (SMD 4, MSL 3) E2 equals E0, so missing E2 cells are filled with `auroc`; missing E2 cells of pairs with overlap are excluded, not filled.
- Seeds: S1 (main results) 0–14; S2 contamination curves, S5 external methods and the capacity sweep 0–2 (`code/paths.py`).
- `results_rescored_g2L.csv` / `_g5L.csv` correspond to purge gaps g = L and g = 4L.
- `adapter_latent` is the proposed configuration; `adapter_isca` is "Adapter + self-refine" in the paper.
- Ties: |ΔAUROC| ≤ 1e-4 (implemented in `a1`/`a2`).
- `a0_prepare_gains.py` applies the zero-removal fill rule above by default (`--no-fill-zero-overlap` disables it).
- `results_rescored.csv` concatenates every `jobs_out/final/*/results.csv`, including smoke runs and re-runs of the same S2 cells (`*_leaked`). `code/clean_rows.py` keeps one row per S2 cell (main run with E2 > re-run with E2 > main run > re-run) and drops smoke rows; all S2 numbers (Tables 4–7, Fig. 3–4) use the cleaned file. S1 numbers use `core_results_seed15/` (single source; the 5-seed row of Table 12 is its seeds 0–4).
- Raw score dumps: the E2 values of 573 of the 603 overlap-pair S1 cells were recomputed from the raw dumps with identical results. For 30 cells (Boiler_1to3, seeds 0–4) the dumps were overwritten by the later seed expansion; their stored values are kept.

## Changes

- **v9 (2026-11):** S2 duplicate/smoke rows removed before computing Tables 4–7 and Fig. 3–4 (`code/clean_rows.py`; tables of the v8 submission kept in `work/tables_final/v8_submitted/`); fill rule and tie threshold implemented in `a0`–`a2` so that the documented commands reproduce Tables 8, 13 and 14; `verify_paper_numbers.py` updated to the v9 values. Main conclusions are unchanged (e.g. full fine-tuning contamination slope on Boiler: E0 −0.312, E2 −0.002).
- **v8:** initial release with the RESS submission.

## Data availability

The piping-system data used for the sim-to-real pair were provided under a non-disclosure agreement and are not included, nor is the channel correspondence. That pair is unscorable under time-span control and does not enter the main results. All other datasets are public.

## Environment

Python 3.13, PyTorch 2.11 (CUDA 12.8) on Windows. See `requirements.txt`.

## Pre-trained models

Model checkpoints are not included because of their size. All stored results in `work/` can be verified without them, and the checkpoints are regenerated by `final_20_run.py` (backbones are cached under each job's output folder).

## Citation

```bibtex
@article{kim2026temporalleakage,
  title   = {Adaptation--evaluation temporal leakage in cross-domain time-series anomaly detection: Reassessing contamination robustness and adaptation strategies for fleet condition monitoring},
  author  = {Kim, Mingyu and Song, Seunghwan},
  journal = {Reliability Engineering \& System Safety},
  note    = {Submitted},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`).

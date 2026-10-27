# Data (not included)

Place the datasets below so that the paths used in `code/` resolve relative to `work/`.

| Dataset | Used for | Expected location |
|---|---|---|
| XJTU-SY bearing datasets (Wang et al., IEEE Trans. Reliab. 2020) | XB1 (35Hz12kN), XB2 (37.5Hz11kN) | `data/xjtu_sy/` → `python run.py preprocess_xjtu_sy.py` writes `XB1_combined.csv`, `XB2_combined.csv` |
| PHM 2012 challenge / PRONOSTIA (Nectoux et al., 2012) | PB1, PB2 | `data/phm2012/{Training_set/Learning_set, Validation_Set/Full_Test_Set, Test_set}` → `python run.py preprocess_phm2012_cross.py` |
| Boiler, preprocessed release of SASA (Cai et al., AAAI 2021), https://github.com/DMIRLAB-Group/SASA-pytorch | Boiler pairs | `data/benchmark/SASA-pytorch/datasets/Boiler/` |
| SMD (Su et al., KDD 2019), https://github.com/NetManAIOps/OmniAnomaly | SMD pairs | `data/benchmark/OmniAnomaly/ServerMachineDataset/` |
| MSL (Hundman et al., KDD 2018), https://github.com/khundman/telemanom | MSL pairs | `data/benchmark/MSL_SMAP/` (with `labeled_anomalies.csv`) → `python run.py setup_msl_smap.py` |

The piping-system data of the sim-to-real pair (`data/pipe/`) are under a non-disclosure agreement and are not distributed.
TranAD and USAD (stage S5 of `run_all_xjtu_fixed.py`, Table 11) are implemented in `code/tranad_model.py` and `code/exp6_baselines.py`; no external repository is needed.

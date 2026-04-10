# Hybrid Filter-Wrapper Feature Selection With Explainable Gradient Boosting for Lightweight IoT Intrusion Detection

## Overview

This repository provides the full reproducible pipeline for the paper:

> *Hybrid Filter-Wrapper Feature Selection With Explainable Gradient Boosting for Lightweight IoT Intrusion Detection*
> — submitted to IEEE Access.

**Authors:** S. Aljaloud, M. Alsaffar, and Z. Al-Mekhlafi, University of Ha'il, Saudi Arabia.

The pipeline transforms three public IDS benchmark datasets (CICIDS2017, BoT-IoT, ToN-IoT) into a compact, interpretable XGBoost classifier via a two-stage hybrid MI→RF feature selection, and quantifies classification performance, deployment efficiency, and dual-layer explainability (LIME + SHAP).

**Funding:** Scientific Research Deanship at University of Ha'il — Saudi Arabia, project number BA-2122.

---

## Results Summary

| Dataset | Model | Features | F1-macro | ROC-AUC | Model Size | Latency |
|---|---|---|---|---|---|---|
| CICIDS2017 | XGBoost (hybrid) | 15 of 78 | 0.9959 | 0.9998 | 1.09 MB | 1.09 µs/sample |
| BoT-IoT | XGBoost (hybrid) | 15 of 20 | 0.9783 | 1.0000 | 0.52 MB | 0.63 µs/sample |
| ToN-IoT | XGBoost (hybrid) | 15 of 40 | 0.9981 | 1.0000 | 1.07 MB | 1.01 µs/sample |

Key efficiency gains versus full-feature Random Forest on CICIDS2017: **53× smaller model**, **7.2× lower latency**.

---

## Repository Structure

```
iot-ids-hybrid-xai/
├── src/
│   ├── config.py                          # paths and shared constants
│   ├── 01_smoke_test_inputs.py            # verify raw CSV availability
│   ├── 01b_list_raw_files.py             # list raw files for debugging
│   ├── 02_preprocess_cicids2017.py        # preprocess CICIDS2017 → parquet
│   ├── 03_preprocess_bot_iot.py           # preprocess BoT-IoT → parquet
│   ├── 04_train_baselines.py              # DT and RF baselines
│   ├── 05_train_ensemble_xgb.py           # hybrid FS + XGBoost (CICIDS2017, BoT-IoT)
│   ├── 06_profile_efficiency.py           # model size and inference latency
│   ├── 07_xai_lime.py                     # LIME and feature-importance plots
│   ├── 08_make_figures_tables.py          # summary tables and comparison figures
│   ├── 09_extra_experiments.py            # ablation study and SHAP (CICIDS2017, BoT-IoT)
│   ├── 10_preprocess_ton_iot.py           # preprocess ToN-IoT → parquet
│   ├── 11_train_ton_iot.py                # hybrid FS + XGBoost + SHAP (ToN-IoT)
│   ├── 12_ton_iot_lime_and_bars.py        # LIME for ToN-IoT and updated bar charts
│   ├── 13_ton_iot_baselines.py            # DT and RF baselines for ToN-IoT
│   └── 14_unified_ablation_efficiency.py  # unified ablation + ToN-IoT efficiency
├── data/
│   ├── raw/          # place downloaded CSVs here (git-ignored)
│   ├── interim/      # intermediate outputs (git-ignored)
│   └── processed/    # parquet datasets (git-ignored)
├── outputs/
│   ├── figures/      # generated PNG figures (git-ignored)
│   ├── tables/       # generated CSV/LaTeX tables (git-ignored)
│   ├── models/       # saved joblib models (git-ignored)
│   └── logs/         # run logs (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Datasets

| Dataset | Records | Features | Classes | Source |
|---|---|---|---|---|
| CICIDS2017 | 2,830,743 | 78 | 15 | https://www.unb.ca/cic/datasets/ids-2017.html |
| BoT-IoT | 73,370,443 | 20 | 5 | https://research.unsw.edu.au/projects/bot-iot-dataset |
| ToN-IoT | 211,043 | 40 | 10 | https://research.unsw.edu.au/projects/toniot-datasets |

Place raw CSV files under the corresponding subdirectories:

```
data/raw/cicids2017/     ← all CICIDS2017 day CSVs
data/raw/bot_iot/        ← BoT-IoT CSV files
data/raw/               ← train_test_network.csv (ToN-IoT)
```

---

## Reproduction

### 1. Install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Preprocess datasets

```bash
python src/02_preprocess_cicids2017.py
python src/03_preprocess_bot_iot.py
python src/10_preprocess_ton_iot.py
```

### 3. Train baseline models

```bash
python src/04_train_baselines.py
python src/13_ton_iot_baselines.py
```

### 4. Train hybrid feature selection + XGBoost

```bash
python src/05_train_ensemble_xgb.py
python src/11_train_ton_iot.py
```

### 5. Profile efficiency

```bash
python src/06_profile_efficiency.py
python src/14_unified_ablation_efficiency.py
```

### 6. Generate explainability outputs

```bash
python src/07_xai_lime.py
python src/09_extra_experiments.py
python src/12_ton_iot_lime_and_bars.py
```

### 7. Generate all figures and tables

```bash
python src/08_make_figures_tables.py
```

All CSV tables are written to `outputs/tables/` and figures to `outputs/figures/`.

---

## Analysis Code (`src/`)

### Data Preprocessing
- `02_preprocess_cicids2017.py` — loads all CICIDS2017 day CSVs, strips whitespace, removes infinities, saves clean parquet
- `03_preprocess_bot_iot.py` — processes BoT-IoT CSVs into binary and multiclass parquets with categorical encoding
- `10_preprocess_ton_iot.py` — processes ToN-IoT network CSV into binary and multiclass parquets

### Baseline Training
- `04_train_baselines.py` — Decision Tree and Random Forest baselines on CICIDS2017 and BoT-IoT (binary and multiclass)
- `13_ton_iot_baselines.py` — Decision Tree and Random Forest baselines on ToN-IoT (binary)

### Hybrid Feature Selection and XGBoost
- `05_train_ensemble_xgb.py` — two-stage MI→RF feature selection and XGBoost training on CICIDS2017 and BoT-IoT
- `11_train_ton_iot.py` — same pipeline applied to ToN-IoT, including SHAP beeswarm output

### Efficiency Profiling
- `06_profile_efficiency.py` — measures model file size and per-sample inference latency for all saved models
- `14_unified_ablation_efficiency.py` — unified ablation study (No FS / MI-only / RF-only / Hybrid) across all three datasets; ToN-IoT DT/RF efficiency profiling

### Explainability
- `07_xai_lime.py` — LIME explanations and XGBoost feature-importance bar charts for CICIDS2017 and BoT-IoT
- `09_extra_experiments.py` — multiclass XGBoost, full ablation study, and SHAP beeswarm for CICIDS2017 and BoT-IoT
- `12_ton_iot_lime_and_bars.py` — LIME explanation for ToN-IoT and updated three-dataset comparison bar charts

### Figures and Tables
- `08_make_figures_tables.py` — aggregates all experiment outputs into final CSV/LaTeX tables and comparison PNG figures

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `lime`, `matplotlib`, `joblib`, `pyarrow`.

---

## Contact

For questions regarding the paper or code: s.aljaloud@uoh.edu.sa

```

Optional extended experiments:

```bash
# all extras
python src/09_extra_experiments.py

# or specific parts only
python src/09_extra_experiments.py a   # multiclass XGBoost
python src/09_extra_experiments.py b   # ablation study
python src/09_extra_experiments.py c   # SHAP plots
```

## Outputs

Generated artifacts are written to:

- outputs/models/: trained model files
- outputs/tables/: CSV and LaTeX-ready tables
- outputs/figures/: confusion matrices, explainability plots, comparison charts

## Reproducibility Notes

- Global random state is configured in src/config.py.
- Several scripts use dataset sampling to keep runtime and memory practical.
- For full-scale runs, ensure enough RAM and disk space for intermediate files.

## License

This project is licensed under the MIT License. See the LICENSE file.

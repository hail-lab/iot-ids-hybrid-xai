# IoT IDS — Explanation Audit Pipeline (SHAP / LIME faithfulness, stability, and adversarial drift)

Reference implementation for the paper

> **Quantifying Explanation Faithfulness and Adversarial Drift in IoT Intrusion Detection: A Reproducible Audit Pipeline**
> S. Aljaloud, M. Alsaffar, Z. Al-Mekhlafi — University of Ha'il, Saudi Arabia.
> Submitted to *MDPI Sensors*, 2026.

**Funding.** Scientific Research Deanship, University of Ha'il — project **BA-2122**.
**Contact.** `s.aljaloud@uoh.edu.sa`

---

## What this repository contains

An end-to-end, reproducible pipeline that **audits** post-hoc explanations of a compact XGBoost IoT intrusion detector on three public benchmarks (CICIDS2017, BoT-IoT, ToN-IoT):

1. Preprocesses the three datasets into a unified tabular format (76 million records total for classification).
2. Applies two-stage hybrid feature selection (mutual-information filter → Random Forest importance wrapper) to 10/15 features per dataset, validated against MI-only and RF-only ablations.
3. Trains an XGBoost classifier and two baselines (Decision Tree, Random Forest) on the same selected features.
4. Produces and **quantitatively audits** SHAP and LIME explanations on 500 stratified test instances per dataset: deletion / insertion AUC (faithfulness), Jaccard@*k* and Kendall-τ (rank stability) over 30 SHAP bootstraps / 10 LIME bootstraps, sized to yield ±0.02 deletion-AUC and ±0.03 Jaccard@5 bootstrap CIs.
5. Runs a black-box **ZOO** adversarial attack (HopSkipJump is reported as a documented negative result — it does not converge on tree ensembles).
6. Defends with a **SHAP-drift detector** that flags inputs whose top-*k* SHAP signature diverges from a benign reference distribution — reusing attributions already computed for analyst-facing explanation at negligible additional latency.
7. Generates sparse counterfactuals with **DiCE** for analyst triage.
8. Reports 5×2 cross-validated **McNemar** significance tests and 95 % bootstrap confidence intervals.
9. Reports deployment efficiency (on-disk model size, per-sample inference latency) versus RF and DT baselines.

Every numeric claim in the paper is sourced from a file under `outputs/tables/`.

---

## Headline findings

**Primary contribution.** A 30-bootstrap audit surfaces a **stable-but-unfaithful** failure mode in LIME not previously documented in the IoT IDS literature: on CICIDS2017 and ToN-IoT, LIME achieves near-perfect top-5 rank stability (Jaccard@5 = 1.00) yet is measurably **less faithful** than SHAP. The deletion-AUC gap is **0.21 points on CICIDS2017** (confirmed, well beyond the ±0.02 bootstrap CI); on ToN-IoT the corresponding gap is 0.02 points (within CI, a tendency rather than a confirmed effect). On BoT-IoT, where all three classifiers converge on a compact high-signal feature set, SHAP and LIME are comparable on both stability and faithfulness.

**Secondary contribution.** The SHAP top-*k* signature shifts predictably under black-box ZOO perturbations, yielding a drift-monitor AUROC of **0.67–0.83** against a no-monitor baseline of 0.50, with no adversarial retraining.

### Binary classification (XGBoost on hybrid-selected features)

| Dataset     | Feats. | Accuracy | Macro-F1 | ROC-AUC | Size (MB) | Latency (μs/sample) |
|-------------|:-----:|:--------:|:--------:|:-------:|:---------:|:-------------------:|
| CICIDS2017  | 15    | 0.9952   | 0.9924   | 0.9997  | 1.13      | 9.99                |
| BoT-IoT     | 10    | 1.0000   | 0.9860   | 1.0000  | 0.49      | 8.31                |
| ToN-IoT     | 15    | 0.9988   | 0.9983   | 0.99999 | 1.09      | 27.09               |

Random Forest trained on the same selected features is **4.75–17.9× larger and 2.7–8.3× slower** than XGBoost (see `outputs/tables/efficiency_metrics.csv`).

### Audit results (all values in `outputs/tables/*.csv`)

| Analysis                          | Finding                                                                                                                                    |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Hybrid vs. MI-only vs. RF-only**| All three selection strategies are statistically indistinguishable on F1 (spread ≤ 0.004 across datasets). Hybrid's value is **dual-criterion validation**, not marginal F1 gain. Selection time is 1.9–2.3× the single-stage variants. |
| **5×2 CV / McNemar**              | All pairwise differences reach α = 0.05 significance due to large test sets; RF edges XGBoost by 0.0005–0.0008 F1 on CIC/ToN-IoT (operationally negligible); XGBoost leads RF by 0.08 F1 on BoT-IoT (statistically and practically significant). |
| **Faithfulness — SHAP**           | Deletion AUC 0.72 / 0.87 / 0.75 (CIC/BoT/ToN); insertion AUC ≥ 0.95 on all three. SHAP is more faithful than LIME on every dataset.        |
| **Faithfulness — LIME**           | Deletion AUC 0.93 / 0.90 / 0.77; insertion AUC 0.90–0.95. **Stable-but-unfaithful on CIC (0.21 gap, confirmed).**                          |
| **Stability — SHAP**              | Jaccard@5 0.86 / 0.98 / 0.96; Kendall-τ 0.93 on all three (30 bootstraps).                                                                 |
| **Stability — LIME**              | Jaccard@5 1.00 / 0.88 / 1.00; Kendall-τ 0.82–0.84 (10 bootstraps).                                                                         |
| **Adversarial (ZOO, 100 inst.)**  | Attack success rate 0.30 (CIC) / 0.01 (BoT) / 1.00 (ToN). HopSkipJump omitted: documented convergence failure on XGBoost tree ensembles.  |
| **SHAP-drift detector**           | Best-*k* AUROC 0.79 (CIC, *k*=5) / 0.67 (BoT, *k*=5) / 0.83 (ToN, *k*=10). Strongest signal on ToN-IoT, which is also the most vulnerable.|
| **Counterfactuals (DiCE)**        | Validity 1.00 on generated CFs. CIC: 100/100 queries, 1.7 features changed, proximity 11.4. BoT-IoT: **only 34/100 queries produced a CF** (harder decision boundary), 5.3 features changed, proximity 48.1. ToN: 100/100, 2.1 features, proximity 7.6. |

---

## Repository layout

```text
iot-ids-hybrid-xai/
├── src/                              # 16 self-contained experiment scripts
│   ├── config.py                     # Paths, seeds, dataset registry, hyperparameters
│   ├── data_utils.py                 # Unified load_dataset() for all three benchmarks
│   ├── model_utils.py                # Hybrid MI+RF feature selection; XGBoost train/cache
│   ├── download_botiot.py            # Fetch BoT-IoT 10-best CSVs
│   ├── preprocess_botiot.py          # Build bot_iot_{binary,multiclass}.parquet
│   ├── preprocess_toniot_multiclass.py  # Build ToN-IoT multiclass parquet
│   ├── train_baselines.py            # DT, RF, XGBoost on hybrid-selected features
│   ├── faithfulness.py               # Deletion/insertion AUC + prediction gap (SHAP & LIME)
│   ├── stability.py                  # Jaccard@k + Kendall-τ across bootstraps
│   ├── adversarial.py                # ZOO black-box attacks (HSJ omitted w/ note)
│   ├── shap_drift.py                 # SHAP top-k Jaccard adversarial detector
│   ├── counterfactuals.py            # DiCE sparse counterfactuals
│   ├── cv_significance.py            # 5×2 CV + McNemar tests + bootstrap CIs
│   ├── multiclass.py                 # Multiclass XGBoost on the hybrid feature set
│   ├── ablation.py                   # MI-only vs. RF-only vs. Hybrid
│   └── efficiency.py                 # Model size on disk + per-sample latency
├── colab/
│   └── run_all.ipynb                 # End-to-end Colab notebook (CPU-only, ~45–60 min)
├── outputs/
│   ├── tables/                       # 19 CSV/JSON result files (committed)
│   ├── figures/                      # 3 faithfulness curves (committed; other figures regenerated by pipeline)
│   ├── models/                       # Cached .joblib bundles (DT/RF committed; XGB CIC & BoT excluded — see below)
│   └── logs/                         # per-run stdout captures (gitignored)
├── data/
│   ├── raw/                          # Raw dataset downloads (gitignored)
│   └── interim/                      # Preprocessed parquets (gitignored)
├── requirements.txt
├── CITATION.cff
├── LICENSE                           # MIT
└── README.md
```

### Not committed to version control

- `data/raw/*`, `data/interim/*` — ~520 MB of preprocessed data; regenerated deterministically from raw downloads.
- `outputs/models/xgb_cic.joblib` (277 MB) and `outputs/models/xgb_bot.joblib` (214 MB) — exceed GitHub's 100 MB per-file limit. **Regenerated on first run of `src/train_baselines.py`** (~20 min CPU-only) or available on request from the corresponding author. `xgb_ton.joblib` (32 MB) and all RF / DT bundles are committed.
- `outputs/logs/*` — per-run stdout captures.

---

## Reproducing the results

CPU-only, no GPU required. ~45–60 min end-to-end on a Colab free-tier instance.

### A. Google Colab (recommended for reviewers)

1. Open `colab/run_all.ipynb` in Colab.
2. Upload the raw dataset files (see *Data availability* below) or the preprocessed `*.parquet` files provided on request.
3. **Runtime → Run all**. The notebook will:
   - install pinned dependencies from `requirements.txt`,
   - preprocess BoT-IoT and ToN-IoT where needed,
   - execute the experiment scripts in order,
   - write `outputs/tables/*.csv` and `outputs/figures/*.png`,
   - zip and offer `pipeline_outputs.zip` for download.

Each script is self-contained; individual scripts can be re-run with `!python -u src/<script>.py`.

### B. Local

```bash
pip install -r requirements.txt

# acquire datasets (see "Data availability" below), then:
python src/preprocess_botiot.py
python src/preprocess_toniot_multiclass.py
python src/train_baselines.py
python src/faithfulness.py
python src/stability.py
python src/adversarial.py
python src/shap_drift.py
python src/counterfactuals.py
python src/cv_significance.py
python src/multiclass.py
python src/ablation.py
python src/efficiency.py
```

XGBoost / RF / DT bundles are cached to `outputs/models/` on first run and reused thereafter. Delete them to force retraining.

Peak memory is driven by CICIDS2017 (~6 GB during MI + RF feature ranking). The `cv_significance.py`, `efficiency.py`, and `ablation.py` scripts apply a stratified sample cap (300–500k rows) before fitting Random Forest baselines to keep Colab free-tier memory under 12 GB; raise this cap locally if you have more RAM.

---

## Data availability

| Dataset     | Source                                                                                 | Subset used                                              |
|-------------|----------------------------------------------------------------------------------------|----------------------------------------------------------|
| CICIDS2017  | Canadian Institute for Cybersecurity — <https://www.unb.ca/cic/datasets/ids-2017.html> | Flow-statistics CSVs concatenated into a single parquet  |
| BoT-IoT     | UNSW Canberra Cyber Range — <https://research.unsw.edu.au/projects/bot-iot-dataset>    | 5 % "10-best features" Training + Testing CSVs           |
| ToN-IoT     | UNSW Canberra Cyber Range — <https://research.unsw.edu.au/projects/toniot-datasets>    | Processed-Network subset                                 |

Preprocessed `*.parquet` files are excluded from version control; regenerate from raw downloads via `src/preprocess_botiot.py`, `src/preprocess_toniot_multiclass.py`, and the steps documented in `src/data_utils.py`, or request from the corresponding author.

---

## Script index (`src/`)

| Script                             | Output file(s)                                                        | Purpose |
|------------------------------------|-----------------------------------------------------------------------|---------|
| `config.py`                        | —                                                                     | Single source of truth for paths, seeds, dataset registry, hyperparameters (`XGB_PARAMS`, `RF_PARAMS`, `DT_PARAMS`, `TOP_K_FILTER = 30`, `TOP_K_MODEL = 15`, `MI_SAMPLE = 300000`). |
| `data_utils.py`                    | —                                                                     | `load_dataset(key)` for all three benchmarks. Normalises labels to {0,1} and drops ancillary columns (`label_multi`, `label_original`, `attack_cat`, …) to prevent leakage. |
| `model_utils.py`                   | `outputs/models/xgb_{key}.joblib`                                     | `hybrid_feature_selection()` and `train_or_load_xgb()` with on-disk caching. |
| `download_botiot.py`               | `data/raw/bot_iot/*.csv`                                              | Fetches BoT-IoT 10-best Training/Testing CSVs. |
| `preprocess_botiot.py`             | `data/bot_iot_{binary,multiclass}.parquet`                            | Concatenates CSVs, normalises labels, saves parquets. |
| `preprocess_toniot_multiclass.py`  | `data/ton_iot_multiclass.parquet`                                     | Builds ToN-IoT multiclass parquet from the processed-network CSVs. |
| `train_baselines.py`               | `outputs/tables/baseline_metrics.csv`                                 | DT / RF / XGBoost on the hybrid-selected features. |
| `faithfulness.py`                  | `outputs/tables/faithfulness_metrics.csv`, `outputs/figures/faithfulness_{cic,bot,ton}.png` | Deletion/insertion AUC and prediction gap over 500 stratified test instances for SHAP and LIME. |
| `stability.py`                     | `outputs/tables/stability_metrics.csv`                                | Top-*k* Jaccard overlap and Kendall-τ across 30 SHAP / 10 LIME bootstraps. |
| `adversarial.py`                   | `outputs/tables/adversarial_metrics.csv`                              | ZOO attacks (IBM ART) on 100 correctly-classified attack instances per dataset. HopSkipJump failure documented in the CSV's `note_hsj` column. |
| `shap_drift.py`                    | `outputs/tables/shap_drift_metrics.csv`                               | Top-*k* Jaccard SHAP-drift detector trained on benign reference; reports AUROC at *k* ∈ {5, 10, 15}. |
| `counterfactuals.py`               | `outputs/tables/counterfactual_metrics.csv`                           | DiCE sparse counterfactuals for 100 attack instances per dataset. |
| `cv_significance.py`               | `outputs/tables/cv_metrics.csv`, `outputs/tables/statistical_tests.csv` | Stratified 5×2 CV with XGBoost / RF / DT; McNemar tests with 95 % bootstrap CIs on F1 differences. |
| `multiclass.py`                    | `outputs/tables/multiclass_*.csv`                                     | Multiclass XGBoost on the binary-derived 15-feature set. |
| `ablation.py`                      | `outputs/tables/ablation_metrics.csv`                                 | MI-only vs. RF-only vs. Hybrid feature selection under identical XGBoost hyperparameters. |
| `efficiency.py`                    | `outputs/tables/efficiency_metrics.csv`                               | Serialises XGBoost / RF / DT to joblib and measures on-disk size + per-sample inference latency over 20 runs of 1000-sample batches. |

---

## Requirements

Pinned versions compatible with Colab Python 3.12 and the ART / DiCE / SHAP / LIME stacks are in `requirements.txt`. Key packages:

```
numpy  pandas  pyarrow  scikit-learn  scipy
xgboost  shap  lime
adversarial-robustness-toolbox  dice-ml
matplotlib  joblib  tqdm
```

---

## Paper ↔ repository cross-reference

Numbers quoted in the paper are sourced from files under `outputs/tables/`.

| Paper element                                   | File(s)                                              |
|-------------------------------------------------|------------------------------------------------------|
| Table — Binary classification (baselines)       | `baseline_metrics.csv`                               |
| Table — Ablation (MI / RF / Hybrid)             | `ablation_metrics.csv`                               |
| Table — Inference efficiency                    | `efficiency_metrics.csv`                             |
| Table — Multiclass generalisation               | `multiclass_metrics.csv`, `multiclass_per_class.csv` |
| Table — Faithfulness & stability (XAI audit)    | `faithfulness_metrics.csv`, `stability_metrics.csv`  |
| Table — Adversarial ZOO + SHAP-drift AUROC      | `adversarial_metrics.csv`, `shap_drift_metrics.csv`  |
| Table — Counterfactual explanations             | `counterfactual_metrics.csv`                         |
| Statistical testing narrative                   | `cv_metrics.csv`, `statistical_tests.csv`            |

---

## License

MIT — see `LICENSE`.

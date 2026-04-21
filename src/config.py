"""Unified project configuration.

Single source of truth for dataset registry, paths, seeds, and hyperparameters.
"""
from __future__ import annotations
from pathlib import Path

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"

OUT = ROOT / "outputs"
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
OUT_MODELS = OUT / "models"
OUT_LOGS = OUT / "logs"


def ensure_dirs() -> None:
    for p in [DATA, DATA_RAW, DATA_INTERIM, OUT_FIG, OUT_TAB, OUT_MODELS, OUT_LOGS]:
        p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Reproducibility
# -------------------------
RANDOM_STATE = 42


# -------------------------
# Dataset registry
# -------------------------
DATASETS = {
    "cic": {
        "display_name": "CICIDS2017",
        "parquet": DATA / "cicids2017_clean.parquet",
        "sample_cap": 1_900_000,
    },
    "bot": {
        "display_name": "BoT-IoT",
        "parquet": DATA / "bot_iot_binary.parquet",
        "sample_cap": 2_000_000,
    },
    "ton": {
        "display_name": "ToN-IoT",
        "parquet": DATA / "ton_iot_binary.parquet",
        "sample_cap": 2_000_000,
    },
}

DATASET_KEYS = ["cic", "bot", "ton"]


# -------------------------
# Model hyperparameters
# -------------------------
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

DT_PARAMS = dict(
    max_depth=None,
    random_state=RANDOM_STATE,
)

# -------------------------
# Feature selection
# -------------------------
TOP_K_FILTER = 30   # MI-based top features before RF ranking
TOP_K_MODEL = 15    # final feature count passed to XGB
MI_SAMPLE = 300_000 # subsample size for MI computation


def display_name(key: str) -> str:
    return DATASETS[key]["display_name"]

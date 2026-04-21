"""Unified dataset loading for CICIDS2017, BoT-IoT, and ToN-IoT."""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd

from config import DATASETS, RANDOM_STATE


def load_dataset(key: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for binary classification.

    Handles column-name variations across the three datasets and
    normalizes labels to {0=benign, 1=attack}.
    """
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {key}. Valid: {list(DATASETS)}")

    meta = DATASETS[key]
    path = meta["parquet"]
    if not path.exists():
        raise FileNotFoundError(
            f"{meta['display_name']} parquet not found at {path}. "
            f"Run the appropriate preprocessing script first."
        )

    df = pd.read_parquet(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize label column to "Label" with 0/1 values
    if "Label" in df.columns:
        # CICIDS raw: string labels like 'BENIGN' / 'DDoS'
        if df["Label"].dtype == object:
            df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    elif "label_binary" in df.columns:
        df = df.rename(columns={"label_binary": "Label"})
    elif "label" in df.columns:
        df = df.rename(columns={"label": "Label"})
    else:
        raise ValueError(
            f"{meta['display_name']}: no Label/label_binary/label column in {path}"
        )

    # Drop non-feature columns that sometimes leak in (attack type strings,
    # alternate label encodings, etc.). CICIDS2017 parquet in particular ships
    # with `label_multi` and `label_original` alongside `label_binary`; if any
    # of those survive into X they produce trivially-perfect classification
    # (label leakage).
    for drop_col in [
        "type", "attack_cat", "Attack", "attack",
        "label_multiclass", "label_multi", "label_original",
        "category", "subcategory",
    ]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # Numeric coercion for all non-label columns
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)

    # Memory cap
    cap = meta["sample_cap"]
    if len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)

    X = df.drop(columns=["Label"])
    y = df["Label"].astype(int)
    return X, y

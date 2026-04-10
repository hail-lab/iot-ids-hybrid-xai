"""Preprocess BoT-IoT CSV files into binary and multiclass parquet datasets.

Outputs:
- data/processed/bot_iot_binary.parquet
- data/processed/bot_iot_multiclass.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_RAW, DATA_PROCESSED, ensure_dirs, RANDOM_STATE

BOT_DIR = DATA_RAW / "bot_iot"
OUT_BIN = DATA_PROCESSED / "bot_iot_binary.parquet"
OUT_MULTI = DATA_PROCESSED / "bot_iot_multiclass.parquet"

# Run full dataset (1.0). If you ever need faster iteration, set e.g. 0.3
SAMPLE_FRAC_PER_FILE = 1.0

# Columns that are identifiers or can leak environment specifics
DROP_COLS = {
    "pkSeqID", "stime", "ltime", "seq",
    "saddr", "daddr", "smac", "dmac"
}

# Categorical columns to encode (if present)
CAT_COLS = ["proto", "state", "flgs"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column whitespace and trim string-like cells."""
    df.columns = [c.strip() for c in df.columns]

    # pandas 2/3/4 compatibility: include both object and string dtypes
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        # Keep as string for trimming; later we encode selected categoricals
        df[c] = df[c].astype(str).str.strip()
    return df


def safe_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Convert specified columns to numeric; invalid parsing -> NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode selected categorical columns to compact integer codes."""
    for c in CAT_COLS:
        if c in df.columns:
            # category -> codes, keep small dtype
            df[c] = df[c].astype("category").cat.codes.astype("int16")
    return df


def process_one_file(fp: Path, sample_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process one BoT-IoT CSV file and return:
    - bin_df: features + Label (0/1)
    - multi_df: features + Label (category as categorical)
    """
    df = pd.read_csv(fp, low_memory=False)
    df = standardize_columns(df)

    # Required columns for this Kaggle version
    if "attack" not in df.columns or "category" not in df.columns:
        raise ValueError(
            f"Missing required columns in {fp.name}. "
            f"Expected 'attack' and 'category'. Found: {list(df.columns)}"
        )

    # Optional sampling (kept deterministic)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)

    # Convert ports safely (mixed types common)
    df = safe_numeric(df, ["sport", "dport"])

    # Attack to numeric 0/1
    df["attack"] = pd.to_numeric(df["attack"], errors="coerce").fillna(0).astype(int)

    # Labels
    df["LabelBinary"] = (df["attack"] != 0).astype(int)

    # Multi-class label as categorical (memory-safe)
    df["LabelMulti"] = df["category"].astype(str).str.strip().astype("category")

    # Drop identifiers/leakage columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Clean inf/-inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # Encode categoricals used as features
    df = encode_categoricals(df)

    # ----- Build binary dataset -----
    bin_df = df.copy()
    bin_df = bin_df.drop(
        columns=[c for c in ["LabelMulti", "category", "subcategory", "attack"] if c in bin_df.columns],
        errors="ignore",
    )
    bin_df = bin_df.rename(columns={"LabelBinary": "Label"})

    # ----- Build multiclass dataset -----
    multi_df = df.copy()
    multi_df = multi_df.drop(
        columns=[c for c in ["LabelBinary", "attack", "category", "subcategory"] if c in multi_df.columns],
        errors="ignore",
    )
    multi_df = multi_df.rename(columns={"LabelMulti": "Label"})

    return bin_df, multi_df


def main():
    ensure_dirs()

    # Only ingest data_*.csv and exclude metadata file
    files = sorted([p for p in BOT_DIR.glob("data_*.csv") if p.name != "data_names.csv"])
    if not files:
        raise FileNotFoundError(
            f"No BoT-IoT data CSV files found in {BOT_DIR}. "
            "Expected files like data_1.csv ... data_75.csv"
        )

    bin_parts = []
    multi_parts = []

    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading: {fp.name}")
        b, m = process_one_file(fp, SAMPLE_FRAC_PER_FILE)
        bin_parts.append(b)
        multi_parts.append(m)

    # Concatenate
    bin_df = pd.concat(bin_parts, ignore_index=True)
    multi_df = pd.concat(multi_parts, ignore_index=True)

    # Drop columns entirely NaN
    bin_df = bin_df.dropna(axis=1, how="all")
    multi_df = multi_df.dropna(axis=1, how="all")

    # Ensure multiclass label stays compact (categorical)
    multi_df["Label"] = multi_df["Label"].astype("category")

    # Drop rows where all features are missing
    bin_feat_cols = [c for c in bin_df.columns if c != "Label"]
    multi_feat_cols = [c for c in multi_df.columns if c != "Label"]

    bin_df = bin_df.dropna(subset=bin_feat_cols, how="all")
    multi_df = multi_df.dropna(subset=multi_feat_cols, how="all")

    # Stats
    print("\nBoT-IoT Binary distribution:")
    print(bin_df["Label"].value_counts().head(10))

    print("\nBoT-IoT Multi-class distribution (top 15):")
    # multi_df["Label"] is categorical; value_counts works
    print(multi_df["Label"].value_counts().head(15))

    print(f"\nBinary dataset: rows={len(bin_df)} cols={len(bin_df.columns)} features={len(bin_feat_cols)}")
    print(f"Multi dataset: rows={len(multi_df)} cols={len(multi_df.columns)} features={len(multi_feat_cols)}")
    print(f"Sampling per file: {SAMPLE_FRAC_PER_FILE}")

    # Save
    bin_df.to_parquet(OUT_BIN, index=False)
    multi_df.to_parquet(OUT_MULTI, index=False)
    print(f"\nSaved: {OUT_BIN}")
    print(f"Saved: {OUT_MULTI}")


if __name__ == "__main__":
    main()
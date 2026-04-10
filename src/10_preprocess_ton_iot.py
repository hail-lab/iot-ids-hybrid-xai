"""Preprocess ToN-IoT network CSV into binary and multiclass parquet datasets.

Expected input:
    data/raw/train_test_network.csv   (211 k rows, 44 columns)

Outputs:
    data/processed/ton_iot_binary.parquet      – features + Label (0/1)
    data/processed/ton_iot_multiclass.parquet  – features + Label (attack type)
"""

import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, ensure_dirs, RANDOM_STATE

SRC_FILE = DATA_RAW / "train_test_network.csv"
OUT_BIN   = DATA_PROCESSED / "ton_iot_binary.parquet"
OUT_MULTI = DATA_PROCESSED / "ton_iot_multiclass.parquet"

# Identifier / leakage columns to drop
DROP_COLS = {"src_ip", "dst_ip"}

# High-cardinality string columns to encode as integer category codes
CAT_COLS = [
    "proto", "service", "conn_state",
    "dns_query", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer",
    "http_method", "http_uri", "http_version",
    "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].astype("category").cat.codes.astype("int16")
    return df


def main():
    ensure_dirs()

    print(f"Reading {SRC_FILE} …")
    df = pd.read_csv(SRC_FILE, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── Drop identifiers ──────────────────────────────────────────
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)

    # ── Encode categoricals ───────────────────────────────────────
    df = encode_categoricals(df)

    # ── Coerce remaining object columns to numeric ────────────────
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    label_cols = {"label", "type"}
    num_obj = [c for c in obj_cols if c not in label_cols]
    for c in num_obj:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ── Remove inf / fully-NaN feature rows ───────────────────────
    feat_cols = [c for c in df.columns if c not in label_cols]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(subset=feat_cols, how="all", inplace=True)
    print(f"  Dropped {before - len(df):,} all-NaN rows → {len(df):,} remain")

    # ── Verify label columns ──────────────────────────────────────
    assert "label" in df.columns, "Missing 'label' column"
    assert "type"  in df.columns, "Missing 'type' column"

    print("  label distribution:\n", df["label"].value_counts().to_string())
    print("  type  distribution:\n", df["type"].value_counts().to_string())

    # ── Binary dataset ────────────────────────────────────────────
    bin_df = df.drop(columns=["type"]).rename(columns={"label": "Label"})
    bin_df["Label"] = bin_df["Label"].astype(int)
    bin_df.to_parquet(OUT_BIN, index=False)
    print(f"\nSaved binary  → {OUT_BIN}  ({len(bin_df):,} rows, {len(bin_df.columns)} cols)")

    # ── Multiclass dataset ────────────────────────────────────────
    multi_df = df.drop(columns=["label"]).rename(columns={"type": "Label"})
    # Encode string attack type → integer code; store mapping
    multi_df["Label"] = multi_df["Label"].astype("category")
    mapping = dict(enumerate(multi_df["Label"].cat.categories))
    print("  Multiclass mapping:", mapping)
    multi_df["Label"] = multi_df["Label"].cat.codes.astype("int16")
    multi_df.to_parquet(OUT_MULTI, index=False)
    print(f"Saved multiclass → {OUT_MULTI}  ({len(multi_df):,} rows, {len(multi_df.columns)} cols)")


if __name__ == "__main__":
    main()

"""Download and preprocess ToN-IoT Network dataset into multiclass+binary parquets.

Uses Kaggle API to fetch TON_IoT_Train_Test_Network.csv (fadiabuzwayed/ton-iot-train-test-network).
Produces:
  - ton_iot_binary.parquet      (40 numeric features + Label)  [replaces existing]
  - ton_iot_multiclass.parquet  (same features + Label + label_multi)

Both files are written to the DATA directory defined in config.py.
Run from the project root or directly:
    python src/preprocess_toniot_multiclass.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DATA_RAW, ensure_dirs

# ---------------------------------------------------------------------------
# Download config
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "fadiabuzwayed/ton-iot-train-test-network"
CSV_NAME = "TON_IoT_Train_Test_Network.csv"

# Allow override via environment variable so local runs can target an alternate
# data directory without modifying config.py.
_DATA_OVERRIDE = os.environ.get("TON_DATA_DIR")
if _DATA_OVERRIDE:
    _DATA_DIR = Path(_DATA_OVERRIDE)
else:
    _DATA_DIR = DATA

RAW_DIR = _DATA_DIR / "raw" / "ton_iot"

OUT_BINARY = _DATA_DIR / "ton_iot_binary.parquet"
OUT_MULTI = _DATA_DIR / "ton_iot_multiclass.parquet"

# Categorical columns that need numeric encoding
CATEGORICAL_COLS = [
    "proto", "service", "conn_state",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_request_body_len", "http_response_body_len",
    "http_status_code", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]

# The 40 expected feature columns (matches existing ton_iot_binary.parquet schema)
EXPECTED_FEATURES = [
    "src_port", "dst_port", "proto", "service", "duration",
    "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "ssl_subject", "ssl_issuer",
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_request_body_len", "http_response_body_len",
    "http_status_code", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]


def download_via_kaggle(dest_dir: Path) -> Path:
    """Download TON_IoT_Train_Test_Network.csv via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")

    api = KaggleApi()
    api.authenticate()

    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / CSV_NAME
    if csv_path.exists() and csv_path.stat().st_size > 100_000:
        print(f"  {CSV_NAME} already present ({csv_path.stat().st_size / 1e6:.1f} MB)")
        return csv_path

    print(f"  downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dest_dir),
        unzip=True,
        quiet=False,
    )

    # The file might land as the original name or inside a folder
    candidates = list(dest_dir.rglob("*.csv"))
    if not candidates:
        raise RuntimeError(f"No CSV found after download in {dest_dir}")
    # Prefer exact name, otherwise take first
    for c in candidates:
        if c.name == CSV_NAME:
            return c
    return candidates[0]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode string categorical columns as integer codes (same approach as original pipeline)."""
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.lower()
            df[col] = pd.Categorical(s).codes.astype(float)
    return df


def find_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "Label", "attack", "Attack"]:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Columns: {df.columns.tolist()}")


def find_type_col(df: pd.DataFrame) -> str | None:
    for c in ["type", "Type", "attack_cat", "category", "label_multi"]:
        if c in df.columns:
            return c
    return None


def main():
    ensure_dirs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    csv_path = RAW_DIR / CSV_NAME
    if not csv_path.exists():
        print(f"CSV not found locally at {csv_path}.")
        print(f"Attempting Kaggle download ...")
        csv_path = download_via_kaggle(RAW_DIR)
    else:
        print(f"Found existing CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  raw shape: {df.shape}")
    print(f"  columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------ labels
    label_col = find_label_col(df)
    type_col = find_type_col(df)

    if type_col:
        label_multi = df[type_col].astype(str).str.strip().str.lower()
        # Normalise "normal" variants to "benign"
        label_multi = label_multi.replace({"normal": "benign", "normal ": "benign"})
        print(f"  multiclass column '{type_col}': {label_multi.value_counts().to_dict()}")
    else:
        print("  WARNING: no type/attack_cat column found — multiclass parquet will not be generated")
        label_multi = None

    # Binary label: 0=benign, 1=attack
    raw_label = df[label_col]
    if raw_label.dtype == object:
        y = (raw_label.astype(str).str.strip().str.lower() != "0").astype(int)
    else:
        y = pd.to_numeric(raw_label, errors="coerce").fillna(0).astype(int)

    print(f"  binary label dist: {y.value_counts().to_dict()}")

    # ------------------------------------------------------------------ features
    drop_always = {label_col, type_col, "ts", "src_ip", "dst_ip", "uid",
                   "attack_cat", "category", "subcategory", "label_multi"}
    drop_always.discard(None)
    feature_df = df.drop(columns=[c for c in drop_always if c in df.columns], errors="ignore")

    # Encode categoricals before numeric coercion
    feature_df = encode_categoricals(feature_df)

    # Coerce everything to numeric
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep only expected feature columns (in the exact same order as existing binary parquet)
    available = [c for c in EXPECTED_FEATURES if c in feature_df.columns]
    missing = [c for c in EXPECTED_FEATURES if c not in feature_df.columns]
    if missing:
        print(f"  WARNING: missing expected columns (will be zero-filled): {missing}")
        for c in missing:
            feature_df[c] = 0.0
    X = feature_df[EXPECTED_FEATURES].copy()

    print(f"  feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------ assemble & save
    binary_out = X.copy()
    binary_out["Label"] = y.values

    print(f"\nBinary parquet shape: {binary_out.shape}")
    print(f"Class dist: {binary_out['Label'].value_counts().to_dict()}")
    binary_out.to_parquet(OUT_BINARY, index=False)
    print(f"✓  saved binary: {OUT_BINARY}")

    if label_multi is not None:
        multi_out = X.copy()
        multi_out["Label"] = y.values
        multi_out["label_multi"] = label_multi.values
        print(f"\nMulticlass parquet shape: {multi_out.shape}")
        print(f"Class dist: {multi_out['label_multi'].value_counts().to_dict()}")
        multi_out.to_parquet(OUT_MULTI, index=False)
        print(f"✓  saved multiclass: {OUT_MULTI}")
    else:
        print("\nSkipped multiclass parquet (no type column in CSV).")


if __name__ == "__main__":
    main()

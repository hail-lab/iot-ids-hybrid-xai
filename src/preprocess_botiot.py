"""Preprocess BoT-IoT 10-best-feature CSVs into bot_iot_binary.parquet."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, DATASETS, ensure_dirs


BOT_DIR = DATA_RAW / "bot_iot"
OUT_PARQUET = DATASETS["bot"]["parquet"]

# BoT-IoT 10-best-feature schema (Koroniotis et al. 2019).
FEATURE_COLS_CANDIDATES = [
    "seq", "stddev", "N_IN_Conn_P_SrcIP", "min", "state_number",
    "mean", "N_IN_Conn_P_DstIP", "drate", "srate", "max",
]


def load_botiot_csvs() -> pd.DataFrame:
    csvs = sorted(BOT_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No BoT-IoT CSVs found in {BOT_DIR}. Run 00_download_botiot.py first."
        )
    print(f"  loading {len(csvs)} CSV(s)...")
    dfs = []
    for p in csvs:
        try:
            df = pd.read_csv(p, low_memory=False)
            print(f"    {p.name}: {len(df):,} rows, {df.shape[1]} cols")
            dfs.append(df)
        except Exception as e:
            print(f"    skipping {p.name}: {e}")
    if not dfs:
        raise RuntimeError("All BoT-IoT CSVs failed to load")
    return pd.concat(dfs, ignore_index=True)


def main():
    ensure_dirs()
    df = load_botiot_csvs()
    df.columns = [c.strip() for c in df.columns]

    # Find label column
    label_col = None
    for c in ["attack", "Attack", "label", "Label"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No label column found. Available: {df.columns.tolist()}")

    # Binary label: 1=attack, 0=normal (BoT-IoT uses 1=attack, 0=normal already)
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    # Drop non-feature columns (category, subcategory, pkSeqID, etc.)
    drop = [
        label_col, "pkSeqID", "stime", "ltime", "proto", "saddr", "sport",
        "daddr", "dport", "category", "subcategory", "flgs", "state",
    ]
    feature_df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # Keep only numeric columns
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.loc[:, feature_df.notna().mean() > 0.5]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = feature_df.copy()
    out["Label"] = y.values

    # Preserve multiclass category for later multiclass evaluation
    if "category" in df.columns:
        out["label_multi"] = df["category"].astype(str).fillna("Normal").values

    # Drop rows with all-zero features
    feat_cols = [c for c in out.columns if c not in ("Label", "label_multi")]
    keep_mask = out[feat_cols].abs().sum(axis=1) > 0
    out = out.loc[keep_mask]

    print(f"\nFinal shape: {out.shape}")
    print(f"Class balance: {out['Label'].value_counts().to_dict()}")
    if "label_multi" in out.columns:
        print(f"Multiclass distribution: {out['label_multi'].value_counts().to_dict()}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # Binary parquet (drop multiclass column to keep binary pipeline unchanged)
    binary_out = out.drop(columns=["label_multi"], errors="ignore")
    binary_out.to_parquet(OUT_PARQUET, index=False)
    print(f"✓ saved binary: {OUT_PARQUET}")
    # Multiclass parquet (separate file)
    if "label_multi" in out.columns:
        mc_path = OUT_PARQUET.parent / "bot_iot_multiclass.parquet"
        out.to_parquet(mc_path, index=False)
        print(f"✓ saved multiclass: {mc_path}")


if __name__ == "__main__":
    main()

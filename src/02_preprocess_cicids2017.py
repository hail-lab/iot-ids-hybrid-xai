import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_RAW, DATA_PROCESSED, RANDOM_STATE, ensure_dirs

CIC_DIR = DATA_RAW / "cicids2017"
OUT_FILE = DATA_PROCESSED / "cicids2017_clean.parquet"

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    # Replace inf/-inf
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop rows with missing label
    label_col = "Label" if "Label" in df.columns else None
    if label_col is None:
        raise ValueError("Could not find 'Label' column in CICIDS2017 file.")
    df = df.dropna(subset=[label_col])
    return df

def main():
    ensure_dirs()
    files = sorted(CIC_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {CIC_DIR}")

    dfs = []
    for f in files:
        print(f"Loading: {f.name}")
        df = pd.read_csv(f)
        df = clean_df(df)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    # Drop fully empty columns
    full = full.dropna(axis=1, how="all")

    print("Rows:", len(full), "Cols:", len(full.columns))
    print("Label distribution (top 10):")
    print(full["Label"].value_counts().head(10))

    full.to_parquet(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()

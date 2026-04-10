"""Profile model size and inference latency for trained models.

Output:
- outputs/tables/efficiency_profile.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, OUT_MODELS, OUT_TAB, ensure_dirs

BATCH_SIZE = 10_000
REPEATS    = 50


# ── helpers ───────────────────────────────────────────────────────

def _load_sample(data_path: Path, features: list | None, n: int) -> np.ndarray:
    """Load a small sample from a parquet file, selecting `features` if given."""
    df = pd.read_parquet(data_path)
    df.columns = [c.strip() for c in df.columns]

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    if features is not None:
        df = df[features]

    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    return df.values


def profile_one(
    model_tag: str,
    dataset_tag: str,
    model_path: Path,
    data_path: Path,
) -> dict:
    """Profile a single saved model."""
    # Model size on disk
    size_bytes = os.path.getsize(model_path)

    obj = joblib.load(model_path)

    # Baseline models are sklearn Pipelines; XGB bundles are dicts
    if isinstance(obj, dict):
        model    = obj["model"]
        features = obj["features"]
        scaler   = obj.get("scaler", None)
    else:
        # sklearn Pipeline saved by 04_train_baselines
        model    = obj
        features = None
        scaler   = None

    X = _load_sample(data_path, features, BATCH_SIZE)
    n_features = X.shape[1]

    # If there's a separate scaler in the bundle, apply it
    if scaler is not None:
        X = scaler.transform(X)

    # Warm-up
    _ = model.predict(X[:10])

    # Timed runs
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)

    batch_lat = float(np.median(times))
    per_sample_us = (batch_lat / X.shape[0]) * 1e6

    print(f"  {dataset_tag:12s} | {model_tag:8s} | "
          f"size={size_bytes/1e6:.1f}MB  feats={n_features}  "
          f"batch={batch_lat*1000:.1f}ms  per_sample={per_sample_us:.2f}µs")

    return dict(
        dataset=dataset_tag,
        model=model_tag,
        n_features=n_features,
        model_size_bytes=size_bytes,
        model_size_mb=round(size_bytes / 1e6, 2),
        batch_size=X.shape[0],
        batch_latency_s=round(batch_lat, 6),
        per_sample_us=round(per_sample_us, 3),
    )


# ── main ──────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    # Map: (model_tag, dataset_tag) → (model_path, data_path)
    jobs = [
        # Baselines – CICIDS2017
        ("DT",      "CICIDS2017", OUT_MODELS / "cicids2017_binary_dt.joblib",
         DATA_PROCESSED / "cicids2017_clean.parquet"),
        ("RF",      "CICIDS2017", OUT_MODELS / "cicids2017_binary_rf.joblib",
         DATA_PROCESSED / "cicids2017_clean.parquet"),
        # Baselines – BoT-IoT
        ("DT",      "BoT-IoT",   OUT_MODELS / "bot-iot_binary_dt.joblib",
         DATA_PROCESSED / "bot_iot_binary.parquet"),
        ("RF",      "BoT-IoT",   OUT_MODELS / "bot-iot_binary_rf.joblib",
         DATA_PROCESSED / "bot_iot_binary.parquet"),
        # XGBoost (produced by 05)
        ("XGBoost", "CICIDS2017", OUT_MODELS / "xgb_cic.joblib",
         DATA_PROCESSED / "cicids2017_clean.parquet"),
        ("XGBoost", "BoT-IoT",   OUT_MODELS / "xgb_bot.joblib",
         DATA_PROCESSED / "bot_iot_binary.parquet"),
    ]

    rows = []
    for mtag, dtag, mpath, dpath in jobs:
        if not mpath.exists():
            print(f"  SKIP {dtag} {mtag}: {mpath.name} not found (run earlier scripts first)")
            continue
        rows.append(profile_one(mtag, dtag, mpath, dpath))

    df = pd.DataFrame(rows)
    out = OUT_TAB / "efficiency_profile.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))

    print("\n✓ 06_profile_efficiency.py complete.")


if __name__ == "__main__":
    main()

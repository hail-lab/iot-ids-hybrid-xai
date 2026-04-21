"""Inference-efficiency benchmarking.

Loads the cached XGBoost bundle per dataset and trains RF + DT on the same
feature subset, then measures:
  - model size on disk (joblib dump, bytes)
  - per-sample inference latency (mean over N_RUNS runs on BATCH_SIZE inputs)
  - total parameters / estimators

Outputs:
  - outputs/tables/efficiency_metrics.csv
"""
from __future__ import annotations
import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, DT_PARAMS, OUT_TAB, RANDOM_STATE, RF_PARAMS,
                    display_name, ensure_dirs)
from model_utils import train_or_load_xgb

N_RUNS = 20
BATCH_SIZE = 1000


def model_size_bytes(model) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as f:
        joblib.dump(model, f.name)
        size = Path(f.name).stat().st_size
    Path(f.name).unlink(missing_ok=True)
    return size


def measure_latency(model, X: np.ndarray, n_runs: int = N_RUNS,
                    batch_size: int = BATCH_SIZE) -> Dict[str, float]:
    """Return mean / std per-sample latency in microseconds."""
    # Warm up
    _ = model.predict(X[:batch_size])

    rng = np.random.RandomState(RANDOM_STATE)
    per_run_us = []
    for _ in range(n_runs):
        idx = rng.choice(len(X), batch_size, replace=False)
        batch = X[idx]
        t0 = time.perf_counter()
        _ = model.predict(batch)
        dt = time.perf_counter() - t0
        per_run_us.append(dt * 1e6 / batch_size)
    return {
        "latency_us_mean": float(np.mean(per_run_us)),
        "latency_us_std": float(np.std(per_run_us)),
    }


def estimator_info(model) -> Dict:
    out = {"n_estimators": None, "max_depth": None, "n_params": None}
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            out["n_estimators"] = len(booster.get_dump())
        except Exception:
            pass
    if hasattr(model, "n_estimators"):
        out["n_estimators"] = getattr(model, "n_estimators", out["n_estimators"])
    if hasattr(model, "estimators_"):
        out["n_estimators"] = len(model.estimators_)
    if hasattr(model, "get_depth"):
        try:
            out["max_depth"] = model.get_depth()
        except Exception:
            pass
    if hasattr(model, "tree_"):
        out["n_params"] = int(model.tree_.node_count)
    return out


def evaluate(key: str) -> List[Dict]:
    print(f"\n{'=' * 60}\n{display_name(key)} — efficiency\n{'=' * 60}")
    xgb_model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)

    rows = []

    # XGBoost (cached)
    info = estimator_info(xgb_model)
    lat = measure_latency(xgb_model, X_te)
    size = model_size_bytes(xgb_model)
    rows.append({
        "dataset": display_name(key), "model": "XGBoost",
        "n_features": len(features),
        "size_bytes": size, "size_kb": size / 1024,
        "n_estimators": info["n_estimators"], "max_depth": info["max_depth"],
        **lat,
    })
    print(f"  XGBoost: size={size/1024:.1f} KB  "
          f"latency={lat['latency_us_mean']:.2f}\u00b1{lat['latency_us_std']:.2f} \u03bcs/sample")

    # RandomForest
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr, y_tr)
    info = estimator_info(rf)
    lat = measure_latency(rf, X_te)
    size = model_size_bytes(rf)
    rows.append({
        "dataset": display_name(key), "model": "RF",
        "n_features": len(features),
        "size_bytes": size, "size_kb": size / 1024,
        "n_estimators": info["n_estimators"], "max_depth": info["max_depth"],
        **lat,
    })
    print(f"  RF:      size={size/1024:.1f} KB  "
          f"latency={lat['latency_us_mean']:.2f}\u00b1{lat['latency_us_std']:.2f} \u03bcs/sample")

    # DecisionTree
    dt = DecisionTreeClassifier(**DT_PARAMS)
    dt.fit(X_tr, y_tr)
    info = estimator_info(dt)
    lat = measure_latency(dt, X_te)
    size = model_size_bytes(dt)
    rows.append({
        "dataset": display_name(key), "model": "DT",
        "n_features": len(features),
        "size_bytes": size, "size_kb": size / 1024,
        "n_estimators": 1, "max_depth": info["max_depth"],
        **lat,
    })
    print(f"  DT:      size={size/1024:.1f} KB  "
          f"latency={lat['latency_us_mean']:.2f}\u00b1{lat['latency_us_std']:.2f} \u03bcs/sample")

    return rows


def main():
    ensure_dirs()
    all_rows = []
    for key in DATASET_KEYS:
        try:
            all_rows.extend(evaluate(key))
        except FileNotFoundError as e:
            print(f"  SKIP {key}: {e}")

    df = pd.DataFrame(all_rows)
    out = OUT_TAB / "efficiency_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n\u2713 saved: {out}")
    print(df.to_string(index=False))

    (OUT_TAB / "efficiency_config.json").write_text(json.dumps({
        "N_RUNS": N_RUNS, "BATCH_SIZE": BATCH_SIZE,
    }, indent=2))


if __name__ == "__main__":
    main()

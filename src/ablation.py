"""Feature-selection ablation study.

Compares three feature-selection strategies under identical XGBoost hyperparameters
on each dataset, using the same number of final features (TOP_K_MODEL=15, or the
full feature set for BoT-IoT 10-best).

Strategies:
  - MI-only:  top-K by mutual information.
  - RF-only:  top-K by RandomForest feature importance on all features.
  - Hybrid:   MI-filter top-30 then RF-rank to top-K (= main pipeline).

Outputs:
  - outputs/tables/ablation_metrics.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, MI_SAMPLE, OUT_TAB, RANDOM_STATE,
                    TOP_K_FILTER, TOP_K_MODEL, XGB_PARAMS,
                    display_name, ensure_dirs)
from data_utils import load_dataset


def mi_rank(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    X_fill = X.fillna(0)
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y.loc[idx]
    else:
        X_mi, y_mi = X_fill, y
    mi = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False).head(k).index.tolist()


def rf_rank(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    X_fill = X.fillna(0)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fill, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False).head(k).index.tolist()


def hybrid_rank(X: pd.DataFrame, y: pd.Series, k_filter: int, k_final: int) -> List[str]:
    filt = mi_rank(X, y, k_filter)
    X_fill = X[filt].fillna(0)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fill, y)
    return pd.Series(rf.feature_importances_, index=filt).sort_values(
        ascending=False).head(k_final).index.tolist()


def fit_eval_xgb(X_tr, X_te, y_tr, y_te, features: List[str]) -> Dict:
    X_tr_f = X_tr[features].fillna(0)
    X_te_f = X_te[features].fillna(0)
    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_te_s = scaler.transform(X_te_f)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    t0 = time.time()
    model.fit(X_tr_s, y_tr)
    train_time = time.time() - t0
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    return {
        "n_features": len(features),
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else np.nan,
        "train_time_sec": train_time,
    }


def evaluate(key: str) -> List[Dict]:
    print(f"\n{'=' * 60}\n{display_name(key)} — ablation\n{'=' * 60}")
    X, y = load_dataset(key)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # k is capped at min(TOP_K_MODEL, available features)
    k = min(TOP_K_MODEL, X.shape[1])
    print(f"  using k={k} final features (dataset has {X.shape[1]} columns)")

    rows = []
    for strategy, ranker in [
        ("MI-only", lambda: mi_rank(X_tr, y_tr, k)),
        ("RF-only", lambda: rf_rank(X_tr, y_tr, k)),
        ("Hybrid (MI+RF)", lambda: hybrid_rank(X_tr, y_tr, TOP_K_FILTER, k)),
    ]:
        t0 = time.time()
        feats = ranker()
        sel_time = time.time() - t0
        metrics = fit_eval_xgb(X_tr, X_te, y_tr, y_te, feats)
        metrics.update({
            "dataset": display_name(key),
            "strategy": strategy,
            "selection_time_sec": sel_time,
            "selected_features": ";".join(feats),
        })
        print(f"  {strategy:18s}  k={metrics['n_features']:>2d}  "
              f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"auc={metrics['roc_auc']:.4f}")
        rows.append(metrics)
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
    ordered = [c for c in [
        "dataset", "strategy", "n_features", "accuracy", "precision", "recall",
        "f1", "roc_auc", "selection_time_sec", "train_time_sec", "selected_features",
    ] if c in df.columns]
    df = df[ordered]

    out = OUT_TAB / "ablation_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ saved: {out}")
    print(df.drop(columns=["selected_features"]).to_string(index=False))

    (OUT_TAB / "ablation_config.json").write_text(json.dumps({
        "TOP_K_FILTER": TOP_K_FILTER,
        "TOP_K_MODEL": TOP_K_MODEL,
        "MI_SAMPLE": MI_SAMPLE,
    }, indent=2))


if __name__ == "__main__":
    main()

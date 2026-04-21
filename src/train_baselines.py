"""Train DT, RF, and XGBoost baselines on all three datasets.

Produces:
- outputs/models/xgb_{key}.joblib, rf_{key}.joblib, dt_{key}.joblib
- outputs/tables/baseline_metrics.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, DT_PARAMS, OUT_MODELS, OUT_TAB, RF_PARAMS,
                    XGB_PARAMS, display_name, ensure_dirs)
from model_utils import train_or_load_xgb


def train_sklearn_model(model_cls, params, X_tr, y_tr, X_te, y_te):
    m = model_cls(**params)
    t0 = time.time()
    m.fit(X_tr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    y_pred = m.predict(X_te)
    pred_s = time.time() - t0
    try:
        y_proba = m.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
    except Exception:
        auc = float("nan")
    return m, {
        "accuracy": accuracy_score(y_te, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,
        "train_sec": train_s,
        "pred_sec_total": pred_s,
    }


def main():
    ensure_dirs()
    rows = []
    for key in DATASET_KEYS:
        print(f"\n{'='*60}\n{display_name(key)}\n{'='*60}")
        try:
            model_xgb, feats, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # XGBoost metrics (re-evaluate on cached test split)
        y_pred = model_xgb.predict(X_te)
        y_proba = model_xgb.predict_proba(X_te)[:, 1]
        rows.append({
            "dataset": display_name(key), "model": "XGBoost",
            "n_features": len(feats),
            "accuracy": accuracy_score(y_te, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y_te, y_proba),
        })

        # RF
        rf, m_rf = train_sklearn_model(RandomForestClassifier, RF_PARAMS, X_tr, y_tr, X_te, y_te)
        joblib.dump({"model": rf, "features": feats, "scaler": scaler}, OUT_MODELS / f"rf_{key}.joblib")
        rows.append({"dataset": display_name(key), "model": "RF", "n_features": len(feats), **{k: m_rf[k] for k in ["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro","roc_auc"]}})

        # DT
        dt, m_dt = train_sklearn_model(DecisionTreeClassifier, DT_PARAMS, X_tr, y_tr, X_te, y_te)
        joblib.dump({"model": dt, "features": feats, "scaler": scaler}, OUT_MODELS / f"dt_{key}.joblib")
        rows.append({"dataset": display_name(key), "model": "DT", "n_features": len(feats), **{k: m_dt[k] for k in ["accuracy","balanced_accuracy","precision_macro","recall_macro","f1_macro","roc_auc"]}})

    df = pd.DataFrame(rows)
    out_csv = OUT_TAB / "baseline_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

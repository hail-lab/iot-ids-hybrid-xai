"""Hybrid feature selection and model train/load utilities."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    OUT_MODELS, RANDOM_STATE, XGB_PARAMS,
    TOP_K_FILTER, TOP_K_MODEL, MI_SAMPLE,
)
from data_utils import load_dataset


def hybrid_feature_selection(
    X_train: pd.DataFrame, y_train: pd.Series,
) -> List[str]:
    """MI filter -> RF importance ranking, returning top-K_MODEL feature names."""
    X_fill = X_train.fillna(0)
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_mi, y_mi = X_fill, y_train

    print(f"  [MI] computing mutual information on {len(X_mi):,} samples...")
    mi = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi})
    top_filter = mi_df.sort_values("mi", ascending=False).head(TOP_K_FILTER)["feature"].tolist()

    print(f"  [RF] ranking {len(top_filter)} filtered features...")
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fill[top_filter], y_train)
    imp = pd.Series(rf.feature_importances_, index=top_filter)
    return imp.sort_values(ascending=False).head(TOP_K_MODEL).index.tolist()


def train_or_load_xgb(
    dataset_key: str,
) -> Tuple[xgb.XGBClassifier, List[str], StandardScaler,
           Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]]:
    """Return (model, feature_names, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)).

    Caches the full bundle (model + features + scaler + splits) in outputs/models/.
    """
    bundle_path = OUT_MODELS / f"xgb_{dataset_key}.joblib"
    if bundle_path.exists():
        print(f"  loading cached model bundle: {bundle_path.name}")
        b = joblib.load(bundle_path)
        return b["model"], b["features"], b["scaler"], (
            b["X_train_scaled"], b["X_test_scaled"], b["y_train"], b["y_test"],
        )

    print(f"  training fresh XGBoost for {dataset_key}...")
    X, y = load_dataset(dataset_key)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    features = hybrid_feature_selection(X_tr, y_tr)
    X_tr_f = X_tr[features].fillna(0)
    X_te_f = X_te[features].fillna(0)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_te_s = scaler.transform(X_te_f)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr_s, y_tr)

    bundle = {
        "model": model,
        "features": features,
        "scaler": scaler,
        "X_train_scaled": X_tr_s,
        "X_test_scaled": X_te_s,
        "y_train": y_tr,
        "y_test": y_te,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    print(f"  saved bundle: {bundle_path}")
    return model, features, scaler, (X_tr_s, X_te_s, y_tr, y_te)

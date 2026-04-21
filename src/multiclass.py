"""Multiclass XGBoost evaluation on the 15 hybrid-selected features per dataset.

Sources of multiclass labels:
- CICIDS2017: `label_multi` column in cicids2017_clean.parquet (15 classes).
- BoT-IoT:    `label_multi` column in bot_iot_multiclass.parquet (5 categories).
- ToN-IoT:    not available locally (only binary parquet); skipped with note.

Outputs:
- outputs/tables/multiclass_metrics.csv
- outputs/tables/multiclass_per_class.csv (per-class precision/recall/F1)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA, DATASET_KEYS, OUT_TAB, RANDOM_STATE, XGB_PARAMS,
                    display_name, ensure_dirs)
from model_utils import hybrid_feature_selection


MULTI_SOURCES = {
    "cic": {"parquet": DATA / "cicids2017_clean.parquet", "label_col": "label_multi"},
    "bot": {"parquet": DATA / "bot_iot_multiclass.parquet", "label_col": "label_multi"},
    "ton": {"parquet": DATA / "ton_iot_multiclass.parquet", "label_col": "label_multi"},
}


def load_multiclass(key: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    src = MULTI_SOURCES.get(key)
    if src is None or not src["parquet"].exists():
        return None
    df = pd.read_parquet(src["parquet"])
    df.columns = [c.strip() for c in df.columns]
    if src["label_col"] not in df.columns:
        return None

    y_raw = df[src["label_col"]]
    # Drop label/aux columns and keep only numeric features
    drop_cols = [c for c in df.columns if c.lower() in (
        "label", "label_binary", "label_original", "label_multi",
        "type", "attack_cat", "category", "subcategory")]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Collapse very rare classes (< 20 samples) into a single "Other" bucket to keep
    # stratified splitting viable.
    counts = y_raw.value_counts()
    rare = counts[counts < 20].index
    if len(rare) > 0:
        y_raw = y_raw.where(~y_raw.isin(rare), other="Other")

    # Cap total size for tractability
    cap = 1_500_000
    if len(X) > cap:
        idx = X.sample(n=cap, random_state=RANDOM_STATE).index
        X = X.loc[idx].reset_index(drop=True)
        y_raw = y_raw.loc[idx].reset_index(drop=True)

    return X, y_raw.astype(str)


def evaluate(key: str) -> Tuple[Optional[Dict], List[Dict]]:
    print(f"\n{'=' * 60}\n{display_name(key)} — multiclass\n{'=' * 60}")
    loaded = load_multiclass(key)
    if loaded is None:
        print(f"  SKIP: multiclass labels not available for {key}")
        return None, []
    X, y = loaded
    print(f"  samples={len(X):,} classes={y.nunique()}")
    print(f"  class distribution: {y.value_counts().to_dict()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE,
    )

    # Hybrid feature selection (reuse binary-target pipeline using y_enc!=benign as proxy)
    # For multiclass we rank features by MI w.r.t. the full multiclass target.
    # hybrid_feature_selection expects binary labels; build a binary proxy and call it.
    features = hybrid_feature_selection(X_tr, pd.Series(y_enc[:len(X_tr)] if False else
                                                         (y_tr != 0).astype(int),
                                                         index=X_tr.index))
    X_tr_f = X_tr[features].fillna(0)
    X_te_f = X_te[features].fillna(0)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_te_s = scaler.transform(X_te_f)

    n_classes = len(le.classes_)
    params = dict(XGB_PARAMS)
    params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    model = xgb.XGBClassifier(**params)

    t0 = time.time()
    model.fit(X_tr_s, y_tr)
    train_time = time.time() - t0
    y_pred = model.predict(X_te_s)
    infer_time = time.time() - t0 - train_time

    acc = accuracy_score(y_te, y_pred)
    p_macro = precision_score(y_te, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    summary = {
        "dataset": display_name(key),
        "n_classes": n_classes,
        "n_features": len(features),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }
    print(f"  accuracy={acc:.4f}  f1_macro={f1_macro:.4f}  f1_weighted={f1_weighted:.4f}")

    # Per-class detail
    report = classification_report(y_te, y_pred, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
    per_class_rows = []
    for cls_name in le.classes_:
        r = report.get(cls_name, {})
        per_class_rows.append({
            "dataset": display_name(key),
            "class": cls_name,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": r.get("support", 0),
        })
    return summary, per_class_rows


def main():
    ensure_dirs()
    summaries, per_class = [], []
    for key in DATASET_KEYS:
        s, pc = evaluate(key)
        if s is not None:
            summaries.append(s)
            per_class.extend(pc)

    df = pd.DataFrame(summaries)
    out = OUT_TAB / "multiclass_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ saved: {out}")
    print(df.to_string(index=False))

    if per_class:
        df_pc = pd.DataFrame(per_class)
        out_pc = OUT_TAB / "multiclass_per_class.csv"
        df_pc.to_csv(out_pc, index=False)
        print(f"✓ saved: {out_pc}")

    (OUT_TAB / "multiclass_config.json").write_text(json.dumps({
        "datasets": list(MULTI_SOURCES.keys()),
        "rare_class_threshold": 20,
        "cap_samples": 1_500_000,
    }, indent=2))


if __name__ == "__main__":
    main()

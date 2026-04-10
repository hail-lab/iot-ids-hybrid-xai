"""Train hybrid MI→RF→XGBoost pipeline on ToN-IoT network dataset.

Mirrors 05_train_ensemble_xgb.py / 09_extra_experiments.py, adapted for
the ToN-IoT binary and multiclass parquets produced by 10_preprocess_ton_iot.py.

Outputs:
    outputs/tables/ton_iot_xgb_metrics.csv       – binary + multiclass metrics
    outputs/tables/ton_iot_feature_selection.csv  – selected features per stage
    outputs/tables/ton_iot_ablation.csv           – ablation study
    outputs/figures/confmat_ton_iot_binary_xgb.png
    outputs/figures/xgb_feature_importance_ton.png
    outputs/figures/shap_beeswarm_ton.png
    outputs/models/xgb_ton_iot.joblib
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

import pickle
import xgboost as xgb
import joblib
import shap

from config import (
    DATA_PROCESSED, OUT_MODELS, OUT_TAB, OUT_FIG,
    RANDOM_STATE, ensure_dirs,
)

# ── Hyper-parameters (identical to 05/09) ────────────────────────
TOP_K_FILTER = 30
TOP_K_MODEL  = 15
MI_SAMPLE    = 200_000        # dataset is 211k; use most for MI

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

TON_SAMPLE = 211_000          # use full dataset (it's manageable)


# ── Utilities ─────────────────────────────────────────────────────

def plot_confusion(cm, labels, title, outpath: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    roc = pr = None
    if y_prob is not None:
        try:    roc = float(roc_auc_score(y_true, y_prob))
        except: pass
        try:    pr  = float(average_precision_score(y_true, y_prob))
        except: pass
    return dict(accuracy=float(acc), balanced_accuracy=float(bacc),
                precision_macro=float(prec), recall_macro=float(rec),
                f1_macro=float(f1), roc_auc=roc, pr_auc=pr)


# ── Data loading ──────────────────────────────────────────────────

def load_binary() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_PROCESSED / "ton_iot_binary.parquet")
    df.columns = [c.strip() for c in df.columns]
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)
    if len(df) > TON_SAMPLE:
        df = df.sample(n=TON_SAMPLE, random_state=RANDOM_STATE)
    return df.drop(columns=["Label"]), df["Label"]


def load_multiclass() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_PROCESSED / "ton_iot_multiclass.parquet")
    df.columns = [c.strip() for c in df.columns]
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)
    return df.drop(columns=["Label"]), df["Label"]


# ── Hybrid Feature Selection ──────────────────────────────────────

def hybrid_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[List[str], pd.DataFrame]:
    X_fill = X_train.fillna(0)

    # Stage 1 – Mutual Information filter
    k_filter = min(TOP_K_FILTER, X_fill.shape[1])
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_mi, y_mi = X_fill, y_train

    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_fill.columns)
    top_filter = mi_series.nlargest(k_filter).index.tolist()

    # Stage 2 – RF importance model
    k_model = min(TOP_K_MODEL, len(top_filter))
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_fill[top_filter], y_train)
    rf_series = pd.Series(rf.feature_importances_, index=top_filter)
    selected = rf_series.nlargest(k_model).index.tolist()

    log_rows = []
    for feat in selected:
        log_rows.append(dict(
            feature=feat,
            mi_score=float(mi_series.get(feat, 0.0)),
            rf_importance=float(rf_series[feat]),
        ))
    log_df = pd.DataFrame(log_rows)

    print(f"  FS: {X_train.shape[1]} → {k_filter} (MI) → {len(selected)} (RF)")
    return selected, log_df


# ── Main training routine ─────────────────────────────────────────

def train_binary(results: list, fs_logs: list):
    print("\n=== ToN-IoT: Binary classification ===")
    X, y = load_binary()
    print(f"  Loaded {len(X):,} rows × {X.shape[1]} features")
    print(f"  Class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Hybrid FS
    selected, log_df = hybrid_feature_selection(X_train, y_train)
    log_df["dataset"] = "ton_iot"
    fs_logs.append(log_df)

    # Scale
    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_train[selected].fillna(0))
    X_te_s = scaler.transform(X_test[selected].fillna(0))

    # Train XGBoost
    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
    model.fit(X_tr_s, y_train)
    train_sec = time.perf_counter() - t0

    # Evaluate
    t0 = time.perf_counter()
    y_pred = model.predict(X_te_s)
    pred_sec = time.perf_counter() - t0
    y_prob = model.predict_proba(X_te_s)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    latency_us = (pred_sec / len(y_test)) * 1e6
    model_mb   = Path(OUT_MODELS / "xgb_ton_iot.joblib").__sizeof__()  # approx pre-save

    print(f"  F1-macro={metrics['f1_macro']:.4f}  AUC={metrics['roc_auc']:.4f}  "
          f"Latency={latency_us:.3f} µs/sample  Train={train_sec:.1f}s")

    # Save model + scaler
    ensure_dirs()
    joblib.dump({"model": model, "scaler": scaler, "features": selected},
                OUT_MODELS / "xgb_ton_iot.joblib")

    # File size (now that it's saved)
    size_mb = (OUT_MODELS / "xgb_ton_iot.joblib").stat().st_size / 1e6

    results.append(dict(
        dataset="ToN-IoT", task="binary", model="XGBoost (Hybrid)",
        n_features=len(selected), **metrics,
        latency_us=latency_us, model_size_mb=round(size_mb, 3),
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, ["Normal", "Attack"], "XGBoost – ToN-IoT Binary",
                   OUT_FIG / "confmat_ton_iot_binary_xgb.png")

    # Feature importance bar chart
    importances = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.plot(kind="barh", ax=ax)
    ax.set_title("XGBoost Feature Importance – ToN-IoT")
    ax.set_xlabel("Gain importance")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "xgb_feature_importance_ton.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # SHAP beeswarm
    print("  Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_te_s), size=min(2000, len(X_te_s)), replace=False)
    shap_values = explainer.shap_values(X_te_s[shap_sample_idx])
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values, X_te_s[shap_sample_idx],
                      feature_names=selected, show=False, plot_size=None)
    plt.title("SHAP Beeswarm – ToN-IoT")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "shap_beeswarm_ton.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    return model, scaler, selected, X_train, X_test, y_train, y_test


def train_multiclass(results: list, selected: list, scaler):
    print("\n=== ToN-IoT: Multiclass classification ===")
    X, y = load_multiclass()
    print(f"  Loaded {len(X):,} rows × {X.shape[1]} features, {y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Use same features selected during binary training
    avail = [f for f in selected if f in X_train.columns]
    X_tr_s = scaler.transform(X_train[avail].fillna(0))
    X_te_s = scaler.transform(X_test[avail].fillna(0))

    params = {**XGB_PARAMS, "objective": "multi:softprob",
              "num_class": int(y.nunique()), "eval_metric": "mlogloss"}
    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(X_tr_s, y_train)

    y_pred = model.predict(X_te_s)
    metrics = compute_metrics(y_test, y_pred)
    print(f"  F1-macro={metrics['f1_macro']:.4f}  Acc={metrics['accuracy']:.4f}")

    results.append(dict(
        dataset="ToN-IoT", task="multiclass", model="XGBoost (Hybrid)",
        n_features=len(avail), **metrics,
        latency_us=None, model_size_mb=None,
    ))


def ablation_study(results: list):
    print("\n=== ToN-IoT: Ablation study ===")
    X, y = load_binary()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_fill = X_train.fillna(0)
    X_te_fill = X_test.fillna(0)

    # MI scores (full pass)
    mi_scores = mutual_info_classif(
        X_fill.sample(n=min(MI_SAMPLE, len(X_fill)), random_state=RANDOM_STATE),
        y_train.loc[X_fill.sample(n=min(MI_SAMPLE, len(X_fill)), random_state=RANDOM_STATE).index],
        random_state=RANDOM_STATE,
    )
    mi_series = pd.Series(mi_scores, index=X_fill.columns)

    # RF importances (full)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_fill, y_train)
    rf_series = pd.Series(rf.feature_importances_, index=X_fill.columns)

    strategies = {
        "No FS (all)":      X_fill.columns.tolist(),
        "MI only":          mi_series.nlargest(TOP_K_MODEL).index.tolist(),
        "RF only":          rf_series.nlargest(TOP_K_MODEL).index.tolist(),
        "Hybrid MI+RF":     rf_series[mi_series.nlargest(TOP_K_FILTER).index].nlargest(TOP_K_MODEL).index.tolist(),
    }

    rows = []
    for name, feats in strategies.items():
        scaler = StandardScaler(with_mean=False)
        Xtr = scaler.fit_transform(X_fill[feats])
        Xte = scaler.transform(X_te_fill[feats].fillna(0))
        m = xgb.XGBClassifier(**XGB_PARAMS, use_label_encoder=False)
        m.fit(Xtr, y_train)
        y_pred = m.predict(Xte)
        met = compute_metrics(y_test, y_pred)
        size_mb = len(pickle.dumps(m)) / 1e6
        rows.append(dict(dataset="ToN-IoT", strategy=name, n_features=len(feats),
                         accuracy=met["accuracy"], f1_macro=met["f1_macro"],
                         model_size_mb=round(size_mb, 3)))
        print(f"  {name:20s}  feats={len(feats):2d}  F1={met['f1_macro']:.4f}  size={size_mb:.3f} MB")

    pd.DataFrame(rows).to_csv(OUT_TAB / "ton_iot_ablation.csv", index=False)
    print(f"  Saved → {OUT_TAB / 'ton_iot_ablation.csv'}")


# ── Entry point ───────────────────────────────────────────────────

def main():
    ensure_dirs()
    results = []
    fs_logs = []

    model, scaler, selected, *_ = train_binary(results, fs_logs)
    train_multiclass(results, selected, scaler)
    ablation_study(results)

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUT_TAB / "ton_iot_xgb_metrics.csv", index=False)
    print(f"\nSaved metrics → {OUT_TAB / 'ton_iot_xgb_metrics.csv'}")

    # Save feature selection log
    if fs_logs:
        pd.concat(fs_logs, ignore_index=True).to_csv(
            OUT_TAB / "ton_iot_feature_selection.csv", index=False
        )

    print("\nAll ToN-IoT experiments complete.")
    print(metrics_df[["dataset", "task", "n_features", "f1_macro",
                       "roc_auc", "latency_us", "model_size_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()

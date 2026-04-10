"""Train binary XGBoost models with hybrid feature selection.

Outputs:
- outputs/tables/xgb_metrics_cic.csv
- outputs/tables/xgb_metrics_bot.csv
- outputs/tables/feature_selection_log.csv
- outputs/figures/confmat_*_xgb.png
- outputs/models/xgb_cic.joblib
- outputs/models/xgb_bot.joblib
"""

from __future__ import annotations

import json
import time
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

import xgboost as xgb
import joblib

from config import (
    DATA_PROCESSED, OUT_MODELS, OUT_TAB, OUT_FIG,
    RANDOM_STATE, ensure_dirs,
)

# ── Hyper-parameters ──────────────────────────────────────────────
TOP_K_FILTER = 30          # features kept after MI filter
TOP_K_MODEL  = 15          # features kept after RF importance

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

# Memory-safe sampling (same philosophy as 04_train_baselines)
CIC_SAMPLE  = 1_900_000   # total rows to keep before split
BOT_SAMPLE  = 2_000_000   # total rows to keep before split
MI_SAMPLE   = 300_000     # rows used for mutual-info (speed)


# ── Utilities ─────────────────────────────────────────────────────

def plot_confusion(cm, labels, title, outpath):
    """Save a confusion-matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
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
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Return dict of metrics (same keys as baseline_metrics.csv)."""
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    roc = pr = None
    if y_prob is not None:
        try:    roc = float(roc_auc_score(y_true, y_prob))
        except: pass
        try:    pr = float(average_precision_score(y_true, y_prob))
        except: pass
    return dict(
        accuracy=float(acc),
        balanced_accuracy=float(bacc),
        precision_macro=float(prec),
        recall_macro=float(rec),
        f1_macro=float(f1),
        roc_auc=roc,
        pr_auc=pr,
    )


# ── Data loading (binary only, memory-safe) ──────────────────────

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for binary classification."""
    if name == "cic":
        df = pd.read_parquet(DATA_PROCESSED / "cicids2017_clean.parquet")
        df.columns = [c.strip() for c in df.columns]
        df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
        cap = CIC_SAMPLE
    else:
        df = pd.read_parquet(DATA_PROCESSED / "bot_iot_binary.parquet")
        cap = BOT_SAMPLE

    # Numeric coercion + inf removal
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)

    # Memory-safe cap
    if len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)

    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y


# ── Hybrid Feature Selection ─────────────────────────────────────

def hybrid_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset: str,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stage 1 – Mutual Information (filter)  → TOP_K_FILTER features
    Stage 2 – RF Feature Importance (model) → TOP_K_MODEL features

    Returns (selected_features, log_df).
    """
    X_fill = X_train.fillna(0)

    # Sub-sample for MI speed
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_mi, y_mi = X_fill, y_train

    print(f"  [MI] Computing mutual information on {len(X_mi)} samples, {X_mi.shape[1]} features …")
    mi = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)

    mi_df = pd.DataFrame({"feature": X_train.columns, "mi_score": mi})
    mi_df.sort_values("mi_score", ascending=False, inplace=True)
    top_filter = mi_df.head(TOP_K_FILTER)["feature"].tolist()

    print(f"  [MI] Top {TOP_K_FILTER} features selected.")

    # Stage 2 – RF importance
    print(f"  [RF] Fitting RF on filtered features …")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_fill[top_filter], y_train)

    imp = pd.Series(rf.feature_importances_, index=top_filter)
    top_model = imp.sort_values(ascending=False).head(TOP_K_MODEL).index.tolist()

    print(f"  [RF] Top {TOP_K_MODEL} features selected.")

    # Build log
    log_rows = []
    for feat in mi_df["feature"]:
        row = {"dataset": dataset, "feature": feat}
        row["mi_score"]  = mi_df.loc[mi_df["feature"] == feat, "mi_score"].values[0]
        row["mi_rank"]   = (mi_df["feature"].tolist()).index(feat) + 1
        row["passed_filter"] = feat in top_filter
        row["rf_importance"] = float(imp[feat]) if feat in top_filter else None
        row["selected_final"] = feat in top_model
        log_rows.append(row)

    log_df = pd.DataFrame(log_rows)
    return top_model, log_df


# ── XGBoost Training ─────────────────────────────────────────────

def run_pipeline(dataset: str) -> Tuple[dict, pd.DataFrame]:
    """Train XGB with hybrid FS for one dataset. Return (metrics_row, fs_log)."""
    print(f"\n{'='*50}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'='*50}")

    X, y = load_dataset(dataset)
    n_total_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Feature selection ──
    features, fs_log = hybrid_feature_selection(X_train, y_train, dataset)

    # Reduce to selected features + fill NaN
    Xtr = X_train[features].fillna(0)
    Xte = X_test[features].fillna(0)

    # Scale (consistent with baseline pipeline)
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # ── Train XGBoost ──
    print(f"  [XGB] Training with {len(features)} features …")
    model = xgb.XGBClassifier(**XGB_PARAMS)

    t0 = time.time()
    model.fit(Xtr_s, y_train)
    fit_s = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(Xte_s)
    pred_s = time.time() - t1

    y_prob = None
    try:
        y_prob = model.predict_proba(Xte_s)[:, 1]
    except Exception:
        pass

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    ds_tag = "cicids2017" if dataset == "cic" else "bot-iot"
    plot_confusion(
        cm, ["Normal(0)", "Attack(1)"],
        f"{ds_tag.upper()} | BINARY | XGBoost",
        OUT_FIG / f"confmat_{ds_tag}_binary_xgb.png",
    )

    # ── Save model bundle ──
    bundle = {"model": model, "features": features, "scaler": scaler}
    joblib.dump(bundle, OUT_MODELS / f"xgb_{dataset}.joblib")

    # ── Metrics row (same schema as baseline_metrics.csv) ──
    row = dict(
        dataset="CICIDS2017" if dataset == "cic" else "BoT-IoT",
        task="binary",
        model="XGBoost",
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(features),
        n_total_features=n_total_features,
        fit_seconds=fit_s,
        pred_seconds=pred_s,
        **metrics,
    )

    # Save per-dataset CSV
    row_df = pd.DataFrame([row])
    row_df.to_csv(OUT_TAB / f"xgb_metrics_{dataset}.csv", index=False)
    print(f"  Saved: xgb_metrics_{dataset}.csv")
    print(f"  F1-macro={metrics['f1_macro']:.4f}  ROC-AUC={metrics['roc_auc']}")

    return row, fs_log


# ── Main ──────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    all_logs = []
    all_rows = []

    for ds in ["cic", "bot"]:
        row, fs_log = run_pipeline(ds)
        all_rows.append(row)
        all_logs.append(fs_log)

    # Save combined feature-selection log
    fs_full = pd.concat(all_logs, ignore_index=True)
    fs_full.to_csv(OUT_TAB / "feature_selection_log.csv", index=False)
    print(f"\nSaved: feature_selection_log.csv ({len(fs_full)} rows)")

    # Save combined XGB metrics
    pd.DataFrame(all_rows).to_csv(OUT_TAB / "xgb_metrics_combined.csv", index=False)
    print("Saved: xgb_metrics_combined.csv")

    print("\n✓ 05_train_ensemble_xgb.py complete.")


if __name__ == "__main__":
    main()

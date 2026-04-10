"""Train Decision Tree and Random Forest baselines.

Tasks:
- Binary classification on CICIDS2017 and BoT-IoT
- Multiclass classification on CICIDS2017 and BoT-IoT

Outputs:
- outputs/tables/baseline_metrics.csv
- outputs/figures/confmat_*.png
- outputs/models/*.joblib
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib

from config import DATA_PROCESSED, OUT_FIG, OUT_TAB, OUT_MODELS, ensure_dirs, RANDOM_STATE


# -------------------------
# Config knobs (16GB-safe)
# -------------------------

# CICIDS: we can train on a sample for speed but still evaluate well
CIC_TRAIN_MAX_ROWS = 1_500_000      # cap training rows to keep runtime sane
CIC_TEST_ROWS = 400_000            # fixed test size

# BoT-IoT: keep all normals (tiny) and sample attacks for training
BOT_TRAIN_ATTACK_MULTIPLIER = 50    # train attacks = normals * multiplier
BOT_TEST_SIZE = 1_000_000           # natural test size cap

# RF size (baseline-friendly; keep light for speed)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_N_JOBS = -1

# DT baseline
DT_MAX_DEPTH = None


@dataclass
class RunResult:
    dataset: str
    task: str           # "binary" or "multiclass"
    model: str          # "DT" or "RF"
    n_train: int
    n_test: int
    n_features: int
    fit_seconds: float
    pred_seconds: float
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: float | None
    pr_auc: float | None


# -------------------------
# Utilities
# -------------------------

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, outpath: Path) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def compute_metrics_binary(y_true, y_prob, y_pred) -> Tuple[float, float, float, float, float, float | None, float | None]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    roc = None
    pr = None
    # AUC metrics need probability estimates
    if y_prob is not None:
        try:
            roc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc = None
        try:
            pr = average_precision_score(y_true, y_prob)
        except Exception:
            pr = None
    return acc, bacc, prec, rec, f1, roc, pr


def compute_metrics_multiclass(y_true, y_pred) -> Tuple[float, float, float, float, float]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, bacc, prec, rec, f1


def get_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "Label" not in df.columns:
        raise ValueError("Expected column 'Label' not found.")
    X = df.drop(columns=["Label"])
    y = df["Label"]
    return X, y


def make_dt() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=DT_MAX_DEPTH
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS
    )


def fit_and_eval(
    dataset_name: str,
    task: str,
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    conf_labels: List[str],
    conf_fig_name: str,
    model_out_name: str
) -> RunResult:

    n_features = X_train.shape[1]

    # A very light pipeline: scale numeric features (mostly numeric already).
    # Trees don't require scaling, but scaling doesn't hurt and provides a consistent pipeline.
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", model)
    ])

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0

    t1 = time.time()
    y_pred = pipe.predict(X_test)
    pred_s = time.time() - t1

    # Probabilities for binary where available
    y_prob = None
    if task == "binary":
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

        acc, bacc, prec, rec, f1, roc, pr = compute_metrics_binary(y_test, y_prob, y_pred)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plot_confusion(
            cm=cm,
            labels=conf_labels,
            title=f"{dataset_name} | {task.upper()} | {model_name}",
            outpath=OUT_FIG / conf_fig_name
        )

    else:
        acc, bacc, prec, rec, f1 = compute_metrics_multiclass(y_test, y_pred)
        roc, pr = None, None

        # confusion matrix (top labels only)
        cm = confusion_matrix(y_test, y_pred, labels=conf_labels)
        plot_confusion(
            cm=cm,
            labels=[str(x) for x in conf_labels],
            title=f"{dataset_name} | {task.upper()} | {model_name}",
            outpath=OUT_FIG / conf_fig_name
        )

    # save model
    joblib.dump(pipe, OUT_MODELS / model_out_name)

    return RunResult(
        dataset=dataset_name,
        task=task,
        model=model_name,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=n_features,
        fit_seconds=fit_s,
        pred_seconds=pred_s,
        accuracy=float(acc),
        balanced_accuracy=float(bacc),
        precision_macro=float(prec),
        recall_macro=float(rec),
        f1_macro=float(f1),
        roc_auc=None if roc is None else float(roc),
        pr_auc=None if pr is None else float(pr),
    )


# -------------------------
# Loaders
# -------------------------

def load_cic(task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns train/test frames with Label column.
    task="binary": Label is 0/1 (BENIGN vs ATTACK)
    task="multiclass": Label is original string label
    """
    fp = DATA_PROCESSED / "cicids2017_clean.parquet"
    df = pd.read_parquet(fp)

    # Standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows with missing label
    df = df.dropna(subset=["Label"])

    if task == "binary":
        df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    else:
        df["Label"] = df["Label"].astype(str)

    # Convert all features to numeric where possible
    X_cols = [c for c in df.columns if c != "Label"]
    for c in X_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Minimal cleaning: drop rows where all features missing
    df = df.dropna(subset=X_cols, how="all")

    # Create fixed test set size (stratified if possible)
    # First, downsample to a workable size for training (keeps speed)
    if len(df) > (CIC_TRAIN_MAX_ROWS + CIC_TEST_ROWS):
        # sample for speed but keep label distribution
        df_small, _ = train_test_split(
            df,
            train_size=(CIC_TRAIN_MAX_ROWS + CIC_TEST_ROWS),
            stratify=df["Label"],
            random_state=RANDOM_STATE
        )
        df = df_small

    train_df, test_df = train_test_split(
        df,
        test_size=min(CIC_TEST_ROWS, int(0.2 * len(df))),
        stratify=df["Label"],
        random_state=RANDOM_STATE
    )

    return train_df, test_df


def load_bot(task: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    BoT-IoT:
    - binary: loads bot_iot_binary.parquet
    - multiclass: loads bot_iot_multiclass.parquet

    We construct a manageable train set:
    - keep all normals (Label=0) for training
    - sample attacks (Label=1) at BOT_TRAIN_ATTACK_MULTIPLIER * n_normals
    Test set:
    - random sample of BOT_TEST_SIZE rows from the full dataset (natural distribution)
    """
    if task == "binary":
        fp = DATA_PROCESSED / "bot_iot_binary.parquet"
    else:
        fp = DATA_PROCESSED / "bot_iot_multiclass.parquet"

    df = pd.read_parquet(fp)

    # Ensure numeric features where possible
    X_cols = [c for c in df.columns if c != "Label"]
    for c in X_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=X_cols, how="all")

    # Build test set (natural distribution, capped)
    if len(df) > BOT_TEST_SIZE:
        test_df = df.sample(n=BOT_TEST_SIZE, random_state=RANDOM_STATE)
    else:
        test_df = df.sample(frac=0.2, random_state=RANDOM_STATE)

    # Remove test rows from training pool (avoid leakage)
    train_pool = df.drop(index=test_df.index)

    if task == "binary":
        # keep ALL normals, sample attacks
        normals = train_pool[train_pool["Label"] == 0]
        attacks = train_pool[train_pool["Label"] == 1]

        n_norm = len(normals)
        if n_norm == 0:
            raise ValueError("BoT-IoT: no normal samples found in training pool. Cannot train properly.")

        n_attack = min(len(attacks), n_norm * BOT_TRAIN_ATTACK_MULTIPLIER)

        attacks_s = attacks.sample(n=n_attack, random_state=RANDOM_STATE)
        train_df = pd.concat([normals, attacks_s], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    else:
        # multiclass: keep all "Normal" class (if present) and sample others
        # Note: Label is categorical in your parquet; convert to string for stratify and safety
        train_pool["Label"] = train_pool["Label"].astype(str)
        test_df["Label"] = test_df["Label"].astype(str)

        normals = train_pool[train_pool["Label"].str.lower() == "normal"]
        others = train_pool[train_pool["Label"].str.lower() != "normal"]

        n_norm = len(normals)
        if n_norm == 0:
            # still proceed: sample a manageable size stratified
            train_df = train_pool.sample(n=min(2_000_000, len(train_pool)), random_state=RANDOM_STATE)
            return train_df, test_df

        # sample others relative to normals
        n_other = min(len(others), n_norm * BOT_TRAIN_ATTACK_MULTIPLIER)
        others_s = others.sample(n=n_other, random_state=RANDOM_STATE)

        train_df = pd.concat([normals, others_s], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    return train_df, test_df


# -------------------------
# Main
# -------------------------

def main():
    ensure_dirs()
    results: List[RunResult] = []

    runs = [
        ("CICIDS2017", "binary"),
        ("CICIDS2017", "multiclass"),
        ("BoT-IoT", "binary"),
        ("BoT-IoT", "multiclass"),
    ]

    for dataset_name, task in runs:
        print(f"\n=== {dataset_name} | {task.upper()} ===")

        if dataset_name == "CICIDS2017":
            train_df, test_df = load_cic(task)
        else:
            train_df, test_df = load_bot(task)

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        print(f"Train: {X_train.shape}  Test: {X_test.shape}")

        # Confusion labels
        if task == "binary":
            conf_labels = ["Normal(0)", "Attack(1)"]
        else:
            # Use top 8 classes from test set for readability
            top_classes = y_test.value_counts().head(8).index.tolist()
            conf_labels = top_classes

        # ---- DT ----
        results.append(
            fit_and_eval(
                dataset_name=dataset_name,
                task=task,
                model_name="DT",
                model=make_dt(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                conf_labels=conf_labels if task == "binary" else conf_labels,
                conf_fig_name=f"confmat_{dataset_name.lower()}_{task}_dt.png",
                model_out_name=f"{dataset_name.lower()}_{task}_dt.joblib"
            )
        )

        # ---- RF ----
        results.append(
            fit_and_eval(
                dataset_name=dataset_name,
                task=task,
                model_name="RF",
                model=make_rf(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                conf_labels=conf_labels if task == "binary" else conf_labels,
                conf_fig_name=f"confmat_{dataset_name.lower()}_{task}_rf.png",
                model_out_name=f"{dataset_name.lower()}_{task}_rf.joblib"
            )
        )

    # Save metrics table
    out_rows = [r.__dict__ for r in results]
    metrics_df = pd.DataFrame(out_rows)
    out_csv = OUT_TAB / "baseline_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics: {out_csv}")

    # Also save run config for reproducibility
    cfg = {
        "CIC_TRAIN_MAX_ROWS": CIC_TRAIN_MAX_ROWS,
        "CIC_TEST_ROWS": CIC_TEST_ROWS,
        "BOT_TRAIN_ATTACK_MULTIPLIER": BOT_TRAIN_ATTACK_MULTIPLIER,
        "BOT_TEST_SIZE": BOT_TEST_SIZE,
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
        "RF_MAX_DEPTH": RF_MAX_DEPTH,
        "DT_MAX_DEPTH": DT_MAX_DEPTH,
        "RANDOM_STATE": RANDOM_STATE
    }
    cfg_path = OUT_TAB / "baseline_run_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved run config: {cfg_path}")


if __name__ == "__main__":
    main()

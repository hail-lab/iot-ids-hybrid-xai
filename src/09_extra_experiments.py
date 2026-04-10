"""Run optional extended experiments.

Includes:
- Multiclass XGBoost with hybrid feature selection
- Ablation study (no FS, MI-only, RF-only, hybrid)
- SHAP beeswarm and bar summaries
"""

from __future__ import annotations

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
    confusion_matrix,
)

import xgboost as xgb
import joblib

from config import (
    DATA_PROCESSED, OUT_MODELS, OUT_TAB, OUT_FIG,
    RANDOM_STATE, ensure_dirs,
)

# ── Parameters (same as 05) ──────────────────────────────────────
TOP_K_FILTER = 30
TOP_K_MODEL  = 15
MI_SAMPLE    = 300_000

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

CIC_SAMPLE = 1_900_000
BOT_SAMPLE = 2_000_000
# Smaller sample for ablation (avoids hours-long training on full feature set)
ABLATION_CIC_SAMPLE = 500_000
ABLATION_BOT_SAMPLE = 500_000


# ── Utilities ─────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return dict(accuracy=acc, balanced_accuracy=bacc,
                precision_macro=prec, recall_macro=rec, f1_macro=f1)


def plot_confusion(cm, labels, title, outpath):
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
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Data loaders ──────────────────────────────────────────────────

def load_binary(name):
    if name == "cic":
        df = pd.read_parquet(DATA_PROCESSED / "cicids2017_clean.parquet")
        df.columns = [c.strip() for c in df.columns]
        df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
        cap = CIC_SAMPLE
    else:
        df = pd.read_parquet(DATA_PROCESSED / "bot_iot_binary.parquet")
        cap = BOT_SAMPLE
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)
    return df.drop(columns=["Label"]), df["Label"]


def load_multiclass(name):
    if name == "cic":
        df = pd.read_parquet(DATA_PROCESSED / "cicids2017_clean.parquet")
        df.columns = [c.strip() for c in df.columns]
        # Keep original string labels for multiclass
        cap = CIC_SAMPLE
    else:
        df = pd.read_parquet(DATA_PROCESSED / "bot_iot_multiclass.parquet")
        cap = BOT_SAMPLE
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)
    # Encode labels to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["Label"].astype(str))
    return df.drop(columns=["Label"]), pd.Series(y, index=df.index), le


# ── Feature selection variants ────────────────────────────────────

def select_mi_only(X_train, y_train, k=TOP_K_MODEL):
    X_fill = X_train.fillna(0)
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_mi, y_mi = X_fill, y_train
    mi = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi})
    return mi_df.nlargest(k, "mi")["feature"].tolist()


def select_rf_only(X_train, y_train, k=TOP_K_MODEL):
    X_fill = X_train.fillna(0)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fill, y_train)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    return imp.nlargest(k).index.tolist()


def select_hybrid(X_train, y_train, k1=TOP_K_FILTER, k2=TOP_K_MODEL):
    X_fill = X_train.fillna(0)
    if len(X_fill) > MI_SAMPLE:
        idx = X_fill.sample(n=MI_SAMPLE, random_state=RANDOM_STATE).index
        X_mi, y_mi = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_mi, y_mi = X_fill, y_train
    mi = mutual_info_classif(X_mi, y_mi, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi})
    top_filter = mi_df.nlargest(k1, "mi")["feature"].tolist()
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fill[top_filter], y_train)
    imp = pd.Series(rf.feature_importances_, index=top_filter)
    return imp.nlargest(k2).index.tolist()


# ══════════════════════════════════════════════════════════════════
# A) MULTICLASS XGBOOST
# ══════════════════════════════════════════════════════════════════

def run_multiclass():
    print("\n" + "=" * 60)
    print("  PART A: Multiclass XGBoost with Hybrid Feature Selection")
    print("=" * 60)

    rows = []
    for ds in ["cic", "bot"]:
        tag = "CICIDS2017" if ds == "cic" else "BoT-IoT"
        print(f"\n--- {tag} multiclass ---")

        X, y, le = load_multiclass(ds)
        n_classes = len(le.classes_)
        print(f"  Classes ({n_classes}): {list(le.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        # Hybrid feature selection (same as binary)
        features = select_hybrid(X_train, y_train)
        print(f"  Selected {len(features)} features")

        Xtr = X_train[features].fillna(0)
        Xte = X_test[features].fillna(0)
        scaler = StandardScaler(with_mean=False)
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        params = XGB_PARAMS.copy()
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params.pop("eval_metric", None)
        params["eval_metric"] = "mlogloss"

        model = xgb.XGBClassifier(**params)
        model.fit(Xtr_s, y_train)
        y_pred = model.predict(Xte_s)

        m = compute_metrics(y_test, y_pred)
        print(f"  F1-macro={m['f1_macro']:.4f}  Acc={m['accuracy']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        ds_tag = "cicids2017" if ds == "cic" else "bot-iot"
        plot_confusion(cm, [str(c) for c in le.classes_],
                       f"{tag} | Multiclass | XGBoost",
                       OUT_FIG / f"confmat_{ds_tag}_multiclass_xgb.png")

        rows.append(dict(dataset=tag, task="multiclass", model="XGBoost",
                         n_features=len(features), **m))

        # Save model
        bundle = {"model": model, "features": features, "scaler": scaler,
                  "label_encoder": le}
        joblib.dump(bundle, OUT_MODELS / f"xgb_{ds}_multiclass.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "xgb_multiclass_metrics.csv", index=False)
    print(f"\nSaved: xgb_multiclass_metrics.csv")
    return df


# ══════════════════════════════════════════════════════════════════
# B) ABLATION STUDY
# ══════════════════════════════════════════════════════════════════

def run_ablation():
    print("\n" + "=" * 60)
    print("  PART B: Ablation Study (Feature Selection Variants)")
    print("=" * 60)

    rows = []
    for ds in ["cic", "bot"]:
        tag = "CICIDS2017" if ds == "cic" else "BoT-IoT"
        print(f"\n--- {tag} ablation ---")

        X, y = load_binary(ds)

        # Use a smaller subsample so all 4 ablation variants finish quickly
        cap = ABLATION_CIC_SAMPLE if ds == "cic" else ABLATION_BOT_SAMPLE
        if len(X) > cap:
            X = X.sample(n=cap, random_state=RANDOM_STATE)
            y = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        variants = {
            "No FS (All)": list(X_train.columns),
            "MI Only (15)": select_mi_only(X_train, y_train, TOP_K_MODEL),
            "RF Only (15)": select_rf_only(X_train, y_train, TOP_K_MODEL),
            "Hybrid MI+RF (15)": select_hybrid(X_train, y_train),
        }

        for vname, feats in variants.items():
            print(f"  {vname}: {len(feats)} features")
            Xtr = X_train[feats].fillna(0)
            Xte = X_test[feats].fillna(0)
            scaler = StandardScaler(with_mean=False)
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            model = xgb.XGBClassifier(**XGB_PARAMS)
            t0 = time.time()
            model.fit(Xtr_s, y_train)
            fit_s = time.time() - t0

            y_pred = model.predict(Xte_s)
            m = compute_metrics(y_test, y_pred)

            # Model size
            tmp_path = OUT_MODELS / "_tmp_ablation.joblib"
            joblib.dump(model, tmp_path)
            size_mb = tmp_path.stat().st_size / (1024 * 1024)
            tmp_path.unlink()

            print(f"    F1={m['f1_macro']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"Size={size_mb:.2f}MB  Time={fit_s:.1f}s")

            rows.append(dict(dataset=tag, variant=vname,
                             n_features=len(feats),
                             size_mb=round(size_mb, 3),
                             train_time_s=round(fit_s, 2),
                             **m))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "ablation_study.csv", index=False)
    print(f"\nSaved: ablation_study.csv")
    return df


# ══════════════════════════════════════════════════════════════════
# C) SHAP BEESWARM PLOTS
# ══════════════════════════════════════════════════════════════════

def run_shap():
    print("\n" + "=" * 60)
    print("  PART C: SHAP Beeswarm Plots")
    print("=" * 60)

    try:
        import shap
    except ImportError:
        print("  ERROR: shap not installed. Run: pip install shap")
        return

    for ds in ["cic", "bot"]:
        tag = "CICIDS2017" if ds == "cic" else "BoT-IoT"
        print(f"\n--- {tag} SHAP ---")

        bundle = joblib.load(OUT_MODELS / f"xgb_{ds}.joblib")
        model = bundle["model"]
        features = bundle["features"]
        scaler = bundle["scaler"]

        # Load and prepare data
        X, y = load_binary(ds)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        Xte = X_test[features].fillna(0)
        Xte_s = scaler.transform(Xte)

        # Use a sample for SHAP (speed)
        n_shap = min(1000, len(Xte_s))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(Xte_s), n_shap, replace=False)
        X_shap = Xte_s[idx]

        # TreeExplainer (fast for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # Create DataFrame with feature names for SHAP plots
        X_shap_df = pd.DataFrame(X_shap, columns=features)

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_shap_df, show=False, max_display=15)
        plt.title(f"SHAP Feature Importance — {tag}", fontsize=12)
        plt.tight_layout()
        outpath = OUT_FIG / f"shap_beeswarm_{ds}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath.name}")

        # Bar summary plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_shap_df, plot_type="bar",
                          show=False, max_display=15)
        plt.title(f"SHAP Mean |SHAP| — {tag}", fontsize=12)
        plt.tight_layout()
        outpath = OUT_FIG / f"shap_bar_{ds}.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath.name}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    import sys
    parts = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    if "all" in parts or "a" in parts:
        run_multiclass()

    if "all" in parts or "b" in parts:
        ablation_df = run_ablation()

    if "all" in parts or "c" in parts:
        run_shap()

    print("\n" + "=" * 60)
    print("  ALL EXTRA EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

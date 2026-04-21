"""XAI faithfulness: deletion/insertion AUC for SHAP and LIME.

Fixes vs v1:
- AUC normalized to [0,1] by dividing the trapezoidal integral by the step count
- LIME attributions now computed alongside SHAP
- Runs on all three datasets uniformly

Outputs:
- outputs/tables/faithfulness_metrics.csv (long format: one row per dataset/method)
- outputs/figures/faithfulness_{key}.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, OUT_FIG, OUT_TAB, RANDOM_STATE,
                    display_name, ensure_dirs)
from model_utils import train_or_load_xgb

N_SAMPLES = 500
LIME_N_SAMPLES = 100    # LIME is slow; use fewer samples
N_STEPS = 15
TOP_K_MAX = 15


def shap_attributions(model, X: np.ndarray, background_size: int = 100) -> np.ndarray:
    bg_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), min(background_size, len(X)), replace=False)
    explainer = shap.TreeExplainer(model, data=X[bg_idx],
                                    feature_perturbation="interventional")
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.asarray(vals)


def lime_attributions(model, X_train: np.ndarray, X_eval: np.ndarray,
                       feature_names: List[str]) -> np.ndarray:
    """Return mean-|coef| LIME attributions per sample in X_eval."""
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["benign", "attack"],
        mode="classification", discretize_continuous=False,
        random_state=RANDOM_STATE,
    )
    n_features = X_eval.shape[1]
    attributions = np.zeros((len(X_eval), n_features))
    for i in range(len(X_eval)):
        try:
            exp = explainer.explain_instance(
                X_eval[i], model.predict_proba,
                num_features=n_features, num_samples=500,
            )
            # exp.as_map()[1] -> list of (feature_idx, weight) for class 1
            for fidx, w in exp.as_map().get(1, []):
                attributions[i, fidx] = w
        except Exception:
            pass
    return attributions


def deletion_auc_norm(model, X: np.ndarray, attributions: np.ndarray,
                       y: np.ndarray, n_steps: int = N_STEPS) -> Tuple[float, List[float]]:
    """Deletion AUC normalized to [0,1]. Lower is more faithful."""
    importance = np.mean(np.abs(attributions), axis=0)
    ranked = np.argsort(importance)[::-1]
    scores = []
    for step in range(n_steps + 1):
        k = int(step / n_steps * min(TOP_K_MAX, X.shape[1]))
        Xp = X.copy()
        if k > 0:
            Xp[:, ranked[:k]] = 0
        scores.append(roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
    # Normalize: trapezoid over x in [0,1], divided by (n_steps) because scores in [0,1]
    x = np.linspace(0, 1, n_steps + 1)
    auc = float(np.trapezoid(scores, x))  # already in [0,1]
    return auc, scores


def insertion_auc_norm(model, X: np.ndarray, attributions: np.ndarray,
                        y: np.ndarray, n_steps: int = N_STEPS) -> Tuple[float, List[float]]:
    """Insertion AUC normalized to [0,1]. Higher is more faithful."""
    importance = np.mean(np.abs(attributions), axis=0)
    ranked = np.argsort(importance)[::-1]
    baseline = np.mean(X, axis=0)
    scores = []
    for step in range(n_steps + 1):
        k = int(step / n_steps * min(TOP_K_MAX, X.shape[1]))
        Xp = np.tile(baseline, (len(X), 1))
        if k > 0:
            Xp[:, ranked[:k]] = X[:, ranked[:k]]
        scores.append(roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
    x = np.linspace(0, 1, n_steps + 1)
    auc = float(np.trapezoid(scores, x))
    return auc, scores


def prediction_gap(model, X: np.ndarray, attributions: np.ndarray, top_k: int = 5) -> float:
    importance = np.mean(np.abs(attributions), axis=0)
    topk = np.argsort(importance)[::-1][:top_k]
    baseline = np.mean(X, axis=0)
    X_topk = np.tile(baseline, (len(X), 1))
    X_topk[:, topk] = X[:, topk]
    p_full = model.predict_proba(X)[:, 1]
    p_topk = model.predict_proba(X_topk)[:, 1]
    return float(np.mean(np.abs(p_full - p_topk)))


def evaluate(key: str) -> List[Dict]:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
    y_te_arr = y_te.values if hasattr(y_te, "values") else np.asarray(y_te)

    rng = np.random.RandomState(RANDOM_STATE)
    # Stratified sampling: guarantee both classes present (critical for BoT-IoT
    # where benign is ~0.01% and random sampling often yields a single-class set).
    pos_idx = np.where(y_te_arr == 1)[0]
    neg_idx = np.where(y_te_arr == 0)[0]
    n_total = min(N_SAMPLES, len(X_te))
    n_neg = min(len(neg_idx), max(n_total // 2, 1))
    n_pos = n_total - n_neg
    sel_neg = rng.choice(neg_idx, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    sel_pos = rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
    idx = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(idx)
    X_eval = X_te[idx]
    y_eval = y_te_arr[idx]
    print(f"  eval set: {len(X_eval)} samples ({(y_eval==1).sum()} pos / {(y_eval==0).sum()} neg)")

    rows = []

    # --- SHAP ---
    print(f"  computing SHAP attributions on {len(X_eval)} samples...")
    shap_vals = shap_attributions(model, X_eval)
    del_shap, del_curve_s = deletion_auc_norm(model, X_eval, shap_vals, y_eval)
    ins_shap, ins_curve_s = insertion_auc_norm(model, X_eval, shap_vals, y_eval)
    gap_shap = prediction_gap(model, X_eval, shap_vals)
    print(f"    SHAP: del={del_shap:.4f} ins={ins_shap:.4f} gap={gap_shap:.4f}")
    rows.append({
        "dataset": display_name(key), "method": "SHAP",
        "n_samples": len(X_eval), "n_features": len(features),
        "deletion_auc": del_shap, "insertion_auc": ins_shap, "prediction_gap": gap_shap,
    })

    # --- LIME ---
    n_lime = min(LIME_N_SAMPLES, len(X_eval))
    print(f"  computing LIME attributions on {n_lime} samples (slower)...")
    lime_eval_idx = rng.choice(len(X_eval), n_lime, replace=False)
    X_lime = X_eval[lime_eval_idx]
    y_lime = y_eval[lime_eval_idx]
    lime_vals = lime_attributions(model, X_tr, X_lime, features)
    del_lime, del_curve_l = deletion_auc_norm(model, X_lime, lime_vals, y_lime)
    ins_lime, ins_curve_l = insertion_auc_norm(model, X_lime, lime_vals, y_lime)
    gap_lime = prediction_gap(model, X_lime, lime_vals)
    print(f"    LIME: del={del_lime:.4f} ins={ins_lime:.4f} gap={gap_lime:.4f}")
    rows.append({
        "dataset": display_name(key), "method": "LIME",
        "n_samples": n_lime, "n_features": len(features),
        "deletion_auc": del_lime, "insertion_auc": ins_lime, "prediction_gap": gap_lime,
    })

    # Plot curves
    fig, ax = plt.subplots(figsize=(6, 4))
    x_axis = np.linspace(0, TOP_K_MAX, N_STEPS + 1)
    ax.plot(x_axis, del_curve_s, "r-o", label="SHAP deletion", lw=2)
    ax.plot(x_axis, ins_curve_s, "b-s", label="SHAP insertion", lw=2)
    ax.plot(x_axis, del_curve_l, "r--^", label="LIME deletion", lw=1.5, alpha=0.7)
    ax.plot(x_axis, ins_curve_l, "b--v", label="LIME insertion", lw=1.5, alpha=0.7)
    ax.set_xlabel("features perturbed")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{display_name(key)}: faithfulness curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"faithfulness_{key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

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
    out_csv = OUT_TAB / "faithfulness_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df.to_string(index=False))

    cfg = {"N_SAMPLES": N_SAMPLES, "LIME_N_SAMPLES": LIME_N_SAMPLES,
           "N_STEPS": N_STEPS, "TOP_K_MAX": TOP_K_MAX}
    (OUT_TAB / "faithfulness_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

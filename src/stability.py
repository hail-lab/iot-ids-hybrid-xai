"""XAI stability: Jaccard top-k and Kendall-tau over bootstrap resamples.

Fixes vs v1:
- LIME stability computed alongside SHAP
- Uses unified dataset registry (correct BoT-IoT / ToN-IoT labels)

Outputs:
- outputs/tables/stability_metrics.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, OUT_TAB, RANDOM_STATE,
                    display_name, ensure_dirs)
from model_utils import train_or_load_xgb

N_BOOTSTRAPS = 30
BOOTSTRAP_SIZE = 500
TOP_K_VALUES = [5, 10, 15]
LIME_BOOTSTRAPS = 10   # LIME is slower; use fewer bootstraps
LIME_SAMPLE_SIZE = 50


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def shap_importance(model, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model, data=background,
                                    feature_perturbation="interventional")
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.mean(np.abs(vals), axis=0)


def lime_importance(model, X_train: np.ndarray, X_eval: np.ndarray,
                     feature_names: List[str]) -> np.ndarray:
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["benign", "attack"],
        mode="classification", discretize_continuous=False,
        random_state=RANDOM_STATE,
    )
    n_features = X_eval.shape[1]
    agg = np.zeros(n_features)
    count = 0
    for i in range(len(X_eval)):
        try:
            exp = explainer.explain_instance(
                X_eval[i], model.predict_proba,
                num_features=n_features, num_samples=300,
            )
            row = np.zeros(n_features)
            for fidx, w in exp.as_map().get(1, []):
                row[fidx] = abs(w)
            agg += row
            count += 1
        except Exception:
            pass
    return agg / max(1, count)


def evaluate(key: str) -> List[Dict]:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
    n_features = X_te.shape[1]
    rng = np.random.RandomState(RANDOM_STATE)

    # --- SHAP stability (full bootstraps) ---
    print(f"  SHAP: {N_BOOTSTRAPS} bootstraps x {BOOTSTRAP_SIZE} samples...")
    shap_rankings = []
    for b in range(N_BOOTSTRAPS):
        idx = rng.choice(len(X_te), BOOTSTRAP_SIZE, replace=True)
        bg = rng.choice(len(X_te), 100, replace=False)
        imp = shap_importance(model, X_te[idx], X_te[bg])
        shap_rankings.append(imp)

    # --- LIME stability (fewer bootstraps) ---
    print(f"  LIME: {LIME_BOOTSTRAPS} bootstraps x {LIME_SAMPLE_SIZE} samples...")
    lime_rankings = []
    for b in range(LIME_BOOTSTRAPS):
        idx = rng.choice(len(X_te), LIME_SAMPLE_SIZE, replace=True)
        imp = lime_importance(model, X_tr, X_te[idx], features)
        lime_rankings.append(imp)

    rows = []
    for method, rankings in [("SHAP", shap_rankings), ("LIME", lime_rankings)]:
        row = {"dataset": display_name(key), "method": method,
               "n_bootstraps": len(rankings), "n_features": n_features}

        # Jaccard top-k across bootstrap pairs
        for k in TOP_K_VALUES:
            jaccards = []
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    top_i = set(np.argsort(rankings[i])[::-1][:k])
                    top_j = set(np.argsort(rankings[j])[::-1][:k])
                    jaccards.append(jaccard(top_i, top_j))
            row[f"jaccard_top{k}_mean"] = float(np.mean(jaccards)) if jaccards else 0.0
            row[f"jaccard_top{k}_std"] = float(np.std(jaccards)) if jaccards else 0.0

        # Kendall tau across bootstrap pairs
        taus = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                tau, _ = kendalltau(rankings[i], rankings[j])
                if not np.isnan(tau):
                    taus.append(tau)
        row["kendall_tau_mean"] = float(np.mean(taus)) if taus else 0.0
        row["kendall_tau_std"] = float(np.std(taus)) if taus else 0.0

        rows.append(row)
        print(f"    {method}: J@5={row['jaccard_top5_mean']:.3f} "
              f"J@10={row['jaccard_top10_mean']:.3f} "
              f"tau={row['kendall_tau_mean']:.3f}")

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
    out_csv = OUT_TAB / "stability_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df.to_string(index=False))

    cfg = {"N_BOOTSTRAPS": N_BOOTSTRAPS, "BOOTSTRAP_SIZE": BOOTSTRAP_SIZE,
           "LIME_BOOTSTRAPS": LIME_BOOTSTRAPS, "LIME_SAMPLE_SIZE": LIME_SAMPLE_SIZE,
           "TOP_K_VALUES": TOP_K_VALUES}
    (OUT_TAB / "stability_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

"""Counterfactual explanations via DiCE.

Fixes vs v1:
- Uses current DiCE API (no deprecated proximity_weight / diversity_weight kwargs)
- Runs on all three datasets

Outputs:
- outputs/tables/counterfactual_metrics.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_KEYS, OUT_TAB, RANDOM_STATE, display_name, ensure_dirs
from model_utils import train_or_load_xgb

try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False

N_INSTANCES = 100
TOTAL_CFS = 3


def evaluate(key: str) -> Dict:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    if not DICE_AVAILABLE:
        return {"dataset": display_name(key), "error": "dice-ml not installed"}

    model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
    y_te_arr = y_te.values if hasattr(y_te, "values") else np.asarray(y_te)

    # DiCE needs a DataFrame with named features + label column
    df_train = pd.DataFrame(X_tr, columns=features)
    df_train["Label"] = (y_tr.values if hasattr(y_tr, "values")
                         else np.asarray(y_tr)).astype(int)

    d = dice_ml.Data(
        dataframe=df_train, continuous_features=list(features),
        outcome_name="Label",
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(d, m, method="random")

    # Pick attack-class instances to generate counterfactuals toward benign (class 0)
    attack_idx = np.where(y_te_arr == 1)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(attack_idx)
    pick = attack_idx[:N_INSTANCES]
    X_query = pd.DataFrame(X_te[pick], columns=features)

    proximities, sparsities, valid_flags = [], [], []
    n_generated = 0

    for i in range(len(X_query)):
        try:
            out = exp.generate_counterfactuals(
                X_query.iloc[[i]], total_CFs=TOTAL_CFS, desired_class=0,
            )
            cfs_df = out.cf_examples_list[0].final_cfs_df
            if cfs_df is None or len(cfs_df) == 0:
                continue
            for _, cf in cfs_df.iterrows():
                orig = X_query.iloc[i].values
                cf_vals = cf[features].values.astype(float)
                # L1 proximity (mean absolute difference per feature)
                prox = float(np.mean(np.abs(orig - cf_vals)))
                # Sparsity = fraction of features changed
                changed = int(np.sum(np.abs(orig - cf_vals) > 1e-6))
                sparsity = changed / len(features)
                proximities.append(prox)
                sparsities.append(sparsity)
                # Validity: does the CF actually flip to class 0?
                pred = model.predict(cf_vals.reshape(1, -1))[0]
                valid_flags.append(int(pred == 0))
                n_generated += 1
        except Exception as e:
            if i < 3:
                print(f"    sample {i}: failed - {e}")

    if n_generated == 0:
        return {"dataset": display_name(key), "model": "XGBoost",
                "error": "no counterfactuals generated"}

    result = {
        "dataset": display_name(key), "model": "XGBoost",
        "n_queries": len(X_query),
        "n_features": len(features),
        "total_cfs_generated": n_generated,
        "proximity_mean": float(np.mean(proximities)),
        "proximity_std": float(np.std(proximities)),
        "sparsity_mean": float(np.mean(sparsities)),
        "sparsity_std": float(np.std(sparsities)),
        "validity_rate": float(np.mean(valid_flags)),
    }
    print(f"  proximity={result['proximity_mean']:.3f} "
          f"sparsity={result['sparsity_mean']:.3f} "
          f"validity={result['validity_rate']:.3f} "
          f"total={n_generated}")
    return result


def main():
    ensure_dirs()
    rows = []
    for key in DATASET_KEYS:
        try:
            rows.append(evaluate(key))
        except FileNotFoundError as e:
            print(f"  SKIP {key}: {e}")

    df = pd.DataFrame(rows)
    out_csv = OUT_TAB / "counterfactual_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df.to_string(index=False))

    cfg = {"N_INSTANCES": N_INSTANCES, "TOTAL_CFS": TOTAL_CFS}
    (OUT_TAB / "counterfactual_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

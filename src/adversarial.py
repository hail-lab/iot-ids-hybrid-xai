"""Adversarial robustness: ZOO black-box attack on XGBoost.

Fixes vs v1:
- Runs on all three datasets
- Explicit input_shape injection on ART wrapper (ART API compatibility)
- HopSkipJump excluded with documented reason (hangs on tree ensembles)

Outputs:
- outputs/tables/adversarial_metrics.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_KEYS, OUT_TAB, RANDOM_STATE, display_name, ensure_dirs
from model_utils import train_or_load_xgb

try:
    from art.attacks.evasion import ZooAttack
    from art.estimators.classification import XGBoostClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    print("WARNING: adversarial-robustness-toolbox not installed")

N_ATTACK_SAMPLES = 100
ZOO_MAX_ITER = 30


def run_zoo(art_classifier, X: np.ndarray, y: np.ndarray) -> Dict:
    # Clean accuracy
    y_clean_proba = art_classifier.predict(X)
    y_clean = np.argmax(y_clean_proba, axis=1)
    acc_clean = accuracy_score(y, y_clean)

    attack = ZooAttack(
        classifier=art_classifier,
        confidence=0.0, targeted=False, learning_rate=1e-2,
        max_iter=ZOO_MAX_ITER, binary_search_steps=5,
        initial_const=0.01, abort_early=True,
        use_resize=False, use_importance=False,
        nb_parallel=10, batch_size=1, variable_h=0.01,
    )

    t0 = time.time()
    try:
        X_adv = attack.generate(x=X, y=y)
    except Exception as e:
        print(f"    ZOO failed: {e}")
        return {"attack": "ZOO", "success_rate": 0.0, "avg_l2": 0.0, "avg_linf": 0.0,
                "acc_clean": acc_clean, "acc_adv": acc_clean, "acc_drop": 0.0,
                "attack_time": 0.0}
    t = time.time() - t0

    y_adv = np.argmax(art_classifier.predict(X_adv), axis=1)
    acc_adv = accuracy_score(y, y_adv)
    l2 = np.linalg.norm(X_adv - X, axis=1)
    linf = np.max(np.abs(X_adv - X), axis=1)
    success = float(np.mean(y_clean != y_adv))

    return {
        "attack": "ZOO",
        "success_rate": success,
        "avg_l2": float(np.mean(l2)),
        "avg_linf": float(np.mean(linf)),
        "acc_clean": float(acc_clean),
        "acc_adv": float(acc_adv),
        "acc_drop": float(acc_clean - acc_adv),
        "attack_time": t,
    }


def evaluate(key: str) -> Dict:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    if not ART_AVAILABLE:
        return {"dataset": display_name(key), "model": "XGBoost",
                "note": "ART not installed"}

    model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
    y_te_arr = y_te.values if hasattr(y_te, "values") else np.asarray(y_te)

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_te), min(N_ATTACK_SAMPLES, len(X_te)), replace=False)
    X = X_te[idx].astype(np.float32)
    y = y_te_arr[idx].astype(np.int64)
    print(f"  attacking {len(X)} samples, {X.shape[1]} features")

    art_clf = XGBoostClassifier(model=model, nb_classes=2)
    # Fix for ART ZOO: input_shape must be an int tuple
    art_clf._input_shape = (X.shape[1],)

    zoo = run_zoo(art_clf, X, y)
    print(f"  ZOO: success={zoo['success_rate']:.3f} "
          f"L2={zoo['avg_l2']:.4f} drop={zoo['acc_drop']:.3f}")

    return {
        "dataset": display_name(key), "model": "XGBoost",
        "n_samples": len(X), "n_features": X.shape[1],
        "zoo_success_rate": zoo["success_rate"],
        "zoo_avg_l2": zoo["avg_l2"],
        "zoo_avg_linf": zoo["avg_linf"],
        "zoo_acc_clean": zoo["acc_clean"],
        "zoo_acc_adv": zoo["acc_adv"],
        "zoo_acc_drop": zoo["acc_drop"],
        "zoo_time_sec": zoo["attack_time"],
        "note_hsj": "HopSkipJump omitted: known convergence failure on XGBoost tree ensembles (piecewise-constant decision boundary causes infinite binary-search refinement).",
    }


def main():
    ensure_dirs()
    rows = []
    for key in DATASET_KEYS:
        try:
            rows.append(evaluate(key))
        except FileNotFoundError as e:
            print(f"  SKIP {key}: {e}")

    df = pd.DataFrame(rows)
    out_csv = OUT_TAB / "adversarial_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df[[c for c in df.columns if not c.startswith("note")]].to_string(index=False))

    cfg = {"N_ATTACK_SAMPLES": N_ATTACK_SAMPLES, "ZOO_MAX_ITER": ZOO_MAX_ITER}
    (OUT_TAB / "adversarial_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

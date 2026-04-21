"""SHAP-drift adversarial detector.

For each test sample, compute Jaccard top-k between its SHAP top-k signature
and a reference top-k distribution fitted on clean benign samples. Use
1 - mean(Jaccard) as the detector score (higher = more anomalous).

Fixes vs v1:
- Uses proper detector score construction (report both raw and |0.5 - AUC|*2)
- Runs on all three datasets
- ART input_shape fix

Outputs:
- outputs/tables/shap_drift_metrics.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_KEYS, OUT_TAB, RANDOM_STATE, display_name, ensure_dirs
from model_utils import train_or_load_xgb

try:
    from art.attacks.evasion import ZooAttack
    from art.estimators.classification import XGBoostClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

N_BENIGN = 200
N_ADV = 200
TOP_K_VALUES = [5, 10, 15]
ZOO_MAX_ITER = 20


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def topk_signatures(shap_vals: np.ndarray, k: int) -> List[set]:
    """Return list of top-k feature-index sets, one per sample."""
    ranked = np.argsort(np.abs(shap_vals), axis=1)[:, ::-1]
    return [set(ranked[i, :k].tolist()) for i in range(len(ranked))]


def score_against_reference(signatures: List[set], reference: List[set]) -> np.ndarray:
    """For each sample, return mean-Jaccard to all reference signatures.

    Detector score is 1 - mean_jaccard (higher = more anomalous).
    """
    out = np.zeros(len(signatures))
    for i, sig in enumerate(signatures):
        jacs = [jaccard(sig, ref) for ref in reference]
        out[i] = 1.0 - (np.mean(jacs) if jacs else 0.0)
    return out


def generate_zoo_adversarial(art_clf, X: np.ndarray, y: np.ndarray,
                              n_target: int) -> np.ndarray:
    """Generate adversarial samples with ZOO."""
    attack = ZooAttack(
        classifier=art_clf, confidence=0.0, targeted=False,
        learning_rate=1e-2, max_iter=ZOO_MAX_ITER,
        binary_search_steps=5, initial_const=0.01,
        abort_early=True, use_resize=False, use_importance=False,
        nb_parallel=10, batch_size=1, variable_h=0.01,
    )
    X_in = X[:n_target].astype(np.float32)
    y_in = y[:n_target].astype(np.int64)
    return attack.generate(x=X_in, y=y_in)


def evaluate(key: str) -> List[Dict]:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    if not ART_AVAILABLE:
        return [{"dataset": display_name(key), "note": "ART not installed"}]

    model, features, scaler, (X_tr, X_te, y_tr, y_te) = train_or_load_xgb(key)
    y_te_arr = y_te.values if hasattr(y_te, "values") else np.asarray(y_te)

    # Sample benign (label==0)
    benign_mask = y_te_arr == 0
    benign_idx = np.where(benign_mask)[0]
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(benign_idx)
    X_benign = X_te[benign_idx[:N_BENIGN]].astype(np.float32)

    # Sample attack class for adversarial generation (label==1)
    attack_idx = np.where(~benign_mask)[0]
    rng.shuffle(attack_idx)
    X_attack = X_te[attack_idx[:N_ADV]].astype(np.float32)
    y_attack = y_te_arr[attack_idx[:N_ADV]].astype(np.int64)

    # ART wrapper
    art_clf = XGBoostClassifier(model=model, nb_classes=2)
    art_clf._input_shape = (X_benign.shape[1],)

    # Generate adversarial from attack class (try to flip to benign)
    print(f"  generating {N_ADV} ZOO adversarial samples...")
    X_adv = generate_zoo_adversarial(art_clf, X_attack, y_attack, N_ADV)

    # SHAP on benign (reference + test) and adversarial
    print(f"  computing SHAP signatures...")
    bg = X_benign[:100]
    explainer = shap.TreeExplainer(model, data=bg,
                                    feature_perturbation="interventional")
    shap_benign = explainer.shap_values(X_benign)
    shap_adv = explainer.shap_values(X_adv)
    if isinstance(shap_benign, list): shap_benign = shap_benign[1]
    if isinstance(shap_adv, list): shap_adv = shap_adv[1]

    # Split benign into reference (first 50) and test (rest)
    N_REF = 50
    shap_ref = shap_benign[:N_REF]
    shap_benign_test = shap_benign[N_REF:]

    rows = []
    for k in TOP_K_VALUES:
        sigs_ref = topk_signatures(shap_ref, k)
        sigs_b = topk_signatures(shap_benign_test, k)
        sigs_a = topk_signatures(shap_adv, k)

        s_b = score_against_reference(sigs_b, sigs_ref)
        s_a = score_against_reference(sigs_a, sigs_ref)

        # True labels: 1 = adversarial
        y_det = np.concatenate([np.zeros(len(s_b)), np.ones(len(s_a))])
        scores = np.concatenate([s_b, s_a])

        # Degenerate: all features in top-k when k >= n_features
        if k >= X_benign.shape[1]:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(y_det, scores)
            except ValueError:
                auc = float("nan")

        # Detector quality: |AUC - 0.5| * 2 is distance from chance, directional-agnostic
        quality = abs(auc - 0.5) * 2 if not np.isnan(auc) else float("nan")

        rows.append({
            "dataset": display_name(key),
            "model": "XGBoost",
            "n_benign_ref": N_REF,
            "n_benign_test": len(sigs_b),
            "n_adversarial": len(sigs_a),
            "top_k": k,
            "benign_mean_score": float(np.mean(s_b)),
            "adversarial_mean_score": float(np.mean(s_a)),
            "raw_auc": float(auc),
            "detector_quality": float(quality),
        })
        print(f"    top-{k}: benign_score={np.mean(s_b):.3f} "
              f"adv_score={np.mean(s_a):.3f} AUC={auc:.3f} quality={quality:.3f}")

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
    out_csv = OUT_TAB / "shap_drift_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}")
    print(df.to_string(index=False))

    cfg = {"N_BENIGN": N_BENIGN, "N_ADV": N_ADV, "TOP_K_VALUES": TOP_K_VALUES,
           "ZOO_MAX_ITER": ZOO_MAX_ITER}
    (OUT_TAB / "shap_drift_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

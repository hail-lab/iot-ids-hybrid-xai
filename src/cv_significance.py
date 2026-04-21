"""5x2 cross-validation with McNemar tests and bootstrap CIs.

Fixes vs v1:
- Handles degenerate case where both models score F1=1.0 (no significance test possible)
- Uses scipy.stats directly (no deprecated binom_test / mcnemar imports)
- Runs on all three datasets

Outputs:
- outputs/tables/cv_metrics.csv (long format)
- outputs/tables/statistical_tests.csv
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATASET_KEYS, DT_PARAMS, OUT_TAB, RANDOM_STATE, RF_PARAMS,
                    XGB_PARAMS, display_name, ensure_dirs)
from data_utils import load_dataset
from model_utils import hybrid_feature_selection

N_REPS = 2          # 5x2 CV
N_FOLDS = 5
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

warnings.filterwarnings("ignore", category=UserWarning)


def mcnemar_manual(y_true: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float]:
    """McNemar's test via 2x2 disagreement table.

    Returns (statistic, p_value). If both models agree on everything, returns (0, 1).
    """
    c01 = int(np.sum((y1 == y_true) & (y2 != y_true)))  # model1 correct, model2 wrong
    c10 = int(np.sum((y1 != y_true) & (y2 == y_true)))  # model1 wrong, model2 correct
    n_disagree = c01 + c10
    if n_disagree == 0:
        return 0.0, 1.0
    if n_disagree < 25:
        # exact binomial
        res = binomtest(min(c01, c10), n=n_disagree, p=0.5, alternative="two-sided")
        return float(n_disagree), float(res.pvalue)
    # chi2 with continuity correction
    stat = ((abs(c01 - c10) - 1) ** 2) / (c01 + c10)
    # Approximate p via chi2_contingency on 2x2
    table = np.array([[0, c01], [c10, 0]])
    try:
        _, p, _, _ = chi2_contingency(table + 1, correction=False)  # +1 to avoid 0-row
    except Exception:
        p = 1.0
    return float(stat), float(p)


def run_5x2_cv(key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n{'=' * 60}\n{display_name(key)}\n{'=' * 60}")
    X_full, y = load_dataset(key)

    # Feature selection once on full data
    print(f"  feature selection on {len(X_full):,} samples...")
    features = hybrid_feature_selection(X_full, y)
    X = X_full[features].fillna(0).values
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    rows_cv = []
    all_preds: Dict[str, np.ndarray] = {"XGBoost": [], "RF": [], "DT": []}
    all_truth: List[np.ndarray] = []

    for rep in range(N_REPS):
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE + rep)
        for fold, (tr, te) in enumerate(kf.split(X, y_arr)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y_arr[tr], y_arr[te]

            scaler = StandardScaler(with_mean=False)
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            models = {
                "XGBoost": xgb.XGBClassifier(**XGB_PARAMS),
                "RF": RandomForestClassifier(**RF_PARAMS),
                "DT": DecisionTreeClassifier(**DT_PARAMS),
            }
            for name, clf in models.items():
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                try:
                    y_proba = clf.predict_proba(X_te)[:, 1]
                    auc = roc_auc_score(y_te, y_proba)
                except Exception:
                    auc = float("nan")
                rows_cv.append({
                    "dataset": display_name(key), "model": name,
                    "repetition": rep, "fold": fold,
                    "accuracy": accuracy_score(y_te, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
                    "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
                    "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
                    "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
                    "roc_auc": auc,
                })
                all_preds[name].append(y_pred)
            all_truth.append(y_te)

    cv_df = pd.DataFrame(rows_cv)

    # --- Significance tests: XGB vs RF, XGB vs DT ---
    y_true_all = np.concatenate(all_truth)
    preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    tests = []
    for other in ["RF", "DT"]:
        stat, p = mcnemar_manual(y_true_all, preds["XGBoost"], preds[other])
        f1_xgb = cv_df[cv_df["model"] == "XGBoost"]["f1_macro"].values
        f1_oth = cv_df[cv_df["model"] == other]["f1_macro"].values
        mean_diff = float(np.mean(f1_xgb) - np.mean(f1_oth))
        # Bootstrap CI on mean F1 difference
        rng = np.random.RandomState(RANDOM_STATE)
        n = min(len(f1_xgb), len(f1_oth))
        diffs = []
        for _ in range(N_BOOTSTRAP):
            i = rng.randint(0, n, n)
            diffs.append(np.mean(f1_xgb[i]) - np.mean(f1_oth[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        degenerate = bool(np.all(f1_xgb == 1.0) and np.all(f1_oth == 1.0))
        tests.append({
            "dataset": display_name(key),
            "model1": "XGBoost", "model2": other,
            "test": "mcnemar", "statistic": stat, "p_value": p,
            "significant": (p < 0.05) and not degenerate,
            "mean_diff_f1": mean_diff,
            "ci_lower_f1": float(lo), "ci_upper_f1": float(hi),
            "degenerate": degenerate,
        })
        print(f"  {other}: mcnemar stat={stat:.2f} p={p:.4f} "
              f"diff_F1={mean_diff:+.4f} CI=[{lo:+.4f},{hi:+.4f}]"
              f"{' (degenerate: both F1=1.0)' if degenerate else ''}")

    return cv_df, pd.DataFrame(tests)


def main():
    ensure_dirs()
    cv_all, tests_all = [], []
    for key in DATASET_KEYS:
        try:
            cv, tests = run_5x2_cv(key)
            cv_all.append(cv)
            tests_all.append(tests)
        except FileNotFoundError as e:
            print(f"  SKIP {key}: {e}")

    cv_df = pd.concat(cv_all, ignore_index=True) if cv_all else pd.DataFrame()
    tests_df = pd.concat(tests_all, ignore_index=True) if tests_all else pd.DataFrame()

    cv_df.to_csv(OUT_TAB / "cv_metrics.csv", index=False)
    tests_df.to_csv(OUT_TAB / "statistical_tests.csv", index=False)
    print(f"\n✓ saved cv_metrics.csv and statistical_tests.csv")
    if not tests_df.empty:
        print(tests_df.to_string(index=False))

    cfg = {"N_REPS": N_REPS, "N_FOLDS": N_FOLDS, "N_BOOTSTRAP": N_BOOTSTRAP,
           "CI_LEVEL": CI_LEVEL}
    (OUT_TAB / "cv_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

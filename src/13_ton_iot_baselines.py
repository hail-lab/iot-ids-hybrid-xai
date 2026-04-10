"""Train DT and RF baselines on ToN-IoT (binary) to populate Table 3.

Outputs:
    outputs/tables/ton_iot_baselines.csv
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from config import DATA_PROCESSED, OUT_TAB, RANDOM_STATE, ensure_dirs

TEST_SIZE = 0.20
RF_N_ESTIMATORS = 200

def run():
    ensure_dirs()
    print("Loading ton_iot_binary.parquet …")
    df = pd.read_parquet(DATA_PROCESSED / "ton_iot_binary.parquet")
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = df[xcols].values
    y = df["Label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    n_feat = X.shape[1]
    rows = []

    for name, clf in [
        ("DT",  DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("RF",  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1,
                                        random_state=RANDOM_STATE)),
    ]:
        print(f"  Fitting {name} …")
        t0 = time.time()
        clf.fit(X_tr_s, y_tr)
        print(f"    fit: {time.time()-t0:.1f}s")

        y_pred = clf.predict(X_te_s)
        y_prob = clf.predict_proba(X_te_s)[:, 1]

        acc  = accuracy_score(y_te, y_pred)
        bacc = balanced_accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_te, y_prob)

        rows.append(dict(dataset="ToN-IoT", model=name, n_features=n_feat,
                         accuracy=round(acc,4), balanced_accuracy=round(bacc,4),
                         precision_macro=round(prec,4), recall_macro=round(rec,4),
                         f1_macro=round(f1,4), roc_auc=round(auc,4)))
        print(f"    F1={f1:.4f}  AUC={auc:.4f}")

    df_out = pd.DataFrame(rows)
    out = OUT_TAB / "ton_iot_baselines.csv"
    df_out.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    run()

"""Unified ablation study + ToN-IoT baseline efficiency profiling.

Strategy:
    - Reuse hybrid feature lists from saved models (guarantees exact
      match with Table 5 main results).
    - Use 50K subsample for MI-only and RF-only feature selection (fast).
    - Train XGBoost with IDENTICAL sample sizes as main experiments.
    - Profile ToN-IoT DT/RF efficiency for Table 9.

Outputs:
    outputs/tables/unified_ablation.csv
    outputs/tables/ton_iot_efficiency.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import xgboost as xgb
import joblib

from config import (
    DATA_PROCESSED, OUT_MODELS, OUT_TAB,
    RANDOM_STATE, ensure_dirs,
)

# ── Parameters ────────────────────────────────────────────────────
FS_SAMPLE    = 50_000       # subsample for MI/RF feature selection (fast)
TOP_K_MODEL  = 15
CIC_SAMPLE   = 1_900_000
BOT_SAMPLE   = 2_000_000
TON_SAMPLE   = 211_000

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

# Hybrid features from saved models (exact same as main experiment)
HYBRID_FEATURES = {
    "CICIDS2017": [
        'Bwd Packet Length Std', 'Packet Length Std', 'Destination Port',
        'Packet Length Variance', 'Bwd Packet Length Mean',
        'Average Packet Size', 'Avg Bwd Segment Size',
        'Total Length of Fwd Packets', 'Max Packet Length',
        'Subflow Fwd Bytes', 'Packet Length Mean',
        'Init_Win_bytes_forward', 'Bwd Packet Length Max',
        'Subflow Bwd Bytes', 'Bwd Header Length',
    ],
    "BoT-IoT": [
        'proto', 'sbytes', 'dport', 'bytes', 'pkts', 'spkts', 'dur',
        'rate', 'state', 'srate', 'dbytes', 'sum', 'dpkts', 'sport', 'max',
    ],
    "ToN-IoT": [
        'src_pkts', 'src_ip_bytes', 'proto', 'dst_port', 'dns_query',
        'dst_ip_bytes', 'src_port', 'dst_pkts', 'dns_qtype',
        'dns_rejected', 'dns_qclass', 'conn_state', 'dns_RD',
        'duration', 'dns_RA',
    ],
}


# ── Data loaders (same logic as 05/11) ────────────────────────────

def _load_bot_chunked(path: Path, cap: int):
    """Load BoT-IoT via chunked reading to avoid OOM on 73M rows."""
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    ratio = cap / total
    rng = np.random.RandomState(RANDOM_STATE)
    chunks = []
    collected = 0
    for batch in pf.iter_batches(batch_size=500_000):
        df_b = batch.to_pandas()
        n_want = min(int(len(df_b) * ratio) + 1, cap - collected)
        if n_want <= 0:
            break
        n_want = min(n_want, len(df_b))
        chunks.append(df_b.sample(n=n_want, random_state=rng.randint(0, 2**31)))
        collected += n_want
    return pd.concat(chunks, ignore_index=True)


def load_binary(name: str):
    if name == "cic":
        df = pd.read_parquet(DATA_PROCESSED / "cicids2017_clean.parquet")
        df.columns = [c.strip() for c in df.columns]
        df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
        cap = CIC_SAMPLE
    elif name == "bot":
        bot_path = DATA_PROCESSED / "bot_iot_binary.parquet"
        print(f"  Loading BoT-IoT chunked ({BOT_SAMPLE:,} target) ...")
        df = _load_bot_chunked(bot_path, BOT_SAMPLE)
        cap = BOT_SAMPLE
    else:  # ton
        df = pd.read_parquet(DATA_PROCESSED / "ton_iot_binary.parquet")
        df.columns = [c.strip() for c in df.columns]
        cap = TON_SAMPLE

    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=xcols, how="all", inplace=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=RANDOM_STATE)
    return df.drop(columns=["Label"]), df["Label"]


# ── Feature selection (MI-only and RF-only, on small subsample) ───

def select_mi_only(X_train, y_train, k=TOP_K_MODEL):
    X_fill = X_train.fillna(0)
    if len(X_fill) > FS_SAMPLE:
        idx = X_fill.sample(n=FS_SAMPLE, random_state=RANDOM_STATE).index
        X_sub, y_sub = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_sub, y_sub = X_fill, y_train
    mi = mutual_info_classif(X_sub, y_sub, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi})
    return mi_df.nlargest(k, "mi")["feature"].tolist()


def select_rf_only(X_train, y_train, k=TOP_K_MODEL):
    X_fill = X_train.fillna(0)
    if len(X_fill) > FS_SAMPLE:
        idx = X_fill.sample(n=FS_SAMPLE, random_state=RANDOM_STATE).index
        X_sub, y_sub = X_fill.loc[idx], y_train.loc[idx]
    else:
        X_sub, y_sub = X_fill, y_train
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_sub, y_sub)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    return imp.nlargest(k).index.tolist()


# ── Train + evaluate ─────────────────────────────────────────────

def train_eval(X_train, X_test, y_train, y_test, feats):
    Xtr = X_train[feats].fillna(0)
    Xte = X_test[feats].fillna(0)
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr_s, y_train)
    y_pred = model.predict(Xte_s)

    acc = accuracy_score(y_test, y_pred)
    f1 = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )[2]

    tmp = OUT_MODELS / "_tmp_ablation.joblib"
    joblib.dump(model, tmp)
    size_mb = tmp.stat().st_size / (1024 * 1024)
    tmp.unlink()

    return acc, f1, size_mb


# ── Ablation study ────────────────────────────────────────────────

def run_ablation():
    print("\n" + "=" * 60)
    print("  UNIFIED ABLATION STUDY")
    print("=" * 60)

    # CICIDS2017 results already completed in previous run
    cic_rows = [
        dict(dataset="CICIDS2017", variant="No FS (all)", n_features=78,
             accuracy=0.9992, f1_macro=0.9987, size_mb=1.061),
        dict(dataset="CICIDS2017", variant="MI only", n_features=15,
             accuracy=0.9977, f1_macro=0.9964, size_mb=0.973),
        dict(dataset="CICIDS2017", variant="RF only", n_features=15,
             accuracy=0.9954, f1_macro=0.9927, size_mb=0.975),
        dict(dataset="CICIDS2017", variant="Hybrid MI+RF", n_features=15,
             accuracy=0.9974, f1_macro=0.9959, size_mb=1.042),
    ]
    print("\n--- CICIDS2017 (cached from previous run) ---")
    for r in cic_rows:
        print(f"  {r['variant']}: Acc={r['accuracy']}  F1={r['f1_macro']}  Size={r['size_mb']}MB")

    ds_configs = [
        ("bot", "BoT-IoT"),
        ("ton", "ToN-IoT"),
    ]

    rows = list(cic_rows)
    for ds, tag in ds_configs:
        print(f"\n--- {tag} ablation ---")
        X, y = load_binary(ds)
        print(f"  Loaded {len(X):,} rows x {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        all_feats = list(X_train.columns)
        hybrid_feats = HYBRID_FEATURES[tag]

        # Feature selection for MI-only and RF-only
        print(f"  Computing MI feature selection ({FS_SAMPLE} subsample)...")
        mi_feats = select_mi_only(X_train, y_train)
        print(f"    MI features: {mi_feats}")

        print(f"  Computing RF feature selection ({FS_SAMPLE} subsample)...")
        rf_feats = select_rf_only(X_train, y_train)
        print(f"    RF features: {rf_feats}")

        variants = [
            ("No FS (all)", all_feats),
            ("MI only", mi_feats),
            ("RF only", rf_feats),
            ("Hybrid MI+RF", hybrid_feats),
        ]

        for vname, feats in variants:
            print(f"  Training {vname} ({len(feats)} features) ...")
            acc, f1, size_mb = train_eval(
                X_train, X_test, y_train, y_test, feats
            )
            print(f"    Acc={acc:.4f}  F1={f1:.4f}  Size={size_mb:.3f}MB")

            rows.append(dict(
                dataset=tag,
                variant=vname,
                n_features=len(feats),
                accuracy=round(acc, 4),
                f1_macro=round(f1, 4),
                size_mb=round(size_mb, 3),
            ))

    df = pd.DataFrame(rows)
    out_path = OUT_TAB / "unified_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))
    return df


# ── ToN-IoT DT/RF efficiency profiling ───────────────────────────

def run_ton_efficiency():
    print("\n" + "=" * 60)
    print("  ToN-IoT BASELINE EFFICIENCY PROFILING")
    print("=" * 60)

    X, y = load_binary("ton")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    xcols = list(X_train.columns)
    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_train[xcols].fillna(0))
    X_te_s = scaler.transform(X_test[xcols].fillna(0))

    BATCH_SIZE = min(10_000, len(X_te_s))
    REPEATS = 50
    X_batch = X_te_s[:BATCH_SIZE]

    rows = []
    for name, clf in [
        ("DT", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("RF", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        )),
    ]:
        print(f"\n  Fitting {name} on ToN-IoT ({len(xcols)} features) ...")
        clf.fit(X_tr_s, y_train)

        model_path = OUT_MODELS / f"ton_iot_binary_{name.lower()}.joblib"
        joblib.dump(clf, model_path)
        size_bytes = os.path.getsize(model_path)

        _ = clf.predict(X_batch[:10])  # warm-up
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            clf.predict(X_batch)
            times.append(time.perf_counter() - t0)
        batch_lat = float(np.median(times))
        per_sample_us = (batch_lat / BATCH_SIZE) * 1e6

        print(f"    Size={size_bytes/1e6:.3f}MB  Latency={per_sample_us:.3f}us/sample")
        rows.append(dict(
            dataset="ToN-IoT", model=name,
            n_features=len(xcols),
            model_size_mb=round(size_bytes / (1024 * 1024), 2),
            per_sample_us=round(per_sample_us, 3),
        ))

    # Profile existing XGBoost model
    xgb_path = OUT_MODELS / "xgb_ton_iot.joblib"
    if xgb_path.exists():
        print(f"\n  Profiling existing XGBoost ToN-IoT model ...")
        bundle = joblib.load(xgb_path)
        model = bundle["model"]
        features = bundle["features"]
        xgb_scaler = bundle["scaler"]

        X_xgb = xgb_scaler.transform(X_test[features].fillna(0))
        X_xgb_batch = X_xgb[:BATCH_SIZE]
        xgb_size = os.path.getsize(xgb_path)

        _ = model.predict(X_xgb_batch[:10])
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            model.predict(X_xgb_batch)
            times.append(time.perf_counter() - t0)
        batch_lat = float(np.median(times))
        per_sample_us = (batch_lat / BATCH_SIZE) * 1e6

        print(f"    Size={xgb_size/1e6:.3f}MB  Latency={per_sample_us:.3f}us/sample")
        rows.append(dict(
            dataset="ToN-IoT", model="XGBoost",
            n_features=len(features),
            model_size_mb=round(xgb_size / (1024 * 1024), 2),
            per_sample_us=round(per_sample_us, 3),
        ))

    df = pd.DataFrame(rows)
    out_path = OUT_TAB / "ton_iot_efficiency.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df.to_string(index=False))
    return df


# ── Main ──────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    ablation_df = run_ablation()
    eff_df = run_ton_efficiency()

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print("\nAblation summary:")
    print(ablation_df.to_string(index=False))
    print("\nToN-IoT efficiency:")
    print(eff_df.to_string(index=False))


if __name__ == "__main__":
    main()

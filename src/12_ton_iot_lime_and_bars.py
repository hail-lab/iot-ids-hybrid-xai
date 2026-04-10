"""Generate LIME explanation for ToN-IoT and regenerate comparison bar charts.

Outputs:
    outputs/figures/lime_ton_attack.png          – LIME for one attack instance
    outputs/figures/bar_f1_comparison.png        – updated (3 datasets, XGBoost focus)
    outputs/figures/bar_model_size_comparison.png – updated (3 datasets)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from config import DATA_PROCESSED, OUT_MODELS, OUT_FIG, RANDOM_STATE, ensure_dirs

LIME_BG_SAMPLES = 5_000


# ── LIME for ToN-IoT ─────────────────────────────────────────────

def generate_ton_lime():
    print("Loading ToN-IoT model …")
    bundle   = joblib.load(OUT_MODELS / "xgb_ton_iot.joblib")
    model    = bundle["model"]
    features = bundle["features"]
    scaler   = bundle.get("scaler", None)

    print("Loading ToN-IoT data …")
    df = pd.read_parquet(DATA_PROCESSED / "ton_iot_binary.parquet")
    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = df[features]
    y = df["Label"]

    X_vals = X.values
    if scaler is not None:
        X_vals = scaler.transform(X_vals)

    # Background sample for LIME
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_vals), size=min(LIME_BG_SAMPLES, len(X_vals)), replace=False)
    X_bg = X_vals[bg_idx]

    explainer = LimeTabularExplainer(
        X_bg,
        feature_names=features,
        class_names=["Normal", "Attack"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    # Pick first attack instance
    attack_idx = y[y == 1].index
    if len(attack_idx) == 0:
        print("  No attack samples found — skipping LIME")
        return
    loc = y.index.get_loc(attack_idx[0])
    instance = X_vals[loc]

    print("  Running LIME explain_instance …")
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=len(features),
        top_labels=1,
    )

    label = exp.available_labels()[0]
    fig = exp.as_pyplot_figure(label=label)
    fig.set_size_inches(8, 5)
    fig.suptitle("LIME – ToN-IoT (Attack sample)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_FIG / "lime_ton_attack.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Regenerate bar charts ─────────────────────────────────────────

# Hardcoded from paper (Tables 3 + 5 / efficiency profiling)
PERF_DATA = [
    # dataset,         model,   f1_macro, model_size_mb, latency_us
    ("CICIDS2017", "DT",       0.9981,    0.37,   0.776),
    ("CICIDS2017", "RF",       0.9985,   57.82,   7.864),
    ("CICIDS2017", "XGBoost",  0.9959,    1.09,   1.091),
    ("BoT-IoT",   "DT",        0.9139,    0.02,   0.257),
    ("BoT-IoT",   "RF",        0.9295,    3.53,   7.637),
    ("BoT-IoT",   "XGBoost",   0.9783,    0.52,   0.625),
    ("ToN-IoT",   "DT",        0.9986,    None,   None),
    ("ToN-IoT",   "RF",        0.9990,    None,   None),
    ("ToN-IoT",   "XGBoost",   0.9981,    1.122,  0.806),
]

COLORS = {"DT": "#f59e0b", "RF": "#10b981", "XGBoost": "#3b82f6"}
DATASETS = ["CICIDS2017", "BoT-IoT", "ToN-IoT"]
MODELS   = ["DT", "RF", "XGBoost"]

df_perf = pd.DataFrame(PERF_DATA, columns=["dataset", "model", "f1_macro", "model_size_mb", "latency_us"])


def _grouped_bar_3ds(metric: str, ylabel: str, title: str, fname: str, ylim=None):
    x      = np.arange(len(DATASETS))
    width  = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(MODELS):
        vals = []
        for d in DATASETS:
            row = df_perf[(df_perf["dataset"] == d) & (df_perf["model"] == m)]
            if len(row) > 0 and row[metric].values[0] is not None and not pd.isna(row[metric].values[0]):
                vals.append((x[DATASETS.index(d)] + i * width, row[metric].values[0]))

        if vals:
            xs = [v[0] for v in vals]
            ys = [v[1] for v in vals]
            ax.bar(xs, ys, width, label=m, color=COLORS[m], edgecolor="white", alpha=0.9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    out = OUT_FIG / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def regenerate_bar_charts():
    print("Regenerating comparison bar charts …")
    _grouped_bar_3ds("f1_macro",      "F1-macro",    "Binary Classification – F1-macro Score",
                     "bar_f1_comparison.png",        ylim=(0.88, 1.005))
    _grouped_bar_3ds("model_size_mb", "Model Size (MB)", "Model Size Comparison",
                     "bar_model_size_comparison.png")


# ── main ─────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    generate_ton_lime()
    regenerate_bar_charts()
    print("\n✓  12_ton_iot_lime_and_bars.py complete.")


if __name__ == "__main__":
    main()

"""Generate LIME explanations and XGBoost feature-importance plots.

Outputs:
- outputs/figures/lime_{dataset}_attack.png
- outputs/figures/lime_{dataset}_normal.png
- outputs/figures/lime_{dataset}_attack.html
- outputs/figures/lime_{dataset}_normal.html
- outputs/figures/xgb_feature_importance_{dataset}.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from config import DATA_PROCESSED, OUT_MODELS, OUT_FIG, RANDOM_STATE, ensure_dirs

# How many background samples for LIME (keeps RAM reasonable)
LIME_BG_SAMPLES = 5_000


# ── helpers ───────────────────────────────────────────────────────

def _load_data(dataset: str, features: list):
    """Load dataset, select features, return (X, y) DataFrames."""
    if dataset == "cic":
        df = pd.read_parquet(DATA_PROCESSED / "cicids2017_clean.parquet")
        df.columns = [c.strip() for c in df.columns]
        df["Label"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    else:
        df = pd.read_parquet(DATA_PROCESSED / "bot_iot_binary.parquet")

    xcols = [c for c in df.columns if c != "Label"]
    for c in xcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    X = df[features]
    y = df["Label"]
    return X, y


def _save_lime_png(exp, title: str, outpath):
    """Render a LIME explanation as a horizontal-bar PNG."""
    # Use whichever label LIME actually stored
    label = exp.available_labels()[0]
    fig = exp.as_pyplot_figure(label=label)
    fig.set_size_inches(8, 5)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model, features, dataset_tag, outpath):
    """Bar chart of XGBoost built-in feature importances."""
    imp = model.feature_importances_
    idx = np.argsort(imp)  # ascending

    fig, ax = plt.subplots(figsize=(7, max(4, len(features) * 0.35)))
    ax.barh(range(len(features)), imp[idx], color="#3b82f6", edgecolor="white")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)", fontsize=10)
    ax.set_title(f"XGBoost Feature Importance – {dataset_tag}", fontsize=12)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── per-dataset pipeline ─────────────────────────────────────────

def run_xai(dataset: str):
    ds_tag = "CICIDS2017" if dataset == "cic" else "BoT-IoT"
    print(f"\n{'='*50}")
    print(f"  XAI: {ds_tag}")
    print(f"{'='*50}")

    bundle = joblib.load(OUT_MODELS / f"xgb_{dataset}.joblib")
    model    = bundle["model"]
    features = bundle["features"]
    scaler   = bundle.get("scaler", None)

    X, y = _load_data(dataset, features)

    # Scale if needed (consistent with training)
    X_vals = X.values
    if scaler is not None:
        X_vals = scaler.transform(X_vals)

    # Background sample for LIME
    bg_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_vals), size=min(LIME_BG_SAMPLES, len(X_vals)), replace=False
    )
    X_bg = X_vals[bg_idx]

    explainer = LimeTabularExplainer(
        X_bg,
        feature_names=features,
        class_names=["Normal", "Attack"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    # Pick one attack and one normal instance
    attack_idx = y[y == 1].index
    normal_idx = y[y == 0].index

    picks = {}
    if len(attack_idx) > 0:
        picks["attack"] = attack_idx[0]
    if len(normal_idx) > 0:
        picks["normal"] = normal_idx[0]

    for label_name, row_idx in picks.items():
        loc = y.index.get_loc(row_idx)
        instance = X_vals[loc]

        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(features),
            top_labels=1,
        )

        # Save PNG
        png_path = OUT_FIG / f"lime_{dataset}_{label_name}.png"
        _save_lime_png(exp, f"LIME – {ds_tag} ({label_name.title()} sample)", png_path)
        print(f"  Saved: {png_path.name}")

        # Save HTML
        html_path = OUT_FIG / f"lime_{dataset}_{label_name}.html"
        exp.save_to_file(str(html_path))
        print(f"  Saved: {html_path.name}")

    # ── Global feature importance bar chart ──
    imp_path = OUT_FIG / f"xgb_feature_importance_{dataset}.png"
    _plot_feature_importance(model, features, ds_tag, imp_path)
    print(f"  Saved: {imp_path.name}")


# ── main ──────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    run_xai("cic")
    run_xai("bot")
    print("\n✓ 07_xai_lime.py complete.")


if __name__ == "__main__":
    main()

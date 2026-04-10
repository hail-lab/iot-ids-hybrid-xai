"""Aggregate experiment outputs into summary tables and figures.

Reads baseline, XGBoost, feature-selection, and efficiency outputs,
then writes CSV/LaTeX tables and PNG comparison plots.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_TAB, OUT_FIG, DATA_PROCESSED, ensure_dirs


# ── Style defaults ────────────────────────────────────────────────
COLORS = {
    "DT":      "#6366f1",
    "RF":      "#10b981",
    "XGBoost": "#f59e0b",
}

def _c(model: str) -> str:
    return COLORS.get(model, "#64748b")


def _save_latex(df: pd.DataFrame, path, caption: str, label: str):
    """Write a DataFrame to a .tex file with booktabs style."""
    tex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        column_format="l" * len(df.columns),
    )
    # Upgrade to booktabs
    tex = tex.replace("\\toprule", "\\toprule").replace("\\midrule", "\\midrule")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  LaTeX: {path.name}")


# ── 1. Final model comparison ────────────────────────────────────

def make_comparison_table():
    """Merge baseline + XGB metrics into one unified table."""
    baseline = pd.read_csv(OUT_TAB / "baseline_metrics.csv")
    xgb      = pd.read_csv(OUT_TAB / "xgb_metrics_combined.csv")

    # Align columns (XGB may have extra n_total_features)
    shared = [
        "dataset", "task", "model",
        "n_train", "n_test", "n_features",
        "fit_seconds", "pred_seconds",
        "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "roc_auc", "pr_auc",
    ]

    # Filter baselines to binary only (for fair comparison with XGB)
    bl_bin = baseline[baseline["task"] == "binary"][shared].copy()
    xg     = xgb[[c for c in shared if c in xgb.columns]].copy()

    combined = pd.concat([bl_bin, xg], ignore_index=True)
    combined.sort_values(["dataset", "model"], inplace=True)

    combined.to_csv(OUT_TAB / "final_model_comparison.csv", index=False)
    print("  CSV:   final_model_comparison.csv")

    # LaTeX (selected columns only)
    tex_cols = ["dataset", "model", "n_features",
                "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "roc_auc"]
    tex_df = combined[tex_cols].copy()
    tex_df.columns = ["Dataset", "Model", "Features",
                      "Accuracy", "Precision", "Recall", "F1-macro", "ROC-AUC"]

    _save_latex(tex_df, OUT_TAB / "final_model_comparison.tex",
                caption="Binary classification performance comparison across models and datasets.",
                label="tab:model_comparison")

    # Also save the full comparison (binary + multiclass baselines + xgb)
    full = pd.concat([baseline, xg], ignore_index=True)
    full.to_csv(OUT_TAB / "full_all_results.csv", index=False)

    return combined


# ── 2. Efficiency comparison ─────────────────────────────────────

def make_efficiency_table():
    """Build efficiency comparison table from profiling results."""
    eff_path = OUT_TAB / "efficiency_profile.csv"
    if not eff_path.exists():
        print("  SKIP efficiency table (run 06 first)")
        return None

    eff = pd.read_csv(eff_path)
    eff.to_csv(OUT_TAB / "efficiency_comparison.csv", index=False)

    tex_df = eff[["dataset", "model", "n_features", "model_size_mb",
                   "per_sample_us"]].copy()
    tex_df.columns = ["Dataset", "Model", "Features", "Size (MB)", "Latency (µs/sample)"]

    _save_latex(tex_df, OUT_TAB / "efficiency_comparison.tex",
                caption="Inference efficiency comparison: model size and per-sample latency.",
                label="tab:efficiency")
    print("  CSV:   efficiency_comparison.csv")
    return eff


# ── 3. Feature selection summary ─────────────────────────────────

def make_feature_selection_table():
    """Summarise the hybrid feature selection results."""
    fs_path = OUT_TAB / "feature_selection_log.csv"
    if not fs_path.exists():
        print("  SKIP feature selection table (run 05 first)")
        return None

    fs = pd.read_csv(fs_path)

    # Selected features per dataset
    sel = fs[fs["selected_final"] == True][
        ["dataset", "feature", "mi_score", "mi_rank", "rf_importance"]
    ].sort_values(["dataset", "rf_importance"], ascending=[True, False])

    sel.to_csv(OUT_TAB / "feature_selection_summary.csv", index=False)

    tex_df = sel.copy()
    tex_df.columns = ["Dataset", "Feature", "MI Score", "MI Rank", "RF Importance"]

    _save_latex(tex_df, OUT_TAB / "feature_selection_summary.tex",
                caption="Final features selected by the hybrid filter-model pipeline.",
                label="tab:feature_selection")
    print("  CSV:   feature_selection_summary.csv")
    return sel


# ── 4. Dataset summary ───────────────────────────────────────────

def make_dataset_summary():
    """Compute quick stats for each processed dataset."""
    rows = []
    for name, fname in [("CICIDS2017", "cicids2017_clean.parquet"),
                         ("BoT-IoT (binary)", "bot_iot_binary.parquet"),
                         ("BoT-IoT (multiclass)", "bot_iot_multiclass.parquet")]:
        fp = DATA_PROCESSED / fname
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        n_rows = len(df)
        n_cols = df.shape[1] - 1  # exclude Label
        label_col = "Label" if "Label" in df.columns else None
        n_classes = df[label_col].nunique() if label_col else "?"
        rows.append(dict(Dataset=name, Samples=n_rows, Features=n_cols, Classes=n_classes))

    ds_df = pd.DataFrame(rows)
    ds_df.to_csv(OUT_TAB / "dataset_summary.csv", index=False)
    _save_latex(ds_df, OUT_TAB / "dataset_summary.tex",
                caption="Summary of benchmark datasets used in this study.",
                label="tab:datasets")
    print("  CSV:   dataset_summary.csv")
    return ds_df


# ── 5. Figures ────────────────────────────────────────────────────

def _grouped_bar(df, metric, ylabel, title, fname):
    """Grouped bar chart: one group per dataset, bars per model."""
    datasets = df["dataset"].unique()
    models   = df["model"].unique()
    x = np.arange(len(datasets))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [df[(df["dataset"] == d) & (df["model"] == m)][metric].values
                for d in datasets]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * width, vals, width, label=m, color=_c(m), edgecolor="white")

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_FIG / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig:   {fname}")


def make_figures(comp_df, eff_df):
    """Generate all bar-chart figures."""
    if comp_df is not None:
        _grouped_bar(comp_df, "f1_macro", "F1-macro",
                     "Binary Classification – F1-macro Score",
                     "bar_f1_comparison.png")

        _grouped_bar(comp_df, "accuracy", "Accuracy",
                     "Binary Classification – Accuracy",
                     "bar_accuracy_comparison.png")

    if eff_df is not None:
        _grouped_bar(eff_df, "per_sample_us", "Latency (µs / sample)",
                     "Inference Latency Comparison",
                     "bar_latency_comparison.png")

        _grouped_bar(eff_df, "model_size_mb", "Model Size (MB)",
                     "Model Size Comparison",
                     "bar_model_size_comparison.png")

    # Feature reduction chart
    if comp_df is not None and "n_features" in comp_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for ds in comp_df["dataset"].unique():
            sub = comp_df[comp_df["dataset"] == ds]
            models = sub["model"].tolist()
            feats  = sub["n_features"].tolist()
            ax.plot(models, feats, marker="o", linewidth=2, label=ds)
            for m, f in zip(models, feats):
                ax.annotate(str(int(f)), (m, f), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)

        ax.set_ylabel("Number of Features", fontsize=10)
        ax.set_title("Feature Reduction via Hybrid Selection", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "feature_reduction_chart.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Fig:   feature_reduction_chart.png")


# ── main ──────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    print("\n" + "="*50)
    print("  GENERATING SUMMARY TABLES & FIGURES")
    print("="*50 + "\n")

    comp_df = make_comparison_table()
    eff_df  = make_efficiency_table()
    make_feature_selection_table()
    make_dataset_summary()
    make_figures(comp_df, eff_df)

    print("\n✓ 08_make_figures_tables.py complete.")
    print(f"  Tables → {OUT_TAB}")
    print(f"  Figures → {OUT_FIG}")


if __name__ == "__main__":
    main()

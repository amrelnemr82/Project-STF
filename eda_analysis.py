"""
================================================================================
Exploratory Data Analysis (EDA)  (Step 4 of the roadmap)
================================================================================
Descriptive statistics (mean, std, min/max, skewness, kurtosis), a
correlation heatmap across mix-design features and both setting-time
targets, and per-feature distribution histograms. All figures are saved as
PNG so they can be embedded directly in the Word report.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def descriptive_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Mean, std, min, max, skewness, kurtosis for each numeric column."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        rows.append({
            "feature": col,
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "skewness": s.skew(),
            "kurtosis": s.kurt(),
        })
    return pd.DataFrame(rows)


def save_correlation_heatmap(df: pd.DataFrame, columns: list, out_path: str, title: str = "Correlation Heatmap"):
    """Pearson correlation heatmap across the given numeric columns."""
    corr = df[columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if abs(val) > 0.6 else "black", fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return corr


def save_distribution_grid(df: pd.DataFrame, columns: list, out_path: str, title: str = "Feature Distributions"):
    """Grid of histograms (with skewness annotated) for the given columns."""
    n = len(columns)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(columns):
        s = df[col].dropna()
        axes[i].hist(s, bins=10, color="#4C72B0", edgecolor="white")
        axes[i].set_title(f"{col}\nskew={s.skew():.2f}", fontsize=9)
        axes[i].tick_params(labelsize=7)
    for j in range(len(columns), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_correlations(corr: pd.DataFrame, target_cols: list, threshold: float = 0.4) -> pd.DataFrame:
    """Feature-target correlations above a threshold (absolute value), for narrative use."""
    rows = []
    for target in target_cols:
        if target not in corr.columns:
            continue
        for feat in corr.index:
            if feat == target or feat in target_cols:
                continue
            r = corr.loc[feat, target]
            if abs(r) >= threshold:
                rows.append({"feature": feat, "target": target, "pearson_r": r})
    return pd.DataFrame(rows).sort_values("pearson_r", key=np.abs, ascending=False)


def run_eda(df_valid: pd.DataFrame, feature_cols: list, target_cols: list, out_prefix: str = "aas_eda"):
    """
    Orchestrates the full EDA step: descriptive stats table, correlation
    heatmap (features + targets), distribution histograms, and a
    threshold-based summary of the strongest feature-target correlations.
    Returns a dict with the DataFrames produced (for use in the Word report).
    """
    all_cols = feature_cols + target_cols
    stats_df = descriptive_stats(df_valid, all_cols)
    stats_df.to_csv(f"{out_prefix}_descriptive_stats.csv", index=False)
    print(f"  [EDA] saved descriptive statistics -> {out_prefix}_descriptive_stats.csv")

    corr = save_correlation_heatmap(
        df_valid, all_cols, f"{out_prefix}_correlation_heatmap.png",
        title="Correlation Heatmap -- Mix Design Features & Setting Time"
    )
    corr.to_csv(f"{out_prefix}_correlation_matrix.csv")
    print(f"  [EDA] saved correlation heatmap -> {out_prefix}_correlation_heatmap.png")

    save_distribution_grid(
        df_valid, all_cols, f"{out_prefix}_distributions.png",
        title="Feature & Target Distributions (with skewness)"
    )
    print(f"  [EDA] saved distribution grid -> {out_prefix}_distributions.png")

    strong_corr = top_correlations(corr, target_cols, threshold=0.4)
    strong_corr.to_csv(f"{out_prefix}_strong_correlations.csv", index=False)
    print(f"  [EDA] saved strong feature-target correlations -> {out_prefix}_strong_correlations.csv")
    print("  [EDA] Strongest correlations (|r| >= 0.4):")
    print("  " + strong_corr.to_string(index=False).replace("\n", "\n  "))

    return {"stats": stats_df, "correlation": corr, "strong_correlations": strong_corr}

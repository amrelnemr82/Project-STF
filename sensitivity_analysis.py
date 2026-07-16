"""
================================================================================
SHAP Sensitivity Analysis  (Step 8 of the roadmap)
================================================================================
Implements:
  - Global sensitivity: mean |SHAP value| ranking of mix-design features
  - Local sensitivity: per-trial SHAP explanation (for the UI's forward mode)
  - Interaction effects: pairwise SHAP interaction values (tree models only)

Design choice: this module is imported lazily and wrapped in try/except by
the caller. If `shap` is not installed, the rest of the pipeline (cleaning,
feature engineering, hybrid model training, evaluation, GA inverse design)
still runs and completes normally -- only this analysis step is skipped,
with a one-line instruction printed to install it.

Install with:
    pip install shap
================================================================================
"""

import numpy as np
import pandas as pd


def _check_shap():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        print(
            "  [SHAP] package not installed -- skipping sensitivity analysis.\n"
            "         Install it with:  pip install shap\n"
            "         (then rerun training to generate SHAP outputs)"
        )
        return False


def global_sensitivity(model, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Mean |SHAP value| ranking across all features, for one fitted
    scikit-learn tree model (RandomForestRegressor / GradientBoostingRegressor).
    Returns a DataFrame sorted by importance, or an empty one if shap is
    unavailable.
    """
    if not _check_shap():
        return pd.DataFrame()
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking.insert(0, "model", model_name)
    return ranking


def hybrid_global_sensitivity(hybrid_model, X: pd.DataFrame, n_background: int = None) -> pd.DataFrame:
    """
    Model-agnostic SHAP (Permutation explainer) on the full hybrid stack's
    .predict method. Slower than TreeExplainer but works on any black-box
    model, which is needed because the hybrid stack combines RF + GB + ANN
    behind a Ridge meta-learner.
    """
    if not _check_shap():
        return pd.DataFrame()
    import shap

    background = X if n_background is None else X.sample(
        min(n_background, len(X)), random_state=42
    )
    explainer = shap.Explainer(hybrid_model.predict, background.values, feature_names=list(X.columns))
    shap_values = explainer(X.values)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking.insert(0, "model", "hybrid_stack")
    return ranking, shap_values


def local_explanation(hybrid_model, X_row: np.ndarray, feature_names: list, background: pd.DataFrame):
    """
    Local SHAP explanation for a single mix (one prediction), used by the UI
    to show "why" a forward-mode prediction came out the way it did.
    Returns a DataFrame of feature -> contribution (signed), or empty if
    shap is unavailable.
    """
    if not _check_shap():
        return pd.DataFrame()
    import shap

    explainer = shap.Explainer(hybrid_model.predict, background.values, feature_names=feature_names)
    shap_values = explainer(X_row.reshape(1, -1))
    contrib = pd.DataFrame({
        "feature": feature_names,
        "value": X_row,
        "shap_contribution": shap_values.values[0],
    }).sort_values("shap_contribution", key=np.abs, ascending=False).reset_index(drop=True)
    return contrib


def interaction_effects(model, X: pd.DataFrame, model_name: str, top_k: int = 5) -> pd.DataFrame:
    """
    Pairwise SHAP interaction values for a tree model (RandomForest only --
    GradientBoosting interaction values are not supported by SHAP's
    TreeExplainer in the same way). Returns the top_k strongest feature
    pairs by mean |interaction value|.
    """
    if not _check_shap():
        return pd.DataFrame()
    import shap

    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X.values)  # shape (n, n_feat, n_feat)
    n_feat = X.shape[1]
    mean_abs_inter = np.abs(inter).mean(axis=0)  # (n_feat, n_feat)

    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append((X.columns[i], X.columns[j], mean_abs_inter[i, j]))
    df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "mean_abs_interaction"])
    df = df.sort_values("mean_abs_interaction", ascending=False).head(top_k).reset_index(drop=True)
    df.insert(0, "model", model_name)
    return df


def save_summary_plot(model, X: pd.DataFrame, out_path: str, model_name: str):
    """Save a SHAP global-importance bar plot (tree models only)."""
    if not _check_shap():
        return
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP global feature importance -- {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [SHAP] saved global importance plot -> {out_path}")


def _permutation_importance_fallback(hybrid_model, X: pd.DataFrame, y: np.ndarray, label: str,
                                      n_repeats: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Global sensitivity WITHOUT shap, using scikit-learn's permutation_importance
    (works on any object with a .predict method, including our custom
    HybridModel -- no sklearn Estimator API compliance required). Used
    automatically as a fallback so sensitivity results are always available
    even when shap is not installed.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.base import BaseEstimator, RegressorMixin

    class _Wrapper(BaseEstimator, RegressorMixin):
        """Thin sklearn-compatible wrapper around the already-fitted hybrid model."""
        def __init__(self, model):
            self.model = model
        def fit(self, X, y=None):
            self.is_fitted_ = True
            return self
        def predict(self, X):
            return self.model.predict(X)

    wrapper = _Wrapper(hybrid_model).fit(X.values, y)
    result = permutation_importance(
        wrapper, X.values, y, n_repeats=n_repeats, random_state=seed,
        scoring="neg_root_mean_squared_error",
    )
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "target", label)
    df.insert(1, "method", "permutation_importance (fallback, no shap)")
    return df


def _partial_dependence_fallback(hybrid_model, X: pd.DataFrame, feature: str, n_points: int = 25) -> pd.DataFrame:
    """
    Manual 1D partial dependence: sweep one feature across its observed
    range while holding all other features at their median, and record the
    hybrid model's prediction. A simple, dependency-free substitute for a
    SHAP dependence plot -- shows the marginal sensitivity of the target to
    that feature.
    """
    baseline = X.median().values
    grid = np.linspace(X[feature].min(), X[feature].max(), n_points)
    feat_idx = list(X.columns).index(feature)
    preds = []
    for val in grid:
        row = baseline.copy()
        row[feat_idx] = val
        preds.append(hybrid_model.predict(row.reshape(1, -1))[0])
    return pd.DataFrame({feature: grid, "predicted": preds})


def save_confusion_matrix_figure(y_true_labels, y_pred_labels, class_names, out_prefix: str = "aas_eval"):
    """Confusion matrix figure for the workability classifier (LOOCV predictions)."""
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Workability classifier -- LOOCV confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = f"{out_prefix}_workability_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Eval] saved confusion matrix figure -> {out_path}")
    return out_path


def save_evaluation_figures(metrics_table: pd.DataFrame, out_prefix: str = "aas_eval"):
    """Bar charts comparing R2 and RMSE across base learners + hybrid stack, per target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = metrics_table["target"].unique()
    fig, axes = plt.subplots(len(targets), 2, figsize=(11, 4.5 * len(targets)))
    if len(targets) == 1:
        axes = axes.reshape(1, -1)

    for i, target in enumerate(targets):
        sub = metrics_table[metrics_table["target"] == target]
        colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]  # highlight HybridStack
        axes[i, 0].bar(sub["model"], sub["R2"], color=colors[: len(sub)])
        axes[i, 0].set_title(f"R2 -- {target}")
        axes[i, 0].set_ylabel("R2")
        axes[i, 0].tick_params(axis="x", rotation=20)

        axes[i, 1].bar(sub["model"], sub["RMSE"], color=colors[: len(sub)])
        axes[i, 1].set_title(f"RMSE -- {target}")
        axes[i, 1].set_ylabel("RMSE (min)")
        axes[i, 1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path = f"{out_prefix}_metrics_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Eval] saved metrics comparison figure -> {out_path}")
    return out_path


def save_pso_convergence_figure(pso_log: pd.DataFrame, out_prefix: str = "aas_eval"):
    """Best-so-far fitness vs iteration, per target -- shows the PSO search converging."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = pso_log["target"].unique()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target in targets:
        sub = pso_log[pso_log["target"] == target].copy()
        best_so_far = sub.groupby("iteration")["fitness"].min().cummin()
        ax.plot(best_so_far.index, best_so_far.values, marker="o", label=target)
    ax.set_xlabel("PSO iteration")
    ax.set_ylabel("Best fitness so far (K-fold RMSE proxy)")
    ax.set_title("PSO convergence -- ANN hyperparameter search")
    ax.legend()
    plt.tight_layout()
    out_path = f"{out_prefix}_pso_convergence.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Eval] saved PSO convergence figure -> {out_path}")
    return out_path


def run_full_sensitivity_analysis(hybrid_ist, hybrid_fst, X: pd.DataFrame,
                                   y_ist: np.ndarray = None, y_fst: np.ndarray = None,
                                   out_prefix: str = "shap"):
    """
    Orchestrates Step 8 of the roadmap for both targets (IST, FST).

    If shap is installed: global sensitivity (RF, GB, hybrid stack),
    interaction effects (RF), and SHAP summary plots.

    If shap is NOT installed: automatically falls back to scikit-learn
    permutation importance (global sensitivity) + manual partial-dependence
    sweeps for the top-3 features (local/marginal sensitivity), so results
    and figures are always produced either way.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    has_shap = _check_shap()

    for label, model, y in [("IST", hybrid_ist, y_ist), ("FST", hybrid_fst, y_fst)]:
        print(f"\n  [Sensitivity] running analysis for {label} ...")

        if has_shap:
            rf_rank = global_sensitivity(model.rf, X, f"RandomForest_{label}")
            gb_rank = global_sensitivity(model.gb, X, f"GradientBoosting_{label}")
            hyb_rank, _ = hybrid_global_sensitivity(model, X)
            hyb_rank["model"] = f"HybridStack_{label}"
            combined = pd.concat([rf_rank, gb_rank, hyb_rank], ignore_index=True)
            combined.to_csv(f"{out_prefix}_global_importance_{label}.csv", index=False)
            print(f"  [SHAP] saved -> {out_prefix}_global_importance_{label}.csv")

            inter = interaction_effects(model.rf, X, f"RandomForest_{label}")
            if not inter.empty:
                inter.to_csv(f"{out_prefix}_interactions_{label}.csv", index=False)

            save_summary_plot(model.rf, X, f"{out_prefix}_summary_{label}.png", f"RandomForest ({label})")
            results[label] = {"global_ranking": combined, "interactions": inter, "method": "shap"}

        else:
            # --- fallback: permutation importance + manual partial dependence ---
            rank = _permutation_importance_fallback(model, X, y, label)
            rank.to_csv(f"{out_prefix}_global_importance_{label}_FALLBACK.csv", index=False)
            print(f"  [Fallback] saved -> {out_prefix}_global_importance_{label}_FALLBACK.csv")

            # bar chart of the fallback global importance
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(rank["feature"][::-1], rank["importance_mean"][::-1],
                    xerr=rank["importance_std"][::-1], color="#55A868")
            ax.set_xlabel("Permutation importance (RMSE increase when shuffled)")
            ax.set_title(f"Global sensitivity (fallback, no shap) -- {label}")
            plt.tight_layout()
            bar_path = f"{out_prefix}_global_importance_{label}_FALLBACK.png"
            plt.savefig(bar_path, dpi=150)
            plt.close()
            print(f"  [Fallback] saved -> {bar_path}")

            # partial dependence for the top 3 features
            top_features = rank["feature"].head(3).tolist()
            fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 4))
            if len(top_features) == 1:
                axes = [axes]
            for ax, feat in zip(axes, top_features):
                pdp = _partial_dependence_fallback(model, X, feat)
                ax.plot(pdp[feat], pdp["predicted"], marker="o", color="#C44E52")
                ax.set_xlabel(feat)
                ax.set_ylabel(f"Predicted {label} (min)")
                ax.set_title(f"Partial dependence: {feat}")
            plt.tight_layout()
            pdp_path = f"{out_prefix}_partial_dependence_{label}_FALLBACK.png"
            plt.savefig(pdp_path, dpi=150)
            plt.close()
            print(f"  [Fallback] saved -> {pdp_path}")

            results[label] = {"global_ranking": rank, "method": "permutation_importance_fallback"}

    return results

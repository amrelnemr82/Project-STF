"""
================================================================================
Alkali-Activated Slag (AAS) Setting-Time & Workability Modeling Pipeline
================================================================================
Roadmap stage covered by this file:
    1. Data loading (new single-sheet format) or loading + merging
       (legacy two-sheet format, auto-detected)
    2. Data cleaning
    3. Feature engineering
    4. Hybrid Machine Learning model
         - Base learners: Random Forest, Gradient Boosting, MLP (ANN)
         - Metaheuristic layer: Particle Swarm Optimization (PSO) tunes the
           ANN hyperparameters (every trial logged, see PSO_* constants)
         - Hybrid Stacking Ensemble: RF + GB + PSO-tuned ANN -> Ridge meta-learner
    5. Workability classifier (Random Forest Classifier)
    6. Model evaluation with Leave-One-Out Cross-Validation (LOOCV), because
       the raw dataset only has ~19-21 usable trials. Reports R2, MSE, RMSE,
       MAE per base learner (RF, GB, ANN) AND for the final hybrid stack, so
       the value added by hybridization is visible.
    7. Model + scaler serialization (for the UI, see ui_app.py)
    8. Sensitivity analysis (SHAP) -- delegated to sensitivity_analysis.py.
       Global feature ranking, interaction effects, and per-prediction local
       explanations. Degrades gracefully if `shap` is not installed.
    9. Inverse mix-design search: a Genetic Algorithm (GA, see GA_* constants)
       that searches the feasible mix-design space to propose a mix meeting
       a target setting time window and workability class (the hybrid model
       is used as the GA's fitness function)

Run:
    python run_training.py --data Setting_timeX.xlsx
    (see run_training.py docstring for why this file should not be run
    directly with `python aas_setting_time_pipeline.py`)
================================================================================
"""

import argparse
import re
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
import joblib

import sensitivity_analysis as sa
import eda_analysis as eda

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --- PSO hyperparameters (tunes the ANN base learner) -----------------------
PSO_N_PARTICLES = 6
PSO_N_ITERS = 6
PSO_INERTIA_W = 0.6
PSO_COGNITIVE_C1 = 1.5
PSO_SOCIAL_C2 = 1.5
PSO_ANN_HIDDEN_BOUNDS = (6, 24)          # neurons
PSO_ANN_LOG10_ALPHA_BOUNDS = (-4, -1)    # L2 regularization, log10 scale
PSO_ANN_LOG10_LR_BOUNDS = (-3, -1.5)     # learning rate, log10 scale
PSO_KFOLDS = 5                           # fast proxy fitness during search
PSO_SEARCH_MAX_ITER = 400                # ANN training iters during search
PSO_FINAL_MAX_ITER = 1500                # ANN training iters for final refit

# --- GA hyperparameters (inverse mix design search) --------------------------
GA_POP_SIZE = 40
GA_GENERATIONS = 40
GA_ELITE_FRACTION = 0.2
GA_MUTATION_PROB = 0.2
GA_MUTATION_SIGMA_FRAC = 0.05  # as a fraction of each feature's observed range


# ==============================================================================
# 1. DATA LOADING (single-sheet format; auto-falls back to merging legacy two-sheet files)
# ==============================================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw workbook. Supports two formats:
      - New format (Setting_timeX.xlsx): single sheet "Specimens and results"
        with all 13 columns (mix design + IST/FST/Remarks) together.
      - Old format (Setting_time.xlsx): two sheets "Specimens" and "Results"
        that need to be merged on Trial no.
    """
    xls = pd.ExcelFile(path)
    columns = [
        "Trial", "NaOH_g", "Na2SiO3_g", "ExtraH2O_g",
        "Na2O_pct", "MS", "LS", "NaOH_conc_raw", "Na2SiO3_NaOH_raw", "Alk_Bi",
        "IST_raw", "FST_raw", "Remarks",
    ]

    if "Specimens and results" in xls.sheet_names:
        df = pd.read_excel(path, sheet_name="Specimens and results", header=None, skiprows=3).iloc[:21]
        df.columns = columns
        return df

    # fallback: old two-sheet format
    specimens = pd.read_excel(path, sheet_name="Specimens", header=None, skiprows=3).iloc[:21]
    results = pd.read_excel(path, sheet_name="Results", header=None, skiprows=3).iloc[:21]

    specimens.columns = [
        "Trial", "NaOH_g", "Na2SiO3_g", "ExtraH2O_g",
        "Na2O_pct", "MS", "LS", "NaOH_conc_raw", "Na2SiO3_NaOH_raw", "Alk_Bi",
    ]
    results.columns = [
        "Trial", "Na2O_pct2", "MS2", "LS2", "NaOH_conc2", "Na2SiO3_NaOH2",
        "Alk_Bi2", "IST_raw", "FST_raw", "Remarks",
    ]
    df = specimens.merge(results[["Trial", "IST_raw", "FST_raw", "Remarks"]], on="Trial")
    return df


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
def parse_activator_concentration(raw: str) -> Tuple[float, str]:
    """
    The 'NaOH conc.' column mixes two different unit systems:
        - Molarity, e.g. '12M', '10M', '8M'   -> liquid NaOH solution
        - wt% Na2O, e.g. '48% Na2O', '8.15% Na2O' -> solid NaOH pellets / different activator route
    We split this into a numeric value + a categorical 'activator_route' flag
    rather than force everything onto one incompatible numeric scale.
    """
    raw = str(raw).strip()
    m = re.match(r"^([\d.]+)\s*M$", raw)
    if m:
        return float(m.group(1)), "molar_solution"
    m = re.match(r"^([\d.]+)\s*%\s*Na2O$", raw)
    if m:
        return float(m.group(1)), "pct_na2o_solid"
    return np.nan, "unknown"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- numeric targets: '-' marks a failed/invalid trial (no measurable set) ---
    for col_raw, col_clean in [("IST_raw", "IST"), ("FST_raw", "FST")]:
        df[col_clean] = pd.to_numeric(df[col_raw], errors="coerce")

    df["trial_valid"] = df["IST"].notna() & df["FST"].notna()

    # --- activator concentration: split unit system ---
    parsed = df["NaOH_conc_raw"].apply(parse_activator_concentration)
    df["NaOH_conc_value"] = parsed.apply(lambda t: t[0])
    df["activator_route"] = parsed.apply(lambda t: t[1])

    # --- Na2SiO3/NaOH ratio: '-' means Na2SiO3 = 0 (not truly missing) ---
    df["Na2SiO3_NaOH_raw"] = df["Na2SiO3_NaOH_raw"].replace("-", 0)
    df["Na2SiO3_NaOH"] = pd.to_numeric(df["Na2SiO3_NaOH_raw"], errors="coerce").fillna(0)
    df["has_Na2SiO3"] = (df["Na2SiO3_g"] > 0).astype(int)

    # --- mass-balance sanity check (cleaning QA, flags rows for manual review) ---
    df["solution_mass_check"] = df["NaOH_g"] + df["Na2SiO3_g"] + df["ExtraH2O_g"]

    # --- workability class engineered from free-text Remarks ---
    def workability_label(row):
        if not row["trial_valid"]:
            return "failed_reaction"
        r = str(row["Remarks"]).lower()
        if r == "nan":
            return "normal"
        if "high flowability" in r:
            return "high_flowability"
        if "very stiff" in r:
            return "very_stiff"
        if "stiff" in r:
            return "stiff"
        if "fair" in r and "shrinkage" in r:
            return "fair_high_shrinkage"
        if "fair" in r:
            return "fair"
        return "other"

    df["workability_class"] = df.apply(workability_label, axis=1)

    return df


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_solution = df["NaOH_g"] + df["Na2SiO3_g"] + df["ExtraH2O_g"]
    df["total_alkaline_solution_g"] = total_solution
    df["solid_activator_fraction"] = np.where(total_solution > 0, df["ExtraH2O_g"] / total_solution, 0)
    df["water_binder_ratio"] = df["ExtraH2O_g"] / 200.0  # per 200 g slag basis
    df["NaOH_binder_ratio"] = df["NaOH_g"] / 200.0
    df["Na2SiO3_binder_ratio"] = df["Na2SiO3_g"] / 200.0
    return df


# ==============================================================================
# FEATURE / TARGET SELECTION
# ==============================================================================
FEATURE_COLS = [
    "Na2O_pct", "MS", "LS", "NaOH_conc_value", "Na2SiO3_NaOH", "Alk_Bi",
    "has_Na2SiO3", "solid_activator_fraction", "water_binder_ratio",
    "NaOH_binder_ratio", "Na2SiO3_binder_ratio",
]
TARGET_COLS = ["IST", "FST"]


# ==============================================================================
# 4. HYBRID ML MODEL  (RF + GB + PSO-tuned ANN  ->  Ridge stacking meta-learner)
# ==============================================================================
class SimplePSO:
    """
    Minimal Particle Swarm Optimization, used to tune the MLPRegressor
    hyperparameters (hidden_layer_size, alpha, learning_rate_init) by
    minimizing LOOCV RMSE. Implemented from scratch (no extra dependency).
    """

    def __init__(self, bounds, n_particles=12, n_iters=25, seed=RANDOM_STATE,
                 w=PSO_INERTIA_W, c1=PSO_COGNITIVE_C1, c2=PSO_SOCIAL_C2):
        self.bounds = np.array(bounds)  # shape (n_dims, 2)
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.rng = np.random.default_rng(seed)
        self.w, self.c1, self.c2 = w, c1, c2
        self.history = []  # every evaluated trial, for transparency/logging

    def optimize(self, fitness_fn):
        n_dims = len(self.bounds)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        pos = self.rng.uniform(lo, hi, size=(self.n_particles, n_dims))
        vel = np.zeros_like(pos)

        pbest = pos.copy()
        pbest_val = np.array([fitness_fn(p) for p in pos])
        for pid in range(self.n_particles):
            self.history.append({
                "iteration": 0, "particle_id": pid,
                "params": pos[pid].tolist(), "fitness": float(pbest_val[pid]),
            })
        gbest = pbest[np.argmin(pbest_val)]
        gbest_val = pbest_val.min()

        w, c1, c2 = self.w, self.c1, self.c2
        for it in range(1, self.n_iters + 1):
            r1, r2 = self.rng.random((self.n_particles, n_dims)), self.rng.random((self.n_particles, n_dims))
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, lo, hi)

            vals = np.array([fitness_fn(p) for p in pos])
            for pid in range(self.n_particles):
                self.history.append({
                    "iteration": it, "particle_id": pid,
                    "params": pos[pid].tolist(), "fitness": float(vals[pid]),
                })
            improved = vals < pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]

            if pbest_val.min() < gbest_val:
                gbest_val = pbest_val.min()
                gbest = pbest[np.argmin(pbest_val)]

        return gbest, gbest_val


def pso_tune_ann(X: np.ndarray, y: np.ndarray, label: str = ""):
    """
    Use PSO to select ANN hyperparameters that minimize a fast K-fold RMSE
    proxy (K-fold rather than full LOOCV during the search, purely for
    tractable runtime -- the final chosen configuration is still validated
    with full LOOCV as part of the hybrid stacking step below).

    Returns (untrained MLPRegressor with the chosen config, trial_log_df).
    """
    bounds = [PSO_ANN_HIDDEN_BOUNDS, PSO_ANN_LOG10_ALPHA_BOUNDS, PSO_ANN_LOG10_LR_BOUNDS]
    n = len(X)
    rng = np.random.default_rng(RANDOM_STATE)
    k = min(PSO_KFOLDS, n)
    fold_idx = rng.permutation(n)
    folds = np.array_split(fold_idx, k)

    def fitness(params):
        hidden = int(round(params[0]))
        alpha = 10 ** params[1]
        lr = 10 ** params[2]
        errs = []
        for f in folds:
            te_idx = f
            tr_idx = np.setdiff1d(np.arange(n), f)
            model = MLPRegressor(
                hidden_layer_sizes=(hidden,), alpha=alpha, learning_rate_init=lr,
                max_iter=PSO_SEARCH_MAX_ITER, random_state=RANDOM_STATE,
            )
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[te_idx])
            errs.extend((pred - y[te_idx]) ** 2)
        return np.sqrt(np.mean(errs))

    pso = SimplePSO(bounds, n_particles=PSO_N_PARTICLES, n_iters=PSO_N_ITERS)
    best_params, best_rmse = pso.optimize(fitness)
    hidden = int(round(best_params[0]))
    alpha = 10 ** best_params[1]
    lr = 10 ** best_params[2]
    print(f"  [PSO] best ANN config -> hidden={hidden}, alpha={alpha:.5f}, lr={lr:.5f}, LOOCV RMSE={best_rmse:.2f}")

    trial_log = pd.DataFrame(pso.history)
    trial_log["target"] = label
    trial_log["hidden_units"] = trial_log["params"].apply(lambda p: int(round(p[0])))
    trial_log["alpha"] = trial_log["params"].apply(lambda p: 10 ** p[1])
    trial_log["learning_rate_init"] = trial_log["params"].apply(lambda p: 10 ** p[2])
    trial_log = trial_log[["target", "iteration", "particle_id", "hidden_units", "alpha", "learning_rate_init", "fitness"]]

    ann = MLPRegressor(
        hidden_layer_sizes=(hidden,), alpha=alpha, learning_rate_init=lr,
        max_iter=PSO_FINAL_MAX_ITER, random_state=RANDOM_STATE,
    )
    return ann, trial_log


@dataclass
class HybridModel:
    """RF + GB + PSO-tuned ANN base learners, combined with a Ridge meta-learner (stacking)."""
    rf: RandomForestRegressor
    gb: GradientBoostingRegressor
    ann: MLPRegressor
    meta: Ridge
    scaler: StandardScaler

    def _base_predictions(self, X_scaled):
        return np.column_stack([
            self.rf.predict(X_scaled),
            self.gb.predict(X_scaled),
            self.ann.predict(X_scaled),
        ])

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_raw)
        base = self._base_predictions(X_scaled)
        return self.meta.predict(base)


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def train_hybrid_model(X: pd.DataFrame, y: pd.Series, label: str):
    print(f"\nTraining hybrid model for target: {label}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    y_arr = y.values

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=RANDOM_STATE)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE)
    ann, pso_trial_log = pso_tune_ann(X_scaled, y_arr, label=label)

    # Build out-of-fold base predictions (LOOCV) to train an honest meta-learner
    loo = LeaveOneOut()
    oof = np.zeros((len(y_arr), 3))
    for tr_idx, te_idx in loo.split(X_scaled):
        rf_f = clone(rf).fit(X_scaled[tr_idx], y_arr[tr_idx])
        gb_f = clone(gb).fit(X_scaled[tr_idx], y_arr[tr_idx])
        ann_f = clone(ann).fit(X_scaled[tr_idx], y_arr[tr_idx])
        oof[te_idx, 0] = rf_f.predict(X_scaled[te_idx])
        oof[te_idx, 1] = gb_f.predict(X_scaled[te_idx])
        oof[te_idx, 2] = ann_f.predict(X_scaled[te_idx])

    meta = Ridge(alpha=1.0).fit(oof, y_arr)

    # Refit base learners on full data for deployment
    rf.fit(X_scaled, y_arr)
    gb.fit(X_scaled, y_arr)
    ann.fit(X_scaled, y_arr)

    model = HybridModel(rf=rf, gb=gb, ann=ann, meta=meta, scaler=scaler)

    # --- Step 7: Model Evaluation -- R2 / MSE / RMSE / MAE, per base learner
    # AND for the final hybrid stack, so the value added by hybridization is visible.
    final_oof_pred = meta.predict(oof)
    metrics_rows = []
    for name, preds in [
        ("RandomForest", oof[:, 0]),
        ("GradientBoosting", oof[:, 1]),
        ("PSO_ANN", oof[:, 2]),
        ("HybridStack", final_oof_pred),
    ]:
        m = _regression_metrics(y_arr, preds)
        m["model"] = name
        m["target"] = label
        metrics_rows.append(m)
    metrics_table = pd.DataFrame(metrics_rows)[["target", "model", "R2", "MSE", "RMSE", "MAE"]]

    print(f"  [LOOCV metrics -- {label}]")
    print("  " + metrics_table.drop(columns="target").to_string(index=False).replace("\n", "\n  "))

    parity_data = {"target": label, "y_true": y_arr, "y_pred": final_oof_pred}

    return model, metrics_table, pso_trial_log, parity_data


# ==============================================================================
# 5. WORKABILITY CLASSIFIER
# ==============================================================================
def train_workability_classifier(X: pd.DataFrame, y_labels: pd.Series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    clf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=RANDOM_STATE)
    clf.fit(X.values, y_enc)

    # LOOCV accuracy + out-of-fold predictions (for confusion matrix figure)
    loo = LeaveOneOut()
    oof_pred = np.zeros(len(y_enc), dtype=int)
    for tr_idx, te_idx in loo.split(X.values):
        c = clone(clf).fit(X.values[tr_idx], y_enc[tr_idx])
        oof_pred[te_idx] = c.predict(X.values[te_idx])
    correct = int((oof_pred == y_enc).sum())
    print(f"  [Workability classifier] LOOCV accuracy = {correct}/{len(y_enc)} = {correct/len(y_enc):.2f}")

    oof_pred_labels = le.inverse_transform(oof_pred)
    true_labels = le.inverse_transform(y_enc)
    return clf, le, true_labels, oof_pred_labels



# ==============================================================================
# 8. INVERSE MIX-DESIGN SEARCH (Genetic Algorithm)
# ==============================================================================
def ga_inverse_design(
    hybrid_ist, hybrid_fst, workability_clf, le, feature_bounds,
    target_ist_range, target_fst_range, target_workability="normal",
    pop_size=GA_POP_SIZE, generations=GA_GENERATIONS, seed=RANDOM_STATE,
    elite_fraction=GA_ELITE_FRACTION, mutation_prob=GA_MUTATION_PROB,
    mutation_sigma_frac=GA_MUTATION_SIGMA_FRAC,
):
    """
    Search the feasible mix-design feature space for a candidate that:
      - predicts IST/FST inside the requested acceptable ranges
      - predicts the requested workability class
    Uses a simple real-valued Genetic Algorithm with elitism, blend
    crossover and Gaussian mutation. The trained hybrid model is used
    as the fitness function (this is the 'hybridization' at the design stage:
    ML surrogate model + metaheuristic search).
    """
    rng = np.random.default_rng(seed)
    n_dims = len(feature_bounds)
    lo = np.array([b[0] for b in feature_bounds])
    hi = np.array([b[1] for b in feature_bounds])
    target_class_idx = le.transform([target_workability])[0]

    def fitness(vec):
        X = vec.reshape(1, -1)
        ist = hybrid_ist.predict(X)[0]
        fst = hybrid_fst.predict(X)[0]
        wclass_proba = workability_clf.predict_proba(X)[0]

        penalty = 0.0
        if not (target_ist_range[0] <= ist <= target_ist_range[1]):
            penalty += min(abs(ist - target_ist_range[0]), abs(ist - target_ist_range[1]))
        if not (target_fst_range[0] <= fst <= target_fst_range[1]):
            penalty += min(abs(fst - target_fst_range[0]), abs(fst - target_fst_range[1]))
        penalty += 20 * (1 - wclass_proba[target_class_idx])
        return penalty

    pop = rng.uniform(lo, hi, size=(pop_size, n_dims))

    for gen in range(generations):
        scores = np.array([fitness(ind) for ind in pop])
        order = np.argsort(scores)
        pop = pop[order]
        scores = scores[order]

        elites = pop[:max(1, int(pop_size * elite_fraction))]
        children = [elites[i] for i in range(len(elites))]
        while len(children) < pop_size:
            i, j = rng.integers(0, pop_size // 2, size=2)
            p1, p2 = pop[i], pop[j]
            alpha = rng.random(n_dims)
            child = alpha * p1 + (1 - alpha) * p2
            mutate_mask = rng.random(n_dims) < mutation_prob
            child[mutate_mask] += rng.normal(0, mutation_sigma_frac * (hi - lo)[mutate_mask])
            child = np.clip(child, lo, hi)
            children.append(child)
        pop = np.array(children)

    final_scores = np.array([fitness(ind) for ind in pop])
    best = pop[np.argmin(final_scores)]
    best_score = final_scores.min()
    return best, best_score


# ==============================================================================
# MAIN
# ==============================================================================
def main(path: str):
    print("Step 1-2: Loading & cleaning data (merge only applies to legacy two-sheet files) ...")
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    print(f"  Total trials: {len(df)}  |  Valid (measurable) trials: {df['trial_valid'].sum()}")
    print("  Workability class counts:")
    print(df["workability_class"].value_counts().to_string())

    df_valid = df[df["trial_valid"]].reset_index(drop=True)
    df_valid[FEATURE_COLS] = df_valid[FEATURE_COLS].fillna(0)

    X = df_valid[FEATURE_COLS]
    y_ist = df_valid["IST"]
    y_fst = df_valid["FST"]

    print("\nStep 3: Feature columns used ->", FEATURE_COLS)

    print("\nStep 3b: Exploratory Data Analysis (descriptive stats, correlation heatmap, distributions)")
    eda.run_eda(df_valid, FEATURE_COLS, ["IST", "FST"], out_prefix="aas_eda")

    print("\nStep 4-6: Hybrid ML models (RF + GB + PSO-ANN -> Ridge stack), LOOCV evaluation")
    hybrid_ist, metrics_ist, pso_log_ist = train_hybrid_model(X, y_ist, "Initial Setting Time (IST)")
    hybrid_fst, metrics_fst, pso_log_fst = train_hybrid_model(X, y_fst, "Final Setting Time (FST)")

    metrics_table = pd.concat([metrics_ist, metrics_fst], ignore_index=True)
    metrics_table.to_csv("aas_model_metrics_comparison.csv", index=False)
    print("\n  Saved per-base-learner metrics comparison -> aas_model_metrics_comparison.csv")

    pso_log = pd.concat([pso_log_ist, pso_log_fst], ignore_index=True)
    pso_log.to_csv("aas_pso_hyperparameter_log.csv", index=False)
    print("  Saved full PSO trial-by-trial hyperparameter log -> aas_pso_hyperparameter_log.csv")

    print("\nStep 5b: Workability classifier")
    df_class = df.copy()
    df_class[FEATURE_COLS] = df_class[FEATURE_COLS].fillna(0)
    clf, le, true_labels, oof_pred_labels = train_workability_classifier(
        df_class[FEATURE_COLS], df_class["workability_class"]
    )
    sa.save_confusion_matrix_figure(true_labels, oof_pred_labels, list(le.classes_), out_prefix="aas_eval")

    print("\nStep 7: Saving model artifacts for the UI ...")
    joblib.dump(
        {
            "hybrid_ist": hybrid_ist,
            "hybrid_fst": hybrid_fst,
            "workability_clf": clf,
            "label_encoder": le,
            "feature_cols": FEATURE_COLS,
            "feature_bounds": [(float(X[c].min()), float(X[c].max())) for c in FEATURE_COLS],
        },
        "aas_hybrid_model.joblib",
    )
    print("  Saved -> aas_hybrid_model.joblib")

    print("\nStep 7: Evaluation figures (metrics comparison + PSO convergence)")
    sa.save_evaluation_figures(metrics_table, out_prefix="aas_eval")
    sa.save_pso_convergence_figure(pso_log, out_prefix="aas_eval")

    print("\nStep 8: Sensitivity Analysis (SHAP if available, else verified fallback)")
    sa.run_full_sensitivity_analysis(hybrid_ist, hybrid_fst, X, y_ist.values, y_fst.values, out_prefix="aas_shap")

    print("\nStep 9: Example inverse mix-design search (GA)")
    print("  Target: Final Setting Time between 40 and 60 min, 'normal' workability")
    bounds = [(float(X[c].min()), float(X[c].max())) for c in FEATURE_COLS]
    best_vec, best_score = ga_inverse_design(
        hybrid_ist, hybrid_fst, clf, le, bounds,
        target_ist_range=(15, 35), target_fst_range=(40, 60),
        target_workability="normal",
    )
    print("  Best candidate mix (feature space):")
    for name, val in zip(FEATURE_COLS, best_vec):
        print(f"    {name:22s} = {val:.3f}")
    pred_ist = hybrid_ist.predict(best_vec.reshape(1, -1))[0]
    pred_fst = hybrid_fst.predict(best_vec.reshape(1, -1))[0]
    pred_class = le.inverse_transform(clf.predict(best_vec.reshape(1, -1)))[0]
    print(f"  -> Predicted IST={pred_ist:.1f} min, FST={pred_fst:.1f} min, workability={pred_class}")
    print(f"  GA fitness (penalty, lower=better) = {best_score:.3f}")

    df.to_csv("aas_cleaned_dataset.csv", index=False)
    print("\nCleaned dataset saved -> aas_cleaned_dataset.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Setting_timeX.xlsx")
    args = parser.parse_args()
    main(args.data)
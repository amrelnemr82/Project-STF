"""
================================================================================
AAS Mix-Design UI  (Streamlit)
================================================================================
Loads the hybrid model artifacts produced by aas_setting_time_pipeline.py
(aas_hybrid_model.joblib) and offers three tabs:

  FORWARD MODE  - user enters a mix design -> app predicts Initial/Final
                  Setting Time and the workability class, with an optional
                  SHAP local explanation of "why" (if shap is installed).

  INVERSE MODE  - user enters a target setting-time window + desired
                  workability -> app runs the same Genetic-Algorithm search
                  used in the pipeline to propose a feasible mix.

  MODEL INFO    - shows the LOOCV metrics comparison (R2/MSE/RMSE/MAE) for
                  each base learner vs the hybrid stack, from
                  aas_model_metrics_comparison.csv (if present).

Run:
    streamlit run ui_app.py
================================================================================
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from aas_setting_time_pipeline import ga_inverse_design, FEATURE_COLS

st.set_page_config(page_title="AAS Mix Design Assistant", layout="centered")

MODEL_PATH = "aas_hybrid_model.joblib"
CLEANED_DATA_PATH = "aas_cleaned_dataset.csv"
METRICS_PATH = "aas_model_metrics_comparison.csv"

try:
    import shap  # noqa: F401
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_background():
    """Cleaned training data, used as the SHAP explainer's background set."""
    if not os.path.exists(CLEANED_DATA_PATH):
        return None
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = df[df["trial_valid"]].reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df[FEATURE_COLS]


artifacts = load_artifacts()
hybrid_ist = artifacts["hybrid_ist"]
hybrid_fst = artifacts["hybrid_fst"]
clf = artifacts["workability_clf"]
le = artifacts["label_encoder"]
feature_cols = artifacts["feature_cols"]
feature_bounds = artifacts["feature_bounds"]
background_df = load_background()

st.title("Alkali-Activated Slag (AAS) Mix Design Assistant")
st.caption(
    "Hybrid ML model (Random Forest + Gradient Boosting + PSO-tuned ANN, "
    "stacked via a Ridge meta-learner) trained on lab setting-time trials, "
    "with a Genetic-Algorithm search for inverse mix design."
)
st.warning(
    "Trained on only ~19 lab trials — treat predictions as directional "
    "guidance for narrowing lab trials, not as a certified design tool."
)

tab_forward, tab_inverse, tab_info = st.tabs(["Forward: predict", "Inverse: design a mix", "Model info"])

with tab_forward:
    st.subheader("Enter mix design parameters")
    values = []
    cols = st.columns(2)
    for i, feat in enumerate(feature_cols):
        lo, hi = feature_bounds[i]
        default = (lo + hi) / 2
        with cols[i % 2]:
            v = st.number_input(feat, min_value=float(lo), max_value=float(hi), value=float(default))
        values.append(v)

    if st.button("Predict setting time & workability"):
        X = np.array(values).reshape(1, -1)
        ist = hybrid_ist.predict(X)[0]
        fst = hybrid_fst.predict(X)[0]
        wclass = le.inverse_transform(clf.predict(X))[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Setting Time", f"{ist:.0f} min")
        c2.metric("Final Setting Time", f"{fst:.0f} min")
        c3.metric("Predicted workability", wclass)

        if SHAP_AVAILABLE and background_df is not None:
            import sensitivity_analysis as sa
            st.write("**Why this prediction (SHAP local explanation, Final Setting Time):**")
            contrib = sa.local_explanation(hybrid_fst, np.array(values), feature_cols, background_df)
            if not contrib.empty:
                st.dataframe(
                    contrib.style.format({"value": "{:.3f}", "shap_contribution": "{:+.2f}"}),
                    hide_index=True,
                )
                st.caption("Positive = pushes FST higher (slower set). Negative = pushes FST lower (faster set).")
        elif not SHAP_AVAILABLE:
            st.caption("Install `shap` (pip install shap) to see a feature-by-feature explanation here.")

with tab_inverse:
    st.subheader("Enter your acceptable setting-time window")
    c1, c2 = st.columns(2)
    with c1:
        ist_lo = st.number_input("Min Initial Setting Time (min)", value=15.0)
        fst_lo = st.number_input("Min Final Setting Time (min)", value=40.0)
    with c2:
        ist_hi = st.number_input("Max Initial Setting Time (min)", value=35.0)
        fst_hi = st.number_input("Max Final Setting Time (min)", value=60.0)

    workability_choice = st.selectbox("Desired workability", list(le.classes_))

    if st.button("Search for a feasible mix"):
        with st.spinner("Running GA search over the hybrid model..."):
            best_vec, score = ga_inverse_design(
                hybrid_ist, hybrid_fst, clf, le, feature_bounds,
                target_ist_range=(ist_lo, ist_hi),
                target_fst_range=(fst_lo, fst_hi),
                target_workability=workability_choice,
            )
        X = best_vec.reshape(1, -1)
        pred_ist = hybrid_ist.predict(X)[0]
        pred_fst = hybrid_fst.predict(X)[0]
        pred_class = le.inverse_transform(clf.predict(X))[0]

        st.success(f"Best candidate found (fit score {score:.2f}, lower = better match)")
        st.write("**Suggested mix-design features (per 200 g slag basis):**")
        st.table({feat: [round(v, 3)] for feat, v in zip(feature_cols, best_vec)})

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted IST", f"{pred_ist:.0f} min")
        c2.metric("Predicted FST", f"{pred_fst:.0f} min")
        c3.metric("Predicted workability", pred_class)

        st.caption(
            "Note: 'has_Na2SiO3' is a binary flag — round it to 0 or 1 before "
            "casting a real trial. Always validate a suggested mix with an "
            "actual lab trial before use."
        )

with tab_info:
    st.subheader("Model evaluation (LOOCV)")
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.write("R2 / MSE / RMSE / MAE per base learner vs the hybrid stack:")
        st.dataframe(
            metrics_df.style.format({"R2": "{:.3f}", "MSE": "{:.1f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}),
            hide_index=True,
        )
        st.caption(
            "HybridStack should outperform (or match) every individual base "
            "learner — that's the empirical justification for hybridization."
        )
    else:
        st.info(f"Run `python run_training.py` first to generate {METRICS_PATH}.")

    st.divider()
    st.subheader("Optimizer hyperparameters used")
    st.markdown(
        """
        **PSO** (tunes the ANN base learner): 6 particles × 6 iterations,
        inertia w=0.6, cognitive c1=1.5, social c2=1.5, search space:
        hidden units 6–24, alpha 1e-4–1e-1 (log), learning rate 1e-3–3.16e-2 (log).

        **GA** (inverse mix design): population 40, generations 40,
        elitism top 20%, blend crossover, Gaussian mutation
        (20% gene-wise probability, σ = 5% of feature range).
        """
    )

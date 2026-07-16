# AAS Mix Design Assistant

Hybrid ML app (Random Forest + Gradient Boosting + PSO-tuned ANN, stacked
with a Ridge meta-learner) for predicting setting time and workability of
alkali-activated slag (AAS) mixes, with exploratory data analysis, SHAP
(or verified fallback) sensitivity analysis, and a Genetic-Algorithm
inverse mix designer.

**Note on model quality:** trained on 19 usable lab trials (21 total, 2
failed to set). Good for narrowing down which mixes to test next in the
lab, not a certified design tool.

See **AAS_Project_Report.docx** for the full write-up: data description
with figures (correlation heatmap, distributions, skewness), the
literature-cited equations for Ms and L/S, correlation analysis, every
model's hyperparameters, evaluation results, and sensitivity analysis.

## Files

| File | Purpose |
|---|---|
| `ui_app.py` | Streamlit app: Forward prediction, Inverse mix design, Model Info tabs |
| `aas_setting_time_pipeline.py` | Data cleaning, feature engineering, hybrid model training, GA inverse design |
| `eda_analysis.py` | Descriptive statistics, correlation heatmap, distribution figures (Step 4) |
| `sensitivity_analysis.py` | SHAP (or verified fallback) sensitivity analysis (Step 8) |
| `run_training.py` | Wrapper to retrain — **always use this, not the pipeline file directly** |
| `aas_hybrid_model.joblib` | Pre-trained model artifacts the app loads on startup |
| `Setting_timeX.xlsx` | Raw lab data (single-sheet format) |
| `aas_cleaned_dataset.csv` | Cleaned dataset (also used as the SHAP/fallback background set) |
| `aas_model_metrics_comparison.csv` | R2/MSE/RMSE/MAE per base learner vs hybrid stack (LOOCV) |
| `aas_pso_hyperparameter_log.csv` | Every PSO trial (particle, iteration, hyperparameters, fitness) |
| `aas_eda_*.csv` | Descriptive statistics, correlation matrix, strong correlations |
| `requirements.txt` | Pinned Python dependencies (includes shap, matplotlib) |
| `runtime.txt` | Pins Python 3.12 for Streamlit Cloud |

## SHAP sensitivity analysis — current status

`shap` could not be installed or tested in the environment used to build
this project (no network access to pip in that sandbox). The pipeline
automatically falls back to scikit-learn permutation importance + manual
partial dependence when `shap` is absent — this fallback path **was**
actually run and verified, and its figures are in the report. Once you
`pip install shap` on your own machine and rerun `run_training.py`, the
pipeline will automatically use real SHAP instead — no code changes
needed. Please verify this path works in your environment and let us know
if anything errors.

## Deploying permanently (Streamlit Community Cloud — free)

### 1. Push this folder to GitHub

```bash
git init
git add .
git commit -m "Initial AAS mix design app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Use a **private** repo if your lab data is proprietary.

### 2. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch `main`, main file path `ui_app.py`.
4. Click **Deploy**.

You get a permanent URL like `https://<your-app-name>.streamlit.app`.

### 3. Updating the app later

```bash
python run_training.py --data Setting_timeX.xlsx
git add Setting_timeX.xlsx aas_hybrid_model.joblib aas_cleaned_dataset.csv \
        aas_model_metrics_comparison.csv aas_pso_hyperparameter_log.csv \
        aas_eda_descriptive_stats.csv aas_eda_correlation_matrix.csv aas_eda_strong_correlations.csv
git commit -m "Retrain with new trials"
git push
```

Streamlit Cloud auto-redeploys on every push to `main`.

## Important: always retrain via `run_training.py`

Never run `python aas_setting_time_pipeline.py` directly — doing so
pickles the model's `HybridModel` class under `__main__`, which breaks
when `ui_app.py` (a different `__main__`) tries to load it. Always use:

```bash
python run_training.py --data Setting_timeX.xlsx
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_training.py --data Setting_timeX.xlsx   # generates aas_hybrid_model.joblib + all figures/CSVs
streamlit run ui_app.py
```

# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Model Interpretability with SHAP
# MAGIC %md
# MAGIC # Walk: Model Interpretability
# MAGIC
# MAGIC A model that performs well isn't enough. Stakeholders (and regulators, in lending) need to understand *why* a model makes the predictions it does.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the best model from MLflow (the one tagged `model_status=best`)
# MAGIC 2. Runs SHAP (SHapley Additive exPlanations) to explain feature contributions
# MAGIC 3. Generates summary plots and individual explanations
# MAGIC 4. Logs all artifacts back to the MLflow run

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %pip install shap --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Configuration and imports
import mlflow
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.loan_features_source"
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/lending_club_walk"

mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# DBTITLE 1,Load best model
# MAGIC %md
# MAGIC ## Load the best model from MLflow
# MAGIC
# MAGIC We find the run tagged with `model_status=best` from the training notebook and load the full pipeline (preprocessor + classifier).

# COMMAND ----------

# DBTITLE 1,Find and load best model
# Find the best run by tag
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.model_status = 'best'",
    order_by=["metrics.test_roc_auc DESC"],
    max_results=1,
)

assert len(runs) > 0, "No run found with tag model_status='best'. Run 02_training first."

best_run_id = runs.iloc[0].run_id
print(f"Best run: {best_run_id}")
print(f"Test AUC: {runs.iloc[0]['metrics.test_roc_auc']:.4f}")

# Load the model pipeline
model_uri = f"runs:/{best_run_id}/lending_club_default_classifier"
pipeline = mlflow.sklearn.load_model(model_uri)
print(f"\n\u2713 Model loaded from: {model_uri}")

# COMMAND ----------

# DBTITLE 1,SHAP explanations
# MAGIC %md
# MAGIC ## Compute SHAP values
# MAGIC
# MAGIC We'll use a sample of the data (SHAP on the full dataset is expensive). The preprocessor transforms features before they hit the classifier, so we run SHAP on the *transformed* feature space and map back to original column names.

# COMMAND ----------

# DBTITLE 1,Prepare data and compute SHAP
# Load feature data
pdf = spark.table(SOURCE_TABLE).toPandas()
X = pdf.drop(columns=["is_default", "loan_id", "created_at"])
y = pdf["is_default"]

# Transform features through the preprocessor
preprocessor = pipeline.named_steps["prep"]
X_transformed = preprocessor.transform(X)

# Get actual output feature names from the fitted ColumnTransformer
try:
    feature_names = list(preprocessor.get_feature_names_out())
except AttributeError:
    # Fallback: use input column lists
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    feature_names = list(num_cols) + list(cat_cols)

# Trim names to match transformed output shape
feature_names = feature_names[:X_transformed.shape[1]]
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

# Sample for SHAP (1000 rows keeps it fast)
np.random.seed(42)
sample_idx = np.random.choice(len(X_transformed_df), size=min(1000, len(X_transformed_df)), replace=False)
X_sample = X_transformed_df.iloc[sample_idx]

print(f"Computing SHAP values on {len(X_sample)} samples...")

# Use TreeExplainer for HistGradientBoosting
classifier = pipeline.named_steps["clf"]
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_sample)

print("\u2713 SHAP values computed")

# COMMAND ----------

# DBTITLE 1,SHAP summary plot
# Global feature importance: which features matter most across all predictions?
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
plt.title("SHAP Feature Importance (Top 20)")
plt.tight_layout()
plt.show()

# Log to MLflow
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_figure(fig, "shap_summary_plot.png")
plt.close(fig)
print("\u2713 Summary plot logged to MLflow")

# COMMAND ----------

# DBTITLE 1,SHAP bar plot
# Mean absolute SHAP values (cleaner view of overall importance)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
plt.title("Mean |SHAP| Feature Importance (Top 20)")
plt.tight_layout()
plt.show()

with mlflow.start_run(run_id=best_run_id):
    mlflow.log_figure(fig, "shap_bar_plot.png")
plt.close(fig)
print("\u2713 Bar plot logged to MLflow")

# COMMAND ----------

# DBTITLE 1,Individual explanation (waterfall)
# Explain a single prediction: why did the model flag this loan as high-risk?
default_indices = y.iloc[sample_idx][y.iloc[sample_idx] == 1].index
if len(default_indices) > 0:
    example_idx = 0  # first defaulted loan in sample
    print(f"Explaining prediction for sample index {example_idx} (predicted default):")
    
    shap_explanation = shap.Explanation(
        values=shap_values[example_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[example_idx].values,
        feature_names=feature_names,
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, show=False, max_display=15)
    plt.title("SHAP Waterfall \u2014 Individual Default Prediction")
    plt.tight_layout()
    plt.show()

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_figure(fig, "shap_waterfall_example.png")
    plt.close(fig)
    print("\u2713 Waterfall plot logged to MLflow")
# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Model Deployment
# MAGIC %md
# MAGIC # Walk: Model Deployment
# MAGIC
# MAGIC Time to ship it. This notebook:
# MAGIC 1. Finds the best model from the training experiment (tagged `model_status=best`)
# MAGIC 2. Registers it to **Unity Catalog** as a versioned model
# MAGIC 3. Sets a `champion` alias for production use
# MAGIC 4. Stands up a **Model Serving endpoint** for real-time REST predictions

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.loan_default_classifier"
ENDPOINT_NAME = "lending-club-default-classifier"
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/lending_club_walk"

mlflow.set_registry_uri("databricks-uc")
print(f"Model will be registered as: {MODEL_NAME}")
print(f"Endpoint name: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Register to Unity Catalog
# MAGIC %md
# MAGIC ## Register the best model to Unity Catalog
# MAGIC
# MAGIC Unity Catalog gives us versioning, lineage, and governance for the model. We find the best run, register it, and assign the `champion` alias so downstream consumers always get the right version.

# COMMAND ----------

# DBTITLE 1,Find best run and register model
# Find the best run
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

# Register to Unity Catalog
model_uri = f"runs:/{best_run_id}/lending_club_default_classifier"
registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

print(f"\n✓ Registered as: {MODEL_NAME} v{registered_model.version}")

# COMMAND ----------

# DBTITLE 1,Set champion alias
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered_model.version,
)

print(f"✓ Alias 'champion' set on v{registered_model.version}")
print(f"  Load with: mlflow.pyfunc.load_model('models:/{MODEL_NAME}@champion')")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint
# MAGIC %md
# MAGIC ## Create Model Serving endpoint
# MAGIC
# MAGIC We use the Databricks SDK to create a serverless Model Serving endpoint. It auto-scales to zero when idle and handles burst traffic automatically.

# COMMAND ----------

# DBTITLE 1,Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import time

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=registered_model.version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

# Check if endpoint already exists
try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists — updating...")
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=served_entities,
    )
except Exception:
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
    )

print(f"\n✓ Endpoint '{ENDPOINT_NAME}' is READY")
print(f"  Model: {MODEL_NAME} v{registered_model.version}")

# COMMAND ----------

# DBTITLE 1,Quick smoke test
import pandas as pd
import numpy as np

# Quick smoke test: grab one row from the feature table and score it
sample = spark.table(f"{CATALOG}.{SCHEMA}.loan_features_source").limit(1).toPandas()
sample = sample.drop(columns=["is_default", "loan_id", "created_at"])

# Replace NaN with None for JSON serialization
sample = sample.replace({np.nan: None})

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=sample.to_dict(orient="records"),
)

print("Smoke test prediction:")
print(f"  Input: {sample.iloc[0][['loan_amnt', 'int_rate', 'grade', 'term']].to_dict()}")
print(f"  Prediction: {response.predictions}")
print(f"\n🎉 Endpoint is live and serving predictions!")
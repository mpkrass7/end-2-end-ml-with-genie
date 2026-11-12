# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Sample Inference
# MAGIC %md
# MAGIC # Walk: Sample Inference
# MAGIC
# MAGIC This notebook tests the deployed Model Serving endpoint by sending a batch of sample loan applications and comparing predictions against actual outcomes.
# MAGIC
# MAGIC This is what real inference looks like: grab some data, POST it to the endpoint, get predictions back. There is no local model loading, dependency management, or GPU provisioning needed.

# COMMAND ----------

# DBTITLE 1,Configuration
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.loan_features_source"
ENDPOINT_NAME = "lending-club-default-classifier"

w = WorkspaceClient()
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Source table: {SOURCE_TABLE}")

# COMMAND ----------

# DBTITLE 1,Load sample data
# MAGIC %md
# MAGIC ## Load a sample of loans from the Feature Store
# MAGIC
# MAGIC We grab a random sample of 20 loans with their actual outcomes so we can compare predictions to reality.

# COMMAND ----------

# DBTITLE 1,Sample 20 loans with actuals
# Get a random sample including the actual label for comparison
sample_df = (
    spark.table(SOURCE_TABLE)
    .sample(fraction=0.01, seed=42)
    .limit(20)
    .toPandas()
)

# Separate actuals from features
actuals = sample_df[["loan_id", "is_default"]].copy()
features = sample_df.drop(columns=["is_default", "loan_id", "created_at"])

print(f"Sample: {len(features)} loans")
print(f"Actual defaults in sample: {actuals['is_default'].sum()} / {len(actuals)}")
features[["loan_amnt", "int_rate", "grade", "term", "annual_inc"]].head(10)

# COMMAND ----------

# DBTITLE 1,Score via endpoint
# MAGIC %md
# MAGIC ## Score via the Model Serving endpoint
# MAGIC
# MAGIC One REST call, 20 predictions. The endpoint handles deserialization, preprocessing (via the sklearn Pipeline we logged), and inference.

# COMMAND ----------

# DBTITLE 1,Query the serving endpoint
import json

# Send the batch to the serving endpoint
# pandas to_json handles NaN → null; to_dict does not
records = json.loads(features.to_json(orient="records"))

response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=records,
)

predictions = response.predictions
print(f"✓ Got {len(predictions)} predictions from endpoint")
print(f"\nFirst 5 predictions: {predictions[:5]}")

# COMMAND ----------

# DBTITLE 1,Compare results
# MAGIC %md
# MAGIC ## Compare predictions to actuals
# MAGIC
# MAGIC Let's see how well the model performs on this sample. Remember: this is hitting a live REST endpoint, exactly how a production application would consume these predictions.

# COMMAND ----------

# DBTITLE 1,Build comparison table
# Build a comparison DataFrame
results = actuals.copy()
results["predicted"] = predictions
results["correct"] = results["is_default"] == results["predicted"]

# Summary stats
accuracy = results["correct"].mean()
print(f"Sample accuracy: {accuracy:.1%} ({results['correct'].sum()}/{len(results)})")
print(f"")
print("Confusion on this sample:")
print(f"  True positives (caught defaults):  {((results['is_default'] == 1) & (results['predicted'] == 1)).sum()}")
print(f"  True negatives (correct non-def):  {((results['is_default'] == 0) & (results['predicted'] == 0)).sum()}")
print(f"  False positives (false alarm):     {((results['is_default'] == 0) & (results['predicted'] == 1)).sum()}")
print(f"  False negatives (missed default):  {((results['is_default'] == 1) & (results['predicted'] == 0)).sum()}")

print("\n--- Full results ---")
results.style.map(
    lambda v: "background-color: #d4edda" if v else "background-color: #f8d7da",
    subset=["correct"],
)
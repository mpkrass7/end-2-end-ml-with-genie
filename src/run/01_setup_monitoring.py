# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Model Monitoring Setup
# MAGIC %md
# MAGIC # Model Monitoring with Lakehouse Monitoring
# MAGIC
# MAGIC This notebook sets up automated monitoring for the loan default classifier:
# MAGIC 1. Creates an **inference log table** simulating production predictions
# MAGIC 2. Attaches a **Lakehouse Monitor** that tracks prediction drift and model health
# MAGIC 3. Generates an auto-managed dashboard for at-a-glance model status

# COMMAND ----------

# DBTITLE 1,Setup & Imports
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, TimestampType
import datetime
import numpy as np

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.loan_features"
INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.inference_log"
CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]
MONITOR_ASSETS_DIR = f"/Workspace/Users/{CURRENT_USER}/genie_ml_cwr/monitoring"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.loan_default_classifier"
ENDPOINT_NAME = "lending-club-default-classifier"

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Inference Log Table
# MAGIC
# MAGIC In production, every prediction the serving endpoint makes gets logged to an **inference table**. For this demo, we'll simulate a week of predictions by scoring the full feature table in batches with synthetic timestamps. This gives Lakehouse Monitoring something to analyze for drift.

# COMMAND ----------

# DBTITLE 1,Build inference log from feature table
# Load the feature table and the champion model
features_df = spark.table(FEATURE_TABLE).toPandas()
model_uri = f"models:/{MODEL_NAME}@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Get feature columns (everything except loan_id and is_default)
label_col = "is_default"
pk_col = "loan_id"
feature_cols = [c for c in features_df.columns if c not in (pk_col, label_col)]

# Score the full dataset in pandas (7k rows — no need for spark UDF)
features_df["prediction"] = model.predict(features_df[feature_cols])
print(f"Scored {len(features_df):,} rows")
print(f"Prediction distribution:\n{features_df['prediction'].value_counts()}")

# COMMAND ----------

# DBTITLE 1,Simulate time-series predictions
# Simulate 7 days of predictions by assigning random timestamps
# This mimics what an inference table would look like in production
now = datetime.datetime.now()
days_back = 7
n_rows = len(features_df)

# Build the inference log in pandas
inference_pd = features_df[[pk_col, label_col, "prediction"]].copy()
inference_pd["prediction"] = inference_pd["prediction"].astype(int)
inference_pd[label_col] = inference_pd[label_col].astype(int)
inference_pd["timestamp"] = [
    now - datetime.timedelta(
        days=int(np.random.randint(0, days_back)),
        hours=int(np.random.randint(0, 24)),
        minutes=int(np.random.randint(0, 60)),
    )
    for _ in range(n_rows)
]
inference_pd["model_version"] = "1"

# Convert to Spark and write as Delta table
inference_df = spark.createDataFrame(inference_pd)
inference_df.write.format("delta").mode("overwrite").saveAsTable(INFERENCE_TABLE)
print(f"✅ Inference log written: {INFERENCE_TABLE}")
print(f"   Rows: {n_rows:,}")
display(spark.table(INFERENCE_TABLE).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up Lakehouse Monitoring
# MAGIC
# MAGIC Now we attach a **Lakehouse Monitor** to the inference log table. This automatically:
# MAGIC - Computes drift metrics (comparing recent predictions to baseline)
# MAGIC - Tracks prediction distribution shifts over time
# MAGIC - Generates a dashboard with health indicators
# MAGIC
# MAGIC The monitor runs on a schedule and alerts us when the model's inputs or outputs drift beyond acceptable thresholds.

# COMMAND ----------

# DBTITLE 1,Create or refresh the quality monitor
# Check if monitor already exists
try:
    existing = w.quality_monitors.get(table_name=INFERENCE_TABLE)
    print(f"Monitor already exists on {INFERENCE_TABLE}, refreshing metrics...")
    w.quality_monitors.run_refresh(table_name=INFERENCE_TABLE)
except Exception:
    print(f"Creating new monitor on {INFERENCE_TABLE}...")
    monitor = w.quality_monitors.create(
        table_name=INFERENCE_TABLE,
        assets_dir=MONITOR_ASSETS_DIR,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            label_col="is_default",
            timestamp_col="timestamp",
            model_id_col="model_version",
            granularities=["1 day"],
        ),
        output_schema_name=f"{CATALOG}.{SCHEMA}",
    )
    print(f"✅ Monitor created!")
    print(f"   Dashboard: {monitor.dashboard_id}")
    print(f"   Profile metrics table: {INFERENCE_TABLE}_profile_metrics")
    print(f"   Drift metrics table: {INFERENCE_TABLE}_drift_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Monitoring Results
# MAGIC
# MAGIC Once the monitor runs, it populates two tables:
# MAGIC - `*_profile_metrics` — summary stats per time window (mean prediction, label rate, etc.)
# MAGIC - `*_drift_metrics` — statistical tests comparing each window to baseline
# MAGIC
# MAGIC Let's check what we have:

# COMMAND ----------

# DBTITLE 1,Check profile metrics
import time

# Give the monitor a moment to compute initial metrics
print("Waiting for initial metrics computation...")
time.sleep(10)

# Check profile metrics
try:
    profile_df = spark.table(f"{INFERENCE_TABLE}_profile_metrics")
    print(f"Profile metrics rows: {profile_df.count()}")
    display(profile_df.select("window", "column_name", "count", "percent_zeros", "percent_nulls").limit(20))
except Exception as e:
    print(f"Profile metrics not yet available (monitor may still be computing): {e}")
    print("Check back in a few minutes or view the auto-generated dashboard.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We now have:
# MAGIC - An **inference log table** capturing every prediction with timestamps and model version
# MAGIC - A **Lakehouse Monitor** that automatically computes drift and profile metrics daily
# MAGIC - An **auto-generated dashboard** showing prediction distributions, accuracy trends, and drift alerts
# MAGIC
# MAGIC When the monitor detects that input features have drifted (e.g., the distribution of interest rates shifts because of market changes), it flags the degradation — telling us it's time to retrain via the DAB job.
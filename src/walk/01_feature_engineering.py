# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Feature Engineering with Feature Views
# MAGIC %md
# MAGIC # Walk: Feature Engineering with Feature Views
# MAGIC
# MAGIC This notebook creates **Feature Views** from our Lending Club data using the `ColumnSelection` pattern. Instead of writing a monolithic feature table, we register each feature column as a governed Unity Catalog object. This gives us lineage, discoverability, and a clean contract between feature producers and model consumers.
# MAGIC
# MAGIC We'll:
# MAGIC 1. Load the raw data and apply the same cleaning logic from crawl
# MAGIC 2. Write the cleaned features to a Delta source table
# MAGIC 3. Register each column as a `ColumnSelection` Feature View in UC

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %pip install databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Configuration
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
SOURCE_TABLE_NAME = "loan_features_source"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.{SOURCE_TABLE_NAME}"
VOLUME_PATH = "/Volumes/samples/databricks/datasets/lending-club-loan-stats/LoanStats_2018Q2.csv"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Load and clean
# MAGIC %md
# MAGIC ## Load raw data and apply cleaning logic
# MAGIC
# MAGIC We replicate the same data prep from crawl but package it as a reusable function. This becomes the **feature computation logic** that the Feature Store will version and serve.

# COMMAND ----------

# DBTITLE 1,Feature computation function
from pyspark.sql import functions as F, DataFrame


def compute_loan_features(raw_df: DataFrame) -> DataFrame:
    """Transform raw Lending Club data into model-ready features.
    
    This is the single source of truth for feature logic.
    Both training and inference use this function.
    """
    # --- Label engineering ---
    default_statuses = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)"]
    exclude_statuses = ["Current", "In Grace Period"]

    df = (
        raw_df
        .filter(~F.col("loan_status").isin(exclude_statuses))
        .filter(F.col("loan_status").isNotNull())
        .withColumn(
            "is_default",
            F.when(F.col("loan_status").isin(default_statuses), 1).otherwise(0),
        )
    )

    # --- Drop leaky / useless columns ---
    leaky_cols = [
        "loan_status", "funded_amnt", "funded_amnt_inv",
        "out_prncp", "out_prncp_inv",
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
        "last_credit_pull_d", "issue_d", "pymnt_plan",
        "initial_list_status", "policy_code",
    ]
    id_text_cols = ["id", "member_id", "url", "desc", "emp_title", "title", "zip_code"]
    hardship_cols = [c for c in df.columns if c.startswith("hardship_") or c.startswith("debt_settlement") or c.startswith("settlement_")]
    joint_cols = [c for c in df.columns if "joint" in c or c.startswith("sec_app_")]

    drop_cols = [c for c in set(leaky_cols + id_text_cols + hardship_cols + joint_cols) if c in df.columns]
    df = df.drop(*drop_cols)

    # --- Clean numeric-as-string columns ---
    df = df.withColumn("int_rate", F.regexp_replace("int_rate", "%", "").cast("double"))
    df = df.withColumn("revol_util", F.regexp_replace("revol_util", "%", "").cast("double"))
    df = df.withColumn("term", F.trim(F.regexp_replace("term", " months", "")).cast("double"))
    df = df.withColumn(
        "emp_length",
        F.when(F.col("emp_length").isin("n/a", "N/A"), None)
        .otherwise(
            F.regexp_replace(
                F.regexp_replace("emp_length", r"\+? years?", ""),
                "< 1", "0"
            )
        ).cast("double"),
    )

    # --- Add a unique key for Feature Store lookups ---
    df = df.withColumn("loan_id", F.monotonically_increasing_id())

    return df

# COMMAND ----------

# DBTITLE 1,Apply feature computation and save source table
raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(VOLUME_PATH)
)

features_df = compute_loan_features(raw_df)

# Add a synthetic timestamp so the Feature Views machinery is happy.
features_df = features_df.withColumn("created_at", F.current_timestamp())

# Write to Delta (this becomes the source for our Feature Views)
try:
    features_df.write.saveAsTable(SOURCE_TABLE)
    print(f"Created source table: {SOURCE_TABLE}")
except Exception:
    # Table already exists from a prior run, overwrite it
    features_df.write.mode("overwrite").saveAsTable(SOURCE_TABLE)
    print(f"Overwrote existing source table: {SOURCE_TABLE}")

print(f"Source table: {SOURCE_TABLE}")
print(f"  {features_df.count():,} rows x {len(features_df.columns)} columns")
features_df.display()

# COMMAND ----------

# DBTITLE 1,Register as Feature Views
# MAGIC %md
# MAGIC ## Register as Feature Views
# MAGIC
# MAGIC Each feature column is registered as a `ColumnSelection` Feature View in Unity Catalog. This gives us:
# MAGIC - **Governance**: every feature is a first-class UC object with lineage and permissions
# MAGIC - **Discoverability**: features are browsable and searchable in the catalog
# MAGIC - **Composability**: training notebooks cherry-pick exactly the features they need via `create_training_set`

# COMMAND ----------

# DBTITLE 1,Register ColumnSelection Feature Views
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    ColumnSelection,
    DeltaTableSource,
    Feature,
)

fe = FeatureEngineeringClient()

src = DeltaTableSource(
    catalog_name=CATALOG,
    schema_name=SCHEMA,
    table_name=SOURCE_TABLE_NAME,
)

# Every column except the entity key, label, and timestamp becomes a ColumnSelection feature
exclude_cols = {"loan_id", "is_default", "created_at"}
feature_cols = sorted(c for c in features_df.columns if c not in exclude_cols)

# Clean up previously registered features for idempotent reruns
for col_name in feature_cols:
    full_name = f"{CATALOG}.{SCHEMA}.{col_name}"
    try:
        for mf in fe.list_materialized_features(feature_name=full_name):
            fe.delete_materialized_feature(materialized_feature=mf)
        fe.delete_feature(full_name=full_name)
    except Exception:
        pass

# Register each column as a ColumnSelection Feature View
registered: list[Feature] = []
for col_name in feature_cols:
    feat = fe.create_feature(
        source=src,
        function=ColumnSelection(col_name),
        entity=["loan_id"],
        timeseries_column="created_at",
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        name=col_name,
    )
    registered.append(feat)

print(f"✅ Registered {len(registered)} ColumnSelection features in {CATALOG}.{SCHEMA}")
print(f"  Entity key: loan_id")
print(f"  Source table: {SOURCE_TABLE}")
print(f"  Sample features: {feature_cols[:5]}")
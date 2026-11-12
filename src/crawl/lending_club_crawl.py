# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Crawl: From Raw Data to a Working Model
# MAGIC %md
# MAGIC # Crawl: From Raw Data to a Working Model
# MAGIC
# MAGIC This notebook takes raw Lending Club loan data and builds a binary classifier that predicts loan defaults.
# MAGIC
# MAGIC We'll go from a messy CSV with 145 columns to a trained, logged model in one shot:
# MAGIC 1. Load & explore the raw data
# MAGIC 2. Engineer a proper binary label from `loan_status`
# MAGIC 3. Remove leaky and post-origination features
# MAGIC 4. Fit a few models, log to MLflow, and pick the best one

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Configuration
# Parameters — point these at your own catalog/schema to reproduce
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_crawl"
VOLUME_PATH = "/Volumes/samples/databricks/datasets/lending-club-loan-stats/LoanStats_2018Q2.csv"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# DBTITLE 1,1. Load & Explore
# MAGIC %md
# MAGIC ## 1. Load & Explore the Raw Data
# MAGIC
# MAGIC The Lending Club dataset ships with Databricks samples. It's a single CSV with **145 columns** covering everything from borrower demographics to repayment history. Most of those columns will need to go — but first, let's see what we're working with.

# COMMAND ----------

# DBTITLE 1,Load raw CSV
raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(VOLUME_PATH)
)

print(f"Rows: {raw_df.count():,}")
print(f"Columns: {len(raw_df.columns)}")
raw_df.display()

# COMMAND ----------

# DBTITLE 1,Check loan_status distribution
# The loan_status column is what we'll derive our label from.
# Let's see the distribution to understand what outcomes exist.
raw_df.groupBy("loan_status").count().orderBy("count", ascending=False).display()

# COMMAND ----------

# DBTITLE 1,2. Label Engineering
# MAGIC %md
# MAGIC ## 2. Engineer the Binary Label
# MAGIC
# MAGIC This is where most AutoML tools would choke. You can't just use `loan_status` as-is because:
# MAGIC
# MAGIC - **"Current"** loans haven't resolved yet — we don't know if they'll default. Using them would be data leakage from the future.
# MAGIC - **"Fully Paid"** is our clear negative class (no default).
# MAGIC - **"Charged Off"** is our clear positive class (definite default).
# MAGIC - **"Late" categories** indicate serious delinquency — these borrowers are on the path to default.
# MAGIC - **"In Grace Period"** is ambiguous — we'll exclude these to keep the label clean.
# MAGIC
# MAGIC We'll drop all "Current" and "In Grace Period" loans, then map the rest to a binary `is_default` label.

# COMMAND ----------

# DBTITLE 1,Create binary label
from pyspark.sql import functions as F

# Define label mapping
default_statuses = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)"]
non_default_statuses = ["Fully Paid"]
exclude_statuses = ["Current", "In Grace Period"]  # unresolved or ambiguous

# Filter to only resolved loans and create binary label
labeled_df = (
    raw_df
    .filter(~F.col("loan_status").isin(exclude_statuses))
    .filter(F.col("loan_status").isNotNull())
    .withColumn(
        "is_default",
        F.when(F.col("loan_status").isin(default_statuses), 1).otherwise(0)
    )
)

print(f"Labeled dataset size: {labeled_df.count():,} rows")
labeled_df.groupBy("is_default").count().display()

# COMMAND ----------

# DBTITLE 1,3. Feature Cleanup
# MAGIC %md
# MAGIC ## 3. Remove Leaky & Useless Features
# MAGIC
# MAGIC Of 145 columns, a huge chunk are **post-origination data** that wouldn't exist at the time you'd make a prediction (e.g., `total_pymnt`, `recoveries`, `last_pymnt_d`). Using these would be target leakage.
# MAGIC
# MAGIC We'll also drop:
# MAGIC - **Identifiers** (`id`, `member_id`, `url`) — no predictive value
# MAGIC - **Free-text fields** (`emp_title`, `title`, `desc`) — high cardinality noise for a tabular model
# MAGIC - **Redundant columns** (`funded_amnt_inv` ≈ `funded_amnt`)
# MAGIC - **Post-origination payment data** — anything about payments received, outstanding principal, etc.
# MAGIC - **Hardship/settlement columns** — these only exist after a loan goes bad
# MAGIC - **Joint application columns** — mostly null for individual applications

# COMMAND ----------

# DBTITLE 1,Drop leaky and useless columns
# Columns that leak future information (post-origination)
leaky_cols = [
    "loan_status",  # used to derive label, drop now
    "funded_amnt", "funded_amnt_inv",  # funded = approved (known post-application)
    "out_prncp", "out_prncp_inv",  # outstanding principal (post-origination)
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",  # payment history
    "total_rec_int", "total_rec_late_fee",  # payment history
    "recoveries", "collection_recovery_fee",  # post-default recovery
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",  # payment timing
    "last_credit_pull_d",  # post-origination credit pull
    "issue_d",  # when the loan was issued (wouldn't know at application time? debatable, but removing for safety)
    "pymnt_plan",  # post-origination modification
    "initial_list_status",  # internal platform mechanics
    "policy_code",  # always 1
]

# Identifiers and free-text (no predictive value for tabular ML)
id_text_cols = ["id", "member_id", "url", "desc", "emp_title", "title", "zip_code"]

# Hardship & debt settlement (only populated after distress)
hardship_cols = [c for c in labeled_df.columns if c.startswith("hardship_") or c.startswith("debt_settlement") or c.startswith("settlement_")]

# Joint/secondary applicant columns (mostly null)
joint_cols = [c for c in labeled_df.columns if "joint" in c or c.startswith("sec_app_")]

# Combine all columns to drop
cols_to_drop = set(leaky_cols + id_text_cols + hardship_cols + joint_cols)
cols_to_drop = [c for c in cols_to_drop if c in labeled_df.columns]

clean_df = labeled_df.drop(*cols_to_drop)
print(f"Columns after cleanup: {len(clean_df.columns)} (removed {len(cols_to_drop)})")

# COMMAND ----------

# DBTITLE 1,Handle remaining data types and nulls
from pyspark.sql.types import StringType, NumericType

# Identify remaining string vs numeric columns
string_cols = [f.name for f in clean_df.schema.fields if isinstance(f.dataType, StringType)]
numeric_cols = [f.name for f in clean_df.schema.fields 
                if isinstance(f.dataType, NumericType) and f.name != "is_default"]

print(f"String columns ({len(string_cols)}): {string_cols}")
print(f"Numeric columns ({len(numeric_cols)})")

# Check null rates for remaining columns
total_rows = clean_df.count()
null_rates = clean_df.select(
    [F.round(F.sum(F.col(c).isNull().cast("int")) / total_rows, 3).alias(c) 
     for c in clean_df.columns]
).toPandas().T.rename(columns={0: "null_rate"}).sort_values("null_rate", ascending=False)

# Drop columns with >50% nulls (too sparse to be useful)
high_null_cols = null_rates[null_rates["null_rate"] > 0.5].index.tolist()
print(f"\nDropping {len(high_null_cols)} columns with >50% nulls: {high_null_cols}")

clean_df = clean_df.drop(*high_null_cols)

# COMMAND ----------

# DBTITLE 1,Encode categoricals and prepare final features
import pandas as pd
import numpy as np

# Convert to pandas for sklearn modeling 
# (dataset is small enough after filtering: ~7k rows)
pdf = clean_df.toPandas()

# Some numeric columns are stored as strings — clean them up
if "int_rate" in pdf.columns:
    pdf["int_rate"] = pdf["int_rate"].str.replace("%", "", regex=False).astype(float)
if "revol_util" in pdf.columns:
    pdf["revol_util"] = pdf["revol_util"].str.replace("%", "", regex=False).astype(float)
if "term" in pdf.columns:
    pdf["term"] = pdf["term"].str.strip().str.replace(" months", "", regex=False).astype(float)
if "emp_length" in pdf.columns:
    pdf["emp_length"] = (
        pdf["emp_length"]
        .str.replace(r"\+? years?", "", regex=True)
        .str.replace("< 1", "0", regex=False)
        .str.strip()
    )
    pdf["emp_length"] = pd.to_numeric(pdf["emp_length"], errors="coerce")

# Identify final categorical vs numeric columns (encoding happens in model pipelines)
cat_cols = pdf.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in pdf.select_dtypes(include="number").columns if c != "is_default"]

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns: {len(num_cols)}")
print(f"\nDataset: {pdf.shape[0]} rows x {pdf.shape[1]} columns")
print(f"Target distribution:\n{pdf['is_default'].value_counts()}")

# COMMAND ----------

# DBTITLE 1,4. Train & Compare Models
# MAGIC %md
# MAGIC ## 4. Train Models & Log to MLflow
# MAGIC
# MAGIC We'll fit three classifiers representing different complexity levels:
# MAGIC 1. **Logistic Regression** — fast, interpretable baseline
# MAGIC 2. **Random Forest** — handles non-linearity, robust to outliers
# MAGIC 3. **Gradient Boosted Trees (XGBoost)** — typically top performer on tabular data
# MAGIC
# MAGIC Each gets logged to MLflow with metrics, parameters, and the model artifact so we can compare them side-by-side in the experiment UI.

# COMMAND ----------

# DBTITLE 1,Train/validate/test split
from sklearn.model_selection import train_test_split

X = pdf.drop(columns=["is_default"])
y = pdf["is_default"]

# 60/20/20 stratified split:
#   train    -> fit models
#   validate -> pick the best model
#   test     -> final held-out evaluation (touched once)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]} | Validate: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"Default rates: Train {y_train.mean():.1%} | Val {y_val.mean():.1%} | Test {y_test.mean():.1%}")
print(f"\nFeatures: {len(num_cols)} numeric + {len(cat_cols)} categorical")

# COMMAND ----------

# DBTITLE 1,Train models and evaluate on validation set
import mlflow
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay,
)

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/lending_club_crawl")


# --- Preprocessing pipelines (encoding lives here, not in the DataFrame) ---

# For tree models: ordinal encode categoricals (trees split on values, order doesn't matter)
tree_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])

# For linear models: one-hot encode + scale numerics
linear_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])


def evaluate_and_log(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict:
    """Train on train set, evaluate on validation set, and log to MLflow."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_proba),
        }

        mlflow.log_metrics(metrics)

        # Log confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"{model_name} \u2014 Confusion Matrix")
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # Log model with descriptive name
        model_slug = model_name.lower().replace(" ", "_")
        mlflow.sklearn.log_model(model, name=model_slug, input_example=X_val.iloc[:5])

        print(f"\n{model_name} (validation)")
        print(f"  AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

    return {**metrics, "y_proba": y_proba}


# --- Define models (each is a full pipeline: preprocessing + classifier) ---
models = {
    "Logistic Regression": Pipeline([
        ("prep", linear_preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("prep", tree_preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ]),
    "Gradient Boosted Trees": Pipeline([
        ("prep", tree_preprocessor),
        ("clf", HistGradientBoostingClassifier(max_iter=200, random_state=42, class_weight="balanced")),
    ]),
}

# Train and evaluate
results = {}
for name, model in models.items():
    results[name] = evaluate_and_log(model, name, X_train, X_val, y_train, y_val)

# COMMAND ----------

# DBTITLE 1,Compare on validation, then test the winner
# --- Validation set comparison (picks the winner) ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    RocCurveDisplay.from_predictions(
        y_val, res["y_proba"], name=f"{name} (AUC={res['roc_auc']:.3f})", ax=ax
    )
ax.set_title("ROC Curves on Validation Set")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.tight_layout()
plt.show()

metrics_only = {k: {m: v for m, v in res.items() if m != "y_proba"} for k, res in results.items()}
val_df = pd.DataFrame(metrics_only).T.sort_values("roc_auc", ascending=False)
print("Validation results (used for model selection):")
display(val_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"))

# --- Held-out test evaluation (winner only, touched once) ---
best_name = val_df.index[0]
best_model = models[best_name]

y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
    "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
    "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
    "test_roc_auc": roc_auc_score(y_test, y_test_proba),
}

print(f"\nBest model: {best_name}")
print("Held-out test results (untouched during model selection):")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# DBTITLE 1,Wrap-up
# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC All three models are logged to MLflow. Check the experiment UI to compare runs, inspect feature importance, and see the full parameter sets.
# MAGIC
# MAGIC **What we accomplished in a single notebook:**
# MAGIC - Loaded raw CSV data (145 columns)
# MAGIC - Engineered a clean binary label from ambiguous loan statuses
# MAGIC - Removed 80+ leaky/useless features that would have poisoned any model
# MAGIC - Trained and compared three classifiers with proper evaluation
# MAGIC
# MAGIC This is the **crawl** stage. Next up in **walk**: Feature Store, model serving, and beefier hyperparameter tuning.
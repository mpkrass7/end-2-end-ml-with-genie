# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Hyperparameter Tuning with Optuna
# MAGIC %md
# MAGIC # Walk: Hyperparameter Tuning
# MAGIC
# MAGIC In crawl, we picked the best model type (HistGradientBoosting had the best AUC). Now we properly tune it with **Optuna** — a Bayesian optimization framework that's smarter than grid search.
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads features from the Feature Store table
# MAGIC 2. Runs an Optuna study to find optimal hyperparameters
# MAGIC 3. Logs every trial to MLflow (so you get a full leaderboard)
# MAGIC 4. Tags the best run for the interpretability and deployment notebooks to pick up

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %pip install optuna --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog", "main", "Catalog")

# COMMAND ----------

# DBTITLE 1,Configuration and imports
import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "lending_club_walk"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.loan_features_source"
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/lending_club_walk"

mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Load features and split
# MAGIC %md
# MAGIC ## Load features and split train/validate/test
# MAGIC
# MAGIC The feature source table was created by `01_feature_engineering`. We split 60/20/20: **train** fits each Optuna trial, **validate** scores each trial to pick the best HP config, and **test** is held out until the very end for an unbiased performance estimate.

# COMMAND ----------

# DBTITLE 1,Load data and split train/validate/test
from sklearn.model_selection import train_test_split

pdf = spark.table(SOURCE_TABLE).toPandas()
y = pdf["is_default"]
X = pdf.drop(columns=["is_default", "loan_id", "created_at"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

# 60/20/20 stratified split:
#   train    -> fit each Optuna trial
#   validate -> score each trial (picks the best HPs)
#   test     -> final held-out evaluation (touched once)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape[0]:,} | Validate: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"Default rates: Train {y_train.mean():.1%} | Val {y_val.mean():.1%} | Test {y_test.mean():.1%}")
print(f"Features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

# COMMAND ----------

# DBTITLE 1,Optuna study
# MAGIC %md
# MAGIC ## Run Optuna hyperparameter search
# MAGIC
# MAGIC We search over `HistGradientBoostingClassifier` hyperparameters by training on the train set and scoring on the validation set. Each trial is logged to MLflow so we get a complete experiment history.

# COMMAND ----------

# DBTITLE 1,Define Optuna objective
# Preprocessor: ordinal encode categoricals, impute nulls
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])

def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize ROC AUC on the validation set."""
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42,
    }

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(**params)),
    ])

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_proba)

    # Log trial to MLflow
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_roc_auc", val_auc)

    return val_auc

# COMMAND ----------

# DBTITLE 1,Run Optuna study
# Run the study (50 trials is a good balance of thoroughness vs. speed)
with mlflow.start_run(run_name="optuna_sweep"):
    study = optuna.create_study(direction="maximize", study_name="lending_club_hgbt")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  Val AUC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Log best params to parent run
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_roc_auc", study.best_value)

# COMMAND ----------

# DBTITLE 1,Train final model on train, evaluate on test
# MAGIC %md
# MAGIC ## Train final model with best params
# MAGIC
# MAGIC Now we retrain on the training set with the best hyperparameters and evaluate on the held-out test set for an unbiased performance estimate.

# COMMAND ----------

# DBTITLE 1,Train and log best model
# Build final pipeline with best params, trained on the training set
best_params = {**study.best_params, "class_weight": "balanced", "random_state": 42}
final_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", HistGradientBoostingClassifier(**best_params)),
])
final_pipe.fit(X_train, y_train)

# Evaluate
y_proba = final_pipe.predict_proba(X_test)[:, 1]
y_pred = final_pipe.predict(X_test)

metrics = {
    "test_roc_auc": roc_auc_score(y_test, y_proba),
    "test_f1": f1_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_recall": recall_score(y_test, y_pred),
}

# Log final model
with mlflow.start_run(run_name="best_model") as run:
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        final_pipe,
        name="lending_club_default_classifier",
        input_example=X_test.iloc[:5],
    )
    mlflow.set_tag("model_status", "best")
    best_run_id = run.info.run_id

print(f"\nFinal model performance (held-out test set):")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print(f"\nLogged as run: {best_run_id}")
print(f"Tagged with model_status='best' for downstream notebooks to find")
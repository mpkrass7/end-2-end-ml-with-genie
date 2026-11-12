# Walk

The walk stage uses two prompts between "build" and "ship".

---

## Prompt 1: Make it good

> I built a basic loan default classifier in `src/crawl/` but it needs to be production-quality. I want to split the next stage into three notebooks in `src/walk/`:
>
> 1. **Feature engineering** — set up a Feature View so my training and inference transformations stay in sync.
> 2. **Training** — run a proper hyperparameter search with Optuna over the best model type from crawl, log all trials to MLflow. Use a serverless GPU cluster for this.
> 3. **Interpretability** — in a separate notebook, load the best model from MLflow and run SHAP to explain its predictions.
>
> Use `stablebox_catalog.lending_club_walk` for any feature views.

---

## Prompt 2: Ship it

> Great, let's deploy that sucker! Register the best model from the walk experiment to Unity Catalog and stand up a Model Serving endpoint so I can hit it with REST calls. Put the deployment notebook in `src/walk/04_deployment`. Also add a `05_sample_inference` notebook that loads the endpoint and tests running a sample of predictions on it.

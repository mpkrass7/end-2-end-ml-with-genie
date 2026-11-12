# Run

Two prompts on run stage to split between DAB and monitoring dashboard

## Prompt 1: Automate it

> I want to wrap the walk stage into a **Declarative Automation Bundle** (DAB) so the whole pipeline runs as an automated Databricks job. Put the bundle config in `src/run/`. The job should:
>
> 1. Run feature engineering first
> 2. Then training (depends on step 1)
> 3. Then deployment (depends on step 2)
>
> Also add a schedule so it retrains weekly. Use `stablebox_catalog.lending_club_walk` for everything.

## Prompt 2: Monitor it

> Last piece: I need to know when this model goes stale. Build a monitoring dashboard in `src/run/` that shows model health including prediction distribution over time, drift metrics, and the latest accuracy against a holdout set.
>
> I want to see when the input data shifts enough that I should retrain. It should be incorporated in the DAB.
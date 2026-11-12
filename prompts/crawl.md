# Crawl

Sample prompt for crawl stage

---

> I have raw Lending Club loan data in `/Volumes/samples/databricks/datasets/lending-club-loan-stats/`. I want to build a binary classifier that predicts whether a borrower will default. Can you explore the data, engineer a proper label from the loan status field, clean up features that wouldn't be available at prediction time or are leaky, and then fit a few models to find the best one? Log everything to MLflow so I can compare runs. Use `stablebox_catalog` for any tables you write.

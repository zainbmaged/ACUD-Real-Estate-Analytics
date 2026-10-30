# For the ML teammate
- `panel_project_month.csv`: 40 projects x 24 months (inventory, sales, funnel, collections, construction) - the training table.
- `features_project_latest.csv`: one row per project incl. the current rule-based `risk_score` (use it only as a benchmark, not as a label).
- Deliver predictions as **data/ML_Predictions.csv** (see `ML_Predictions_TEMPLATE.csv`): required columns `Project_ID`, `Unsold_Risk_Probability` (0-1);
  optional `Model_Name`, `Evaluation_WAPE`, `Forecast_Horizon_Months`. Drop it in the `data/` folder - the Unsold Inventory Agent and the dashboard
  pick it up automatically (no code change). Evaluate on a time period NOT used for training and state it in `Evaluation_Period`.
- For forecasting (sales / inventory / collections) provide a long table: `Project_ID, Month, Metric, Actual, Forecast, Lower, Upper`.
- Data is synthetic: 24 months only, so keep the validation window short (e.g. last 6 months) and report WAPE.

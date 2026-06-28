# Prediction Pipeline

Two-stage ML pipeline for last-mile delivery delay prediction.
**Stage 1** classifies orders as delayed / on-time (Random Forest binary
classifier). **Stage 2** assigns a severity level to delayed orders
(Short / Medium / Long via a second Random Forest).

Results are persisted to CSV and a SQLite database with 27 tables
(predictions, summary aggregates, and a metadata dictionary).

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  TRAINING PATH  (run once — notebook)                                ║
║                                                                      ║
║  Delivery_Logistics.csv (25 000 rows, Kaggle)                        ║
║       │                                                              ║
║       ▼                                                              ║
║  DataExtract → DataEDA → DataProcessing → FeatureEngineering         ║
║                                               │                      ║
║                          10+ engineered features                     ║
║                          (schedule_risk, vehicle_load_strain,        ║
║                           km_per_expected_hr, weather_severity …)    ║
║                               │                                      ║
║                               ▼                                      ║
║                   ClassificationModels                               ║
║                   Stage 1 — binary: delay Y/N                        ║
║                   LR| DT| RF| XGB | LGBM | Ada +  GridSearch + CV    ║
║                   Model Selection - RF shortlisted.                  ║
║                               │                                      ║
║                               ▼                                      ║
║                  best_classification_rf.pkl  (models/)               ║
║                               │                                      ║
║                  filter: delayed == YES only                         ║
║                               │  (~26% of rows)                      ║
║                               ▼                                      ║
║                   ClassificationModels                               ║
║                   Stage 2 — severity: Short / Medium / Long          ║
║                   RF |  +  GridSearch + CV                           ║
║                               │                                      ║
║                               ▼                                      ║
║                  best_severity_rf.pkl  (models/)                     ║
╚══════════════════════════════════════════════════════════════════════╝
                            │ trained models
╔══════════════════════════════════════════════════════════════════════╗
║  DAILY PREDICTION PATH  (DailyPredictionPipeline — daily_predict.py) ║
║                                                                      ║
║  daily_delivery_logistics.csv  (new orders)                          ║
║       │                                                              ║
║       ▼                                                              ║
║  Clean → FeatureEngineering → Stage 1 RF (all rows)                  ║
║                                    │                                 ║
║                               delayed Y/N                            ║
║                                    │                                 ║
║                         delayed == YES rows only                     ║
║                                    │                                 ║
║                               Stage 2 RF → Short / Medium / Long     ║
║                                         │                            ║
║                                         ▼                            ║
║                             daily_delivery_delay_prediction.csv      ║
║                             daily_delivery_delay_prediction_meta.json║
║                                         │                            ║
║                                         ▼                            ║
║                              DatabaseOperations (SQLite)             ║
║                              27 tables — daily + historical          ║
║                              summaries across 12 dimensions          ║
╚══════════════════════════════════════════════════════════════════════╝
                            │ CSV + DB ready
╔══════════════════════════════════════════════════════════════════════╗
║  MCP SERVER  (prediction_server.py — FastMCP, stdio transport)       ║
║                                                                      ║
║   ┌──────────┐   ┌──────────────────────┐   ┌───────────────────┐    ║
║   │ predict  │   │      diagnose        │   │     simulate      │    ║
║   │          │   │                      │   │                   │    ║
║   │ Runs the │   │ Reads all 27 SQLite  │   │ Applies scenario  │    ║
║   │ daily    │   │ summary tables and   │   │ filters/changes,  │    ║
║   │ pipeline │   │ returns daily vs     │   │ returns modified  │    ║
║   │ on demand│   │ historical KPIs      │   │ rows with context │    ║
║   └──────────┘   └──────────────────────┘   └───────────────────┘    ║
║                                                                      ║
║   Consumed by: supply_chain_delivery_app agents via MCP protocol     ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Project Structure

```
prediction_pipeline/
├── src/
│   ├── __init__.py                      # Exports all pipeline classes
│   ├── data_extract_1.py                # CSV loading and basic inspection
│   ├── data_eda_2.py                    # EDA — visualisations, statistics, outliers
│   ├── data_processing_3.py             # Cleaning, preprocessing, standardisation
│   ├── feature_engineering_4.py         # Feature creation (pre-split & post-split)
│   ├── model_evaluation_5.py            # Regression & classification metrics
│   ├── baseline_models_6.py             # Linear / Logistic Regression baselines
│   ├── regression_models_7.py           # Ridge, Lasso, RF, XGB, LGBM regression
│   ├── classification_models_8.py       # LR, RF, XGB, LGBM classification + GridSearch
│   ├── model_persistence_9.py           # Pickle save / load with metadata
│   ├── database_operations_10.py        # SQLite table management (27 tables)
│   ├── generate_daily_test_data_11.py   # Generates 3 synthetic test CSVs (5 000 rows each)
│   ├── simulate_delays.py               # Delay simulation logic — called by the MCP server
│   └── daily_predict.py                 # DailyPredictionPipeline — standalone entry point
├── config/
│   └── hyperparameters.py               # GridSearch parameter grids
├── models/
│   ├── best_classification_random_forest.pkl          # Stage 1 — delay Y/N (gitignored)
│   ├── best_classification_random_forest_metadata.json
│   ├── best_severity_random_forest.pkl                # Stage 2 — severity (gitignored)
│   └── best_severity_random_forest_metadata.json
├── db/
│   └── delivery_predictions.db          # SQLite database (auto-created, gitignored)
├── data/
│   ├── raw/                             # Source CSVs — Delivery_Logistics.csv + daily test files
│   └── processed/                       # Intermediate outputs
├── notebooks/
│   ├── train_predict_delay_model.ipynb  # Full training notebook (EDA → model selection)
│   └── ml-pipeline-doc.ipynb           # Pipeline documentation notebook
└── prediction_server.py                 # FastMCP server — exposes predict/diagnose/simulate tools
```

## Pipeline Stages

### Training (notebook)

Run `notebooks/train_predict_delay_model.ipynb` end-to-end to:

1. Load raw CSV (`DataExtract`)
2. Explore and visualize (`DataEDA`)
3. Clean and preprocess (`DataProcessing`)
4. Engineer features (`FeatureEngineering`) — creates interaction terms
   (`weight_x_distance`, `km_per_expected_hr`, `cost_per_km`), ordinal
   encodings (`weather_severity`, `mode_urgency`), and carrier / region
   aggregates (`schedule_risk`, `vehicle_load_strain`, etc.)
5. Train and evaluate multiple models (`BaselineModels`, `RegressionModels`,
   `ClassificationModels`, `ModelEvaluation`)
6. Save the best Stage 1 + Stage 2 models to `models/` (`ModelPersistence`)
7. Persist predictions and summaries to SQLite (`DatabaseOperations`)

### Daily Prediction (`daily_predict.py`)

Standalone 9-step pipeline that reuses trained models for new data:

```bash
# Run from prediction_pipeline/
cd prediction_pipeline
python -m src.daily_predict
python -m src.daily_predict --file data/raw/new_orders.csv
python -m src.daily_predict --file data/raw/new_orders.csv --if-exists append
```

Or call from Python (used by the delivery app's `predict_delays` tool):

```python
from src.daily_predict import DailyPredictionPipeline
result = DailyPredictionPipeline.run()
# result["df_predictions"]  — DataFrame with predictions
# result["df_engineered"]   — full engineered feature set
```

Steps: load CSV → clean → engineer features → load models → Stage 1 predict
(delay Y/N) → Stage 2 predict (severity) → save to CSV → refresh DB summaries.

### MCP Server (`prediction_server.py`)

FastMCP server (stdio transport) that exposes three tools to the agent layer:

| Tool | Backing function | Purpose |
|---|---|---|
| `predict` | `DailyPredictionPipeline.run()` | Run two-stage ML pipeline on new order CSV |
| `diagnose` | `DatabaseOperations` reads | Return all 27-table summaries for pattern analysis |
| `simulate` | `simulate_delays.py` | Apply scenario changes and return enriched rows |

Start the server before launching the delivery app:

```bash
# From the project root
python prediction_pipeline/prediction_server.py
```

## Database Schema

`DatabaseOperations` manages **27 tables** in `db/delivery_predictions.db`:

| Group | Tables | Description |
|---|---|---|
| Predictions | `daily_delivery_delay_prediction`, `hist_delivery_delay_prediction` | Row-level predictions (daily = new data, hist = training data with actuals) |
| Daily summaries | `daily_summary_by_partner`, `_by_package_type`, `_by_vehicle`, `_by_mode`, `_by_region`, `_by_weather`, `_by_severity`, `_by_partner_region`, `_by_mode_weather`, `_by_partner_severity`, `_by_distance_category`, `_overall` | Aggregated metrics for daily predictions |
| Historical summaries | Same 12 breakdowns prefixed `hist_summary_*` | Aggregated metrics for training data |
| Metadata | `metadata_dictionary` | Column descriptions for every table (auto-generated) |

## Engineered Features

Key features created by `feature_engineering_4.py`:

| Feature | Formula / Description |
|---|---|
| `weight_x_distance` | `package_weight_kg × distance_km` |
| `km_per_expected_hr` | `distance_km / (expected_time_hrs + ε)` — strongest predictor (r ≈ 0.59) |
| `cost_per_km` | `delivery_cost / (distance_km + ε)` |
| `weather_severity` | Ordinal encoding of `weather_condition` (clear=0 … stormy=3) |
| `mode_urgency` | Ordinal encoding of `delivery_mode` (standard=0 … same_day=3) |
| `schedule_risk` | `km_per_expected_hr × mode_urgency` — combined deadline pressure |
| `vehicle_load_strain` | `package_weight_kg / vehicle_capacity` |
| `carrier_avg_schedule` | Mean schedule risk per `delivery_partner` |
| `carrier_avg_weight` | Mean package weight per `delivery_partner` |
| `region_avg_distance` | Mean distance per `region` |

## Severity Labels

| Code | Label |
|---|---|
| 0 | No Delay |
| 1 | Short (1–2 h) |
| 2 | Medium (3–5 h) |
| 3 | Long (6+ h) |

## Prerequisites

- Python 3.11+
- Dependencies installed via `uv sync` from the project root (see root `pyproject.toml`)
- Project-root `.env` with these variables (paths relative to project root):

| Variable | Example |
|---|---|
| `SC_PREDICTION_MODEL_DIR` | `prediction_pipeline/models` |
| `SC_PREDICTION_DB_PATH` | `prediction_pipeline/db/delivery_predictions.db` |
| `SC_PREDICTION_SRC_DIR` | `prediction_pipeline` |

## Data Provenance

Delivery Logistics Dataset (India – Multi‑Partner) Link: https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner saved as Delivery_Logistics.csv

The primary training dataset contains 25,000 historical delivery records representing operations across India's five geographic regions and four delivery modes. The dataset reflects realistic patterns in last-mile logistics, including seasonal weather variation, partner performance variability, operational conditions, package characteristics, environmental factors, and delivery outcomes.

| Item | Detail |
|---|---|
| **Source** | `data/raw/Delivery_Logistics.csv` — [Kaggle: Delivery Logistics Dataset (India – Multi-Partner)](https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner) |
| **Type** | 25 000 historical delivery records, publicly available |
| **Licence** | Kaggle dataset licence (publicly available for research use) |
| **Daily test data** | `generate_daily_test_data_11.py` — stratified resample of the source CSV (5 000 rows, 74/26 on-time/delayed ratio); MIT (project-owned) |

## Test Data Generation

```bash
# Run from prediction_pipeline/
cd prediction_pipeline
python -m src.generate_daily_test_data_11
```

Produces 3 CSVs in `data/raw/` (5 000 rows each, 74 / 26 on-time / delayed
ratio) for testing the daily pipeline without production data.

# Data Architecture & Methodology

## Overview

The system operates across two data sub-systems: a batch ML prediction pipeline that processes logistics orders and writes structured results to SQLite, and a conversational agent application that reads from SQLite, retrieves SLA policy from a vector store, and writes output files for downstream use. Data flows in one direction — from raw input through the ML pipeline into persistent stores, then consumed by agents — with freshness sidecar files acting as cross-process coordination signals between the two sub-systems.

---

## 1. Data Sources

### 1.1 Historical Training Dataset

| Attribute | Detail |
|---|---|
| **Source** | Kaggle — E-Commerce Shipping Dataset |
| **Volume** | 25,000 orders × 15 raw columns |
| **Target variables** | `delayed` (binary: 0/1), `delay_hours` (continuous, for severity) |
| **Key raw features** | `delivery_partner`, `delivery_mode`, `region`, `weather_condition`, `vehicle_type`, `distance_km`, `package_weight_kg`, `delivery_cost`, `expected_time_hours` |
| **Licence** | Project-owned; restricted to evaluation use by upGrad, IIIT-B, and LJMU |
| **Purpose** | Train Stage 1 (binary delay classifier) and Stage 2 (severity classifier); build 12 historical summary tables in SQLite |

### 1.2 Daily Inference Data (Synthetic)

| Attribute | Detail |
|---|---|
| **Source** | `generate_daily_test_data_11.py` — synthetic batch generator |
| **Volume** | 3 × 5,000-row batches (configurable) |
| **Generation method** | Stratified sampling from training distribution with controlled delay rate injection |
| **Why synthetic** | Production live-order feeds require API integration and real-time infrastructure outside the scope of this demonstration; synthetic data preserves realistic statistical properties while enabling reproducible evaluation |
| **Purpose** | Daily inference input to the prediction pipeline; drives all agent pipeline runs |

### 1.3 SLA / Policy Corpus (RAG Knowledge Base)

| Attribute | Detail |
|---|---|
| **Source** | Project-authored SLA markdown document (`supply_chain_delivery_app/knowledge/`) |
| **Volume** | 36 sections → 73 chunks after two-stage LangChain splitting |
| **Content** | Delivery SLA targets, penalty clauses, partner obligations, regional commitments, escalation thresholds |
| **Why separate from structured data** | SLA clauses are unstructured policy text — not queryable via SQL. A vector store allows semantic retrieval against the specific delay patterns diagnosed by the ML pipeline |
| **Purpose** | Retrieved by `retrieve_sla_context()` during recommendation generation; each recommended action cites the specific SLA clause it addresses |

---

## 2. Preprocessing & Feature Engineering

### 2.1 Preprocessing Pipeline

The preprocessing pipeline runs in `data_processing_3.py` and applies the following transformations before any feature engineering:

| Step | Operation | Justification |
|---|---|---|
| **Label standardisation** | Lowercase and strip all categorical values | Prevents duplicate categories from case variation (e.g. "Express" vs "express") |
| **Date parsing** | Parse `order_date` and `delivery_date`; compute `slack_time` | Derived temporal features require parsed datetimes; `slack_time` = scheduled window − actual delivery time |
| **Missing value imputation** | Median for numeric, mode for categorical; no row drops | Daily inference requires a prediction for every order — dropping rows is not acceptable in a production pipeline |
| **Post-delivery leakage removal** | Drop `actual_delivery_time`, `delay_hours`, `slack_time` from inference features | These columns are only known after delivery and would constitute data leakage if used as model inputs |
| **Duplicate removal** | Drop exact-duplicate rows on `delivery_id` | Training data integrity |

### 2.2 Feature Engineering — 8 Derived Features

Feature engineering is the most analytically significant step. The top three model predictors by feature importance are all engineered features, not raw inputs — together accounting for over 63% of all model decisions.

#### Interaction Features

| Feature | Formula | Importance | Business Interpretation |
|---|---|---|---|
| `km_per_expected_hr` | `distance_km ÷ (expected_time_hours + ε)` | **27.1%** | Schedule tightness — the single strongest predictor. Overly optimistic delivery windows relative to route distance are the primary delay driver |
| `weight_x_distance` | `package_weight_kg × distance_km` | ~5% | Combined load-distance complexity proxy — heavier packages over longer routes strain vehicle capacity |
| `cost_per_kg` | `delivery_cost ÷ (package_weight_kg + ε)` | ~3% | Weight-adjusted pricing — under-priced heavy packages may be deprioritised by partners |

#### Ordinal Risk Features

| Feature | Encoding | Values | Importance | Purpose |
|---|---|---|---|---|
| `mode_urgency` | Ordinal (1–4) | 1=Standard, 2=Two-Day, 3=Express, 4=Same-Day | **21.5%** | Delivery mode urgency — Express and Same-Day compress the window and carry the highest SLA risk |
| `schedule_risk` | `weather_severity × mode_urgency` | 0–16 | **14.9%** | Compounded risk score — bad weather on a tight-deadline mode amplifies delay risk non-linearly |
| `weather_severity` | Ordinal (0–4) | 0=Clear, 1=Hot/Cold, 2=Foggy, 3=Rainy, 4=Stormy | ~7% | Direct weather risk input to the `schedule_risk` interaction |
| `vehicle_load_strain` | `(package_weight_kg × distance_km) ÷ vehicle_capacity` | ~10% | Overload indicator — normalises load burden by vehicle capability |

#### Group Aggregate Features

| Feature | Aggregation | Values | Importance | Purpose |
|---|---|---|---|---|
| `carrier_avg_schedule` | Mean `km_per_expected_hr` per `delivery_partner` (training set) | Continuous float | ~8% | Partner-level pattern — identifies carriers who systematically accept routes too tight for their fleet |

**Key finding:** Swapping between Random Forest, XGBoost, and LightGBM moved recall by 2–3 percentage points. Engineering `km_per_expected_hr` and `schedule_risk` alone moved recall by ~15 points. In structured tabular ML problems, domain-informed feature design has significantly higher ROI than model selection or hyperparameter tuning.

### 2.3 Feature Selection

After feature engineering, the following columns are dropped before modelling to eliminate collinearity and leakage:

| Dropped Column | Reason |
|---|---|
| `delivery_cost` | Absorbed into `cost_per_kg`; raw cost adds no independent signal |
| `cost_per_km` | Created during training FE but r > 0.85 with `km_per_expected_hr`; redundant |
| `expected_time_hrs_num` | Absorbed into `km_per_expected_hr` |
| `delivery_rating` | Post-delivery signal; not available at prediction time |

One-hot encoding (`drop_first=True`) is then applied to six categorical columns — `delivery_partner`, `package_type`, `vehicle_type`, `delivery_mode`, `region`, `weather_condition` — expanding the feature matrix to approximately 85–120 columns depending on cardinality. Encoding is applied **after** the train/test split to prevent data leakage.

---

## 3. Persistence Architecture & Decisions

### 3.1 Architecture Overview

```
RAW INPUT (CSV)
      │
      ▼
prediction_pipeline/  (process-isolated via MCP stdio)
      │
      ├──→ [writes] data/processed/daily_delivery_delay_prediction.csv   ← all 5K rows
      ├──→ [writes] daily_delivery_delay_prediction_meta.json            ← freshness sidecar
      └──→ [writes] SQLite: delivery_predictions.db  (27 tables)
                │
                ├──→ get_delay_diagnosis      (reads daily_* + hist_* summary tables)
                │         └──→ [app writes] diagnosis_meta.json          ← freshness sidecar
                │
                ├──→ simulate_order_delays    (reads daily CSV + hist_summary tables)
                │         └──→ [writes] simulate_delays_latest.csv
                │
                └──→ recommend_actions        (reads daily_summary tables)
                          │
                          └──→ ChromaDB vectorstore  (SLA policy chunks)
                                    └──→ [returns] top-8 SLA chunks → Recommend Agent

fetch_delayed_orders_for_email
      └──→ reads daily_delivery_delay_prediction.csv
                └──→ [writes] email_alerts.csv
```

### 3.2 SQLite — Structured Prediction Store

**Decision:** Use SQLite rather than a hosted database (PostgreSQL, MySQL) or in-memory DataFrames.

**Justification:** SQLite is file-based and requires no server process, making it the appropriate choice for a demonstration and development setting. The MCP server and agent application are separate processes — SQLite provides the shared persistent store without introducing a network dependency. 27 tables are written after every prediction run: the full prediction table, 12 `daily_*` dimension summary tables (region, partner, weather, vehicle type, mode, distance bucket), and 12 matching `hist_*` tables built from the training set. This structure allows the diagnosis and recommendation tools to perform SQL joins and aggregations directly against structured data rather than loading raw DataFrames into the agent context.

### 3.3 ChromaDB — SLA Vector Store

**Decision:** Persist SLA embeddings in ChromaDB with hash-based rebuild detection.

**Justification:** SLA policy is unstructured text that cannot be queried via SQL. A persistent ChromaDB collection avoids re-embedding the entire SLA corpus (36 sections, 73 chunks, 1536-dimension embeddings via `text-embedding-3-small`) on every application start — embedding all chunks via the OpenAI API adds 5–10 seconds of startup latency and unnecessary API cost. The collection is rebuilt only when the SHA-256 hash of the source SLA file changes, giving automatic freshness with no manual step. A two-stage LangChain chunking pipeline (`MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter`, 2000 chars / 200 overlap) is used to preserve section boundaries, ensuring retrieved chunks contain complete SLA clauses rather than mid-sentence fragments.

### 3.4 Filesystem CSVs and JSON Sidecars

**Decision:** Use file-based sidecars for cross-process freshness coordination rather than in-memory session state.

**Justification:** The Gradio application and the MCP prediction server are separate processes. In-memory state (e.g. `gr.State`) resets on browser reconnect, which would force a full pipeline re-run after every page refresh. File `mtime` checks against a 1-hour TTL survive process restarts, browser refreshes, and long sessions. The sidecars serve a dual role: freshness gate (read before each pipeline step to decide whether to skip) and audit record (the meta JSON contains the full prediction summary, readable after the session ends).

### 3.5 MCP Process Isolation

**Decision:** Expose the ML prediction pipeline via FastMCP over stdio transport rather than importing it directly into the agent application.

**Justification:** Process isolation creates a clean separation of concerns — the ML pipeline can be retrained, updated, or replaced without touching agent code, and agent logic can be modified without risking ML regressions. The stdio transport is lightweight and requires no network configuration. The isolation boundary also made debugging significantly easier during development: ML bugs (feature alignment errors, column order mismatches) surfaced as explicit MCP tool failures, never as silent corruption propagating through Pydantic models into the UI.

---

## 4. Data Flow Summary

| Layer | Technology | Data In | Data Out |
|---|---|---|---|
| Raw input | CSV file | Logistics orders (5K rows) | — |
| ML pipeline | Python + Random Forest | Raw CSV | Prediction CSV + 27 SQLite tables + meta sidecar |
| Diagnosis | SQLite queries | daily_* + hist_* tables | Comparison JSON + diagnosis sidecar |
| Simulation | SQLite + CSV | daily CSV + hist tables | Simulated scenarios CSV |
| Recommendation | SQLite + ChromaDB | Summary tables + SLA chunks | SLA-grounded action list |
| Email | CSV | Prediction CSV | Email alerts CSV |
| Agent layer | GPT-5.4 | Tool outputs (JSON) | Narrated Pydantic-typed `MasterOutput` |
| UI | Gradio | `MasterOutput` + CSVs | 5 rendered tabs |

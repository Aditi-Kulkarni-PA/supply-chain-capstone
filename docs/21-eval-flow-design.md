# Eval Flow Design — Supply Chain Delivery App

## Overview

This document describes the evaluation (eval) framework for the `supply_chain_delivery_app`. All eval code lives in the `evals/` directory. The existing app, agents, tools, and tests are untouched.

---

## Agent Inventory

| # | Agent | Tool Type | Tool Name | Output Model |
|---|-------|-----------|-----------|--------------|
| 1 | `predict_delivery_delays_agent` | MCP | `predict_delivery_delays` | `DeliveryDelayPredictionResult` |
| 2 | `diagnose_delay_patterns_agent` | MCP | `get_delay_diagnosis` | `DelayDiagnosisResult` |
| 3 | `delay_simulation_agent` | MCP | `simulate_order_delays` | `SimulationsList` |
| 4 | `recommendation_agent` | Function tool | `recommend_actions` + RAG (internal) | `RecommendedActionsList` |
| 5 | `email_alert_agent` | Function tool | `fetch_delayed_orders_for_email` | `EmailsList` |

The RAG pipeline (`retrieve_sla_context`) is embedded inside `recommend_actions` — not a separate agent tool, but evaluated via RAGAS in `test_eval_rag.py`.

`fallback_advisor_agent` and `format_summary_agent` are out of scope for evals.

---

## Design Principles

1. **Zero disruption**: `evals/` is a standalone directory. Nothing in `supply_chain_delivery_app/` is modified.
2. **Real LLM, controlled inputs**: Evals call real OpenAI APIs. DB and file I/O are redirected via env vars.
3. **Isolated eval DBs**: Evals write to `evals/db/delivery_predictions_eval.db` and `evals/db/delivery_predictions_eval_ragas.db`. Production DB is never touched.
4. **Full production input**: All evals run against the full 5 000-row input (`daily_delivery_logistics_1.csv`). The judge evaluates final text output — cost is determined by output length, not input size.
5. **LLM-as-judge on all 5 agents**: One LLM call per agent scoring relevance, faithfulness, and safety on a 1–5 scale.
6. **RAGAS on full pipeline path**: `retrieve_sla_context` is always called inside `recommend_actions()`. The RAGAS eval runs the full chain end-to-end and scores faithfulness and answer relevancy against the retrieved SLA context.
7. **Structured output as ground truth**: Pydantic models define the schema contract — passing schema validation is the baseline check for every agent.

---

## Directory Structure

```
0_supply_chain_capstone/
└── evals/
    ├── conftest.py              # shared pytest fixtures, report writer
    ├── pytest.ini               # registers ragas marker, asyncio_mode=auto
    ├── eval_config.py           # thresholds and dataset paths
    ├── judge.py                 # LLM-as-judge helper
    ├── test_eval_predict.py     # MCP: predict agent
    ├── test_eval_diagnose.py    # MCP: diagnose agent
    ├── test_eval_simulate.py    # MCP: simulate agent
    ├── test_eval_recommend.py   # Function tool: recommend agent + inline RAG grounding
    ├── test_eval_email.py       # Function tool: email agent
    ├── test_eval_rag.py         # RAGAS: full pipeline RAG path — gated @ragas
    ├── db/                      # eval-only SQLite DBs (never production)
    │   ├── delivery_predictions_eval.db        # standard agent evals
    │   └── delivery_predictions_eval_ragas.db  # RAGAS eval
    └── reports/                 # auto-generated markdown eval reports
```

---

## How to Run

From the `0_supply_chain_capstone/` directory:

| Command | What it runs |
|---------|-------------|
| `uv run python evals/run_evals.py` | Full suite — all 5 agents + RAGAS (recommended) |
| `uv run python evals/run_evals.py --agent recommend` | Single agent |
| `uv run pytest evals/test_eval_predict.py -v` | Single agent (direct pytest) |
| `uv run pytest evals/ -v` | Full suite via pytest directly |

The report is written automatically to `evals/reports/eval_report_<timestamp>.md` after each run.

---

## Shared Infrastructure

### DB Isolation

| DB file | Location | Purpose |
|---------|----------|---------|
| `delivery_predictions.db` | `prediction_pipeline/db/` | Production — never touched by evals |
| `delivery_predictions_eval.db` | `evals/db/` | Standard agent evals — full 5K-row input |
| `delivery_predictions_eval_ragas.db` | `evals/db/` | RAGAS eval |

Eval DBs live in `evals/db/` — they are eval infrastructure, not ML pipeline artifacts. The prediction pipeline writes to wherever `SC_PREDICTION_DB_PATH` is set; `conftest.py` overrides this to `evals/db/` before any test runs.

The `seeded_eval_db` fixture sets `SC_PREDICTION_DB_PATH` to the eval DB, copies `hist_*` tables from the production DB (read-only), and runs `DailyPredictionPipeline` to populate the `daily_*` tables. The RAGAS eval reuses the same eval DB — no separate large-scale seeding needed.

### Email Row Cap

`SC_EMAIL_MAX_ROWS` controls how many delayed orders are processed for email generation. Default: `10` for development, `3` for evals, `0` for unlimited (production).

### Fixture Execution Order

1. `set_eval_db` (autouse) — redirects DB path to eval DB and sets `SC_EMAIL_MAX_ROWS=3`
2. `pipeline_mcp_server` — starts the MCP stdio server
3. Predict eval runs first to populate the eval DB; downstream evals (diagnose, recommend, email) read from it

### LLM-as-Judge Rubric

The judge scores each agent's output on three dimensions (1–5 scale). Pass threshold: mean ≥ 3.0.

| Agent | Judge Criteria |
|-------|---------------|
| Predict | Quantitative derived-feature stats cited; cross-functional factors referenced (vehicle + weather + route) |
| Diagnose | Specific root cause patterns named; daily vs historical numbers cited |
| Simulate | Condition changes explained; severity changes attributed to specific factors |
| Recommend | Recommendations specific and data-backed; SLA references cite actual policy targets; categorization logically consistent |
| Email | Tone appropriate for severity level; template matches severity (Long/Medium/Short); personalized fields filled correctly |

---

## Per-Agent Eval Design

### 1. Predict Agent (`test_eval_predict.py`) — MCP tool

**Schema + content checks:**

| Check | Method |
|-------|--------|
| Tool called | `predict_delivery_delays` appears in result tool calls |
| Output schema valid | `DeliveryDelayPredictionResult` Pydantic validation |
| `predict_summary` non-empty | Length > 50 chars |
| `delayed_orders` non-empty | Count ≥ `MIN_DELAYED_ORDERS` |
| `llm_insights` references features | Each insight mentions ≥ 1 of: schedule risk, vehicle load strain, km per expected hour, vehicle type |
| Latency | < `MAX_PREDICT_LATENCY_S` |

**LLM-as-judge:** Passes `predict_summary` + 5 sample `llm_insights` to judge.

---

### 2. Diagnose Agent (`test_eval_diagnose.py`) — MCP tool

**Schema + content checks:**

| Check | Method |
|-------|--------|
| Tool called | `get_delay_diagnosis` appears in result tool calls |
| Output schema valid | `DelayDiagnosisResult` Pydantic validation |
| High-risk patterns non-empty | Count ≥ `MIN_HIGH_RISK_PATTERNS` |
| Comparison rows present | Count ≥ `MIN_COMPARISON_ROWS` |
| `diagnosis_summary` non-empty | Length > 50 chars |
| Risk levels valid | Each risk level in `{critical, high, medium}` |
| Latency | < `MAX_DIAGNOSE_LATENCY_S` |

**LLM-as-judge:** Passes `diagnosis_summary` to judge.

**Prerequisite:** Predict must have run first to populate the eval DB.

---

### 3. Simulate Agent (`test_eval_simulate.py`) — MCP tool

**Schema + content checks:**

| Check | Method |
|-------|--------|
| Tool called | `simulate_order_delays` appears in result tool calls |
| Output schema valid | `SimulationsList` Pydantic validation |
| Simulations returned | Count ≥ `MIN_SIMULATIONS` |
| Severity fields populated | Each `simulated_severity` non-empty |
| What-if produces change | At least one row where `original_severity != simulated_severity` |
| Latency | < `MAX_SIMULATE_LATENCY_S` |

**Test query:** `"Simulate delays for stormy weather in East region"`

**LLM-as-judge:** Passes `simulate_summary` + first 5 simulation rows to judge.

---

### 4. Recommendation Agent (`test_eval_recommend.py`) — Function tool + RAG

**Schema + content checks:**

| Check | Method |
|-------|--------|
| Tool called | `recommend_actions` appears in result tool calls |
| Output schema valid | `RecommendedActionsList` Pydantic validation |
| 9+ total recommendations | Count ≥ `MIN_RECOMMENDATIONS` |
| Category balance | `quick-win`, `short-term`, `long-term` counts each ≥ 3 |
| `sla_reference` populated | Each action has a non-empty SLA reference |
| Latency | < `MAX_RECOMMEND_LATENCY_S` |

**Inline RAG grounding (no extra API calls):**

| Check | Method |
|-------|--------|
| SLA context retrieved | Tool output contains SLA Knowledge Context block |
| SLA terms present | At least one `sla_reference` contains: SLA, OLA, penalty, threshold, or target |
| No hallucinated SLA | Each `sla_reference` has a 4+ word phrase traceable verbatim to `delivery_sla_github_ready.md` |

**LLM-as-judge:** Passes all `recommended_actions` to judge.

---

### 5. Email Agent (`test_eval_email.py`) — Function tool

**Schema + content checks:**

| Check | Method |
|-------|--------|
| Tool called | `fetch_delayed_orders_for_email` appears in result tool calls |
| Output schema valid | `EmailsList` Pydantic validation |
| At least 3 emails | Count ≥ `MIN_EMAILS` |
| Template correctness | Long → subject contains "Urgent"; Medium → "Moderate"; Short → "Minor" |
| `email_content` non-empty | Each email body length > 100 chars |
| Latency | < `MAX_EMAIL_LATENCY_S` |

**LLM-as-judge:** Passes 3 sample emails (one per severity level) to judge.

---

### 6. RAG Eval (`test_eval_rag.py`) — RAGAS on the real pipeline path

**Gated:** carries `@pytest.mark.ragas`. Included in the full `uv run pytest evals/ -v` run.

#### Why there is no standalone retrieval test

`retrieve_sla_context` is always called from inside `recommend_actions()`. Its input is the aggregated markdown built from DB stats, then condensed by `_summarize_query()` before embedding. Testing retrieval with hand-crafted standalone queries would bypass `_summarize_query()` and test a code path that never runs in production.

#### What it tests

Runs `recommendation_agent` against the production-scale RAGAS DB. Extracts the `--- SLA Knowledge Context ---` block from the tool output. Evaluates only the `sla_reference` fields against the SLA context block — scoping faithfulness to SLA grounding, not data-driven claims from the prediction DB.

**RAGAS Metrics:**

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `faithfulness` | SLA reference claims stay within the retrieved SLA context — no hallucinated policies | ≥ 0.55 |
| `answer_relevancy` | Recommendations address the actual delay patterns in the data | ≥ 0.60 |

---

## Eval Execution Flow

```
uv run pytest evals/ -v
  Session start
    └── seeded_eval_db       → SC_PREDICTION_DB_PATH = evals/db/delivery_predictions_eval.db
    │                          SC_EMAIL_MAX_ROWS = 3
    └── pipeline_mcp_server  → MCP stdio server started

  Execution order (full 5K-row input, eval DB):
    test_eval_predict.py     → predict_delivery_delays_agent    → schema + LLM-as-judge
    test_eval_diagnose.py    → diagnose_delay_patterns_agent    → schema + LLM-as-judge
    test_eval_simulate.py    → delay_simulation_agent           → schema + LLM-as-judge
    test_eval_recommend.py   → recommendation_agent             → schema + RAG grounding + LLM-as-judge
    test_eval_email.py       → email_alert_agent                → schema + LLM-as-judge
    test_eval_rag.py         → recommendation_agent (RAGAS DB)  → RAGAS faithfulness + answer_relevancy

  Session end
    └── evals/reports/eval_report_<timestamp>.md written
```

---

## What Does NOT Change

| File / Directory | Status |
|-----------------|--------|
| `delivery_agents.py` | untouched |
| `delivery_chat_app.py` | untouched |
| `tools/` | untouched |
| `config/` | untouched |
| `helpers/` | untouched |
| `prediction_pipeline/` | untouched |
| `tests/` | untouched — existing tests run independently |
| `delivery_predictions.db` | untouched — production DB, never read or written by evals |

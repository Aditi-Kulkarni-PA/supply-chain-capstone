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
6. **RAGAS on real recommendation output, sampled per-topic**: `recommend_actions()` runs once per eval, same as production. RAGAS then scores faithfulness/answer-relevancy per distinct SLA topic cited in the output (see §6) rather than one blended sample — see "Update (2026-07-12)" at the bottom of this doc below for why.
7. **Structured output as ground truth**: Pydantic models define the schema contract — passing schema validation is the baseline check for every agent.

---

## Directory Structure

```
0_supply_chain_capstone/
└── evals/
    ├── conftest.py              # shared pytest fixtures, report writer
    ├── pytest.ini               # asyncio_mode=auto (no ragas marker — RAGAS runs in the default suite)
    ├── eval_config.py           # thresholds and dataset paths
    ├── judge.py                 # LLM-as-judge helper
    ├── test_eval_predict.py     # MCP: predict agent
    ├── test_eval_diagnose.py    # MCP: diagnose agent
    ├── test_eval_simulate.py    # MCP: simulate agent
    ├── test_eval_recommend.py   # Function tool: recommend agent + inline RAG grounding
    ├── test_eval_email.py       # Function tool: email agent
    ├── test_eval_rag.py         # RAGAS: per-topic multi-sample eval over recommendation output
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
| `uv run python evals/run_evals.py` | Full suite — 5 agent evals + RAG/RAGAS + human-baseline calibration (recommended) |
| `uv run python evals/run_evals.py --agent <name>` | Single agent: predict / diagnose / simulate / recommend / email / rag |
| `uv run pytest evals/test_eval_predict.py -v` | Single agent (direct pytest) |
| `uv run pytest evals/ -v` | Full suite via pytest directly |

Notes:
- RAGAS metrics run as part of the standard suite (`test_eval_rag.py`); there is
  no separate gating flag. The former `--ragas` flag and `ragas` pytest marker
  were vestigial (no test carried the marker) and have been removed.
- A `master` single-agent option existed historically; `test_eval_master.py`
  was removed and the option with it. Master-level behaviour is exercised via
  the app flow and covered indirectly by the per-agent evals.

Each run writes to `evals/reports/`: `eval_report_<ts>.md`,
`judge_scores_<ts>.json` + merged `judge_scores_latest.json`, the raw pytest
`<ts>.json`, and (when the baseline tests run) `human_baseline_report_<ts>.md`.

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

### 6. RAG Eval (`test_eval_rag.py`) — RAGAS, sampled per SLA topic

**Not gated.** Runs as part of the default `uv run pytest evals/ -v` suite — no flag or marker needed.

#### Why per-topic sampling instead of one blended query

The recommendation agent's instruction is fixed (`"Recommend ways to optimize delivery timelines and reduce delays"`) — it never varies run to run, so varying *that* query wouldn't give RAGAS a meaningful spread of samples to average over. The original design ran `recommend_actions()` once, concatenated every `sla_reference` citation into one response string, and scored it against one shared retrieved-context block as a single `n=1` RAGAS sample. That is a genuinely noisy setup: RAGAS's Faithfulness metric has an LLM decompose the response into atomic claims and judge each against context, and with n=1 there is no averaging to smooth out that judge LLM's own run-to-run variance — observed faithfulness swung ~0.80–0.94 across identical, unchanged code.

The fix isn't a retrieval-depth tweak (we tested narrowing the final rerank to top-5 and broadening upstream retrieval — neither reliably improved or stabilized the score; broadening made the spread *worse*). Instead: the recommendation agent's **output** naturally cites several distinct SLA topics per run — a quick-win about weather policy, a short-term about partner benchmarks, a long-term about distance rules. `retrieve_sla_context()` gained an optional `query_override` param so the eval can issue a **separate, independently-retrieved query per topic**, bypassing `_summarize_query()`'s tool-output heuristic on purpose (that heuristic is still exercised for real inside `recommend_actions()` — this override exists only for eval sampling, not production).

#### What it tests

1. Runs `recommendation_agent` once against the eval DB (unchanged from before).
2. Selects up to 2 `recommended_actions` per category (quick-win / short-term / long-term) with a non-empty `sla_reference`.
3. For each selected action, builds a topic query — `"What does the SLA say about {dimension} — {action}?"` — and calls `retrieve_sla_context(tool_output="", query_override=topic_query)` to retrieve that topic's own SLA chunks independently.
4. Builds one RAGAS `SingleTurnSample` per topic (`response=action.sla_reference`, `retrieved_contexts=<that topic's chunks>`) — a real `n≈6` dataset.
5. Scores the mean faithfulness/answer_relevancy across all topics, and reports a per-topic breakdown table (category, dimension, action, faithfulness, relevancy) alongside the aggregate — so a low score is attributable to a specific SLA citation, not lost inside one blended number.

**RAGAS Metrics (mean across ~6 topic samples):**

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `faithfulness` | SLA reference claims stay within their own topic's retrieved SLA context — no hallucinated policies | ≥ 0.55 |
| `answer_relevancy` | Each `sla_reference` actually answers its topic's question | ≥ 0.60 |

**Known limitation:** averaging 6 topics did not eliminate run-to-run variance in the aggregate score (observed range ~0.76–0.91 faithfulness across repeated runs) — RAGAS's per-claim judging noise persists at this sample size. What changed is diagnostic: the per-topic table now shows *which* citation was weak on a given run (it isn't the same topic each time), which is actionable in a way a single blended score never was.

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
    test_eval_rag.py         → recommendation_agent (eval DB)   → RAGAS faithfulness + answer_relevancy, per SLA topic (n≈6)

  Session end
    └── evals/reports/eval_report_<timestamp>.md written
```

---

## What Does NOT Change

| File / Directory | Status |
|-----------------|--------|
| `delivery_agents.py` | untouched |
| `delivery_chat_app.py` | untouched |
| `tools/` | one addition: `rag_knowledge.py`'s `retrieve_sla_context()` gained an optional `query_override` param (see "Update (2026-07-12)" at the bottom of this doc) — production call sites don't pass it, so production behavior is unchanged |
| `config/` | untouched |
| `helpers/` | untouched |
| `prediction_pipeline/` | untouched |
| `tests/` | untouched — existing tests run independently |
| `delivery_predictions.db` | untouched — production DB, never read or written by evals |

---

## Update (2026-07-04) — Human Baseline Uses Latest Judge Scores

`pytest_sessionfinish` now also writes machine-readable judge scores to
`evals/reports/judge_scores_<timestamp>.json` and `judge_scores_latest.json`.
The human-baseline comparison (`test_eval_human_baseline.py`) reads the LLM
side from the judge's in-memory records when it runs inside the same pytest
session as the agent evals (no file write-ordering dependence), else from
`judge_scores_latest.json` — so it always reflects the most recent
eval run — while human scores continue to come from `human_scores.xls`
(never modified). The `llm_*` columns in the XLS are used only as a fallback
when no eval run has been recorded, and the report header states which source
was used.

---

## Update (2026-07-12) — Per-Topic RAGAS Sampling

**Also fixed the same session:** `test_eval_predict.py` was judging `predict_summary`
and per-row `llm_insights` as one blended text; the summary's length crowded the
insights out of the report's truncated output block. Split into two judge calls
(`Predict Delivery Delays — Summary` / `— LLM Insights`); `judge.grouped_scores()`
averages per-part records with a matching `" — "` prefix back into one row for the
top-level report table and `judge_scores_latest.json`, so the human-baseline
comparison still matches on the plain agent name. `conftest.py`'s report
truncation was raised 3000→5500 chars (also applied to the judge's own input
truncation in `judge.py`) so both parts render in full.

**RAG eval redesign:** `test_eval_rag.py` previously ran RAGAS on a single
blended sample (all `sla_reference` citations concatenated, scored against one
shared retrieved-context block) — genuinely `n=1`. Observed faithfulness swung
~0.80–0.94 across otherwise-identical runs (verified by rerunning with zero code
changes), which is expected for a single-sample LLM-judge metric but made a real
score drop indistinguishable from noise.

Two retrieval-depth hypotheses were tested and rejected before the redesign:
narrowing the final cross-encoder rerank stage from top-8 to top-5 (single run:
0.786 faithfulness — no better than baseline), and broadening upstream retrieval
from `TOP_K=15/HYBRID_PRE_FILTER_N=12` to `20/16` (two runs: 0.696 and 1.000 —
*wider* spread, not narrower). Both were reverted; `rag_knowledge.py`'s retrieval
constants are unchanged from the original committed values.

The actual fix, per repo owner direction: since the recommendation agent's own
instruction never varies, vary the *retrieval query* per distinct SLA topic the
agent's output already cites instead. See §6 above for the resulting design —
`query_override` on `retrieve_sla_context()`, up to 2 sampled actions per
category, one RAGAS sample per topic (n≈6), plus a per-topic breakdown table in
the report. `conftest.py`'s RAG→Recommendation merge logic (which folds RAGAS
scores into the `Recommendation Expert Agent` row when both tests run in one
session) was extended to also carry over the per-topic breakdown text, which the
merge previously discarded.

Net result: the aggregate score is still noisy run to run (observed ~0.76–0.91)
— per-topic sampling didn't fix that, and neither did the retrieval-depth
changes — but the report now shows *which* topic was weak on a given run, which
the single blended score never could.


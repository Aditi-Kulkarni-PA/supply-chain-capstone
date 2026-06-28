# AI-Powered Supply Chain Last-Mile Delivery Optimization

**IIIT Bangalore / upGrad — Post Graduate Diploma in ML & AI — Capstone Project**  
**Author:** Aditi Kulkarni

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Multi-Agentic AI? — Problem Fit Justification](#2-why-multi-agentic-ai--problem-fit-justification)
3. [Data Provenance](#3-data-provenance)
4. [System Overview](#4-system-overview)
5. [Architecture](#5-architecture)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Prompt Engineering](#8-prompt-engineering)
9. [MCP Server](#9-mcp-server)
10. [Agent Tools](#10-agent-tools)
11. [LLM Model Selection](#11-llm-model-selection)
12. [SQLite Database](#12-sqlite-database)
13. [ChromaDB / RAG Pipeline](#13-chromadb--rag-pipeline)
14. [Caching](#14-caching)
15. [Observability & Logging](#15-observability--logging)
16. [Security Guardrails](#16-security-guardrails)
17. [Documentation Index](#17-documentation-index)
18. [Setup](#18-setup)
19. [Testing](#19-testing)
20. [Evaluation](#20-evaluation)
21. [Strategic Deductions and Business Impact](#21-strategic-deductions-and-business-impact)
22. [Key Learnings](#22-key-learnings)
23. [Potential Future Extensions](#23-potential-future-extensions)
24. [Gradio UI App Screenshots](#24-gradio-ui-app-screenshots)
25. [Licence](#25-licence)

---
---

## 1. Problem Statement

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Last-mile delivery is the most expensive and delay-prone segment of the supply chain, accounting for 41–53% of total logistics costs (Statista, "Last Mile Share of Total Shipping Costs," 2018–2023 e-commerce benchmark). This project builds an AI system that predicts delivery delays before they happen, diagnoses root causes from historical patterns, simulates what-if scenarios, generates SLA-grounded optimization recommendations, and drafts personalised customer email alerts — all orchestrated through a conversational multi-agent interface.

---
---

## 2. Why Multi-Agentic AI? — Problem Fit Justification

<sub>[↑ Back to TOC](#table-of-contents)</sub>

This project deliberately combines classical ML with a multi-agent AI system, and each layer is justified by the nature of the task.

**The prediction layer uses classical ML** (two-stage Random Forest) because delay classification is a well-defined tabular task: fixed input features, binary and multi-class targets, and ground-truth labels for supervised training. A classifier is the right tool — fast, interpretable, and does not require language generation.

**The LLM agent layer addresses tasks where rule-based or deterministic systems would produce brittle, non-contextual outputs:**

- **Prediction inference narration** — explaining which feature combinations are causing delays and providing plain-English inference per record (`llm_insights`). A deterministic template cannot vary reasoning across 85+ features and their interactions.
- **Diagnosis narration** — identifying which feature combinations (e.g. `same_day × stormy × long distance`) drive today's delay spike requires synthesising across multiple summary tables and narrating the interaction. A heuristic cannot produce contextual root cause explanations.
- **Simulation reasoning** — translating a natural-language what-if scenario (*"what if weather turns stormy in the East region?"*) into database filters and enriching every result row with a cause-and-effect explanation requires language understanding and generation.
- **Recommendation generation** — producing non-generic, SLA-cited recommendations that reference specific penalty clauses, escalation tiers, and today's data requires grounding language generation in retrieved policy context (RAG). No rule-based system can do this without encoding every SLA clause as a hand-crafted rule.
- **Customer email drafting** — severity-appropriate, personalised email language requires natural language generation; template filling produces robotic, low-quality customer communication.

**The multi-agent orchestration layer is necessary — not incidental — for three reasons:**

1. **Task decomposition across specialist domains.** Prediction, diagnosis, simulation, recommendation, and alerting are five distinct capabilities with different tools, data sources, and output contracts. A single monolithic agent with all tools attached would receive conflicting instructions, mix output schemas, and be impossible to debug. Separate specialist agents — each with one tool, one Pydantic schema, one focused prompt — keep each capability independently testable and replaceable.

2. **Enforced sequential dependency chains.** Diagnosis requires prediction data in SQLite; recommendations require diagnosis patterns to ground SLA citations. Multi-agent orchestration makes these data dependencies explicit and machine-enforceable through the Master Orchestrator's prerequisite logic and freshness sidecar gates — a single-agent or pipeline approach cannot enforce this at the tool-call level.

3. **Natural language as the user interface across a complex workflow.** Operations teams need to query predictions, run simulations, and request recommendations through a single conversational interface without knowing which tools to invoke or in what order. The Master Orchestrator interprets intent, builds and confirms an action plan, routes to the correct specialist agents in sequence, and assembles structured outputs — a workflow that is not achievable with deterministic heuristics or a single-step LLM call.

The system avoids the common pitfall of forcing AI where simpler tools suffice. Classical ML handles structured tabular classification; deterministic Python handles email generation; the multi-agent layer handles tasks that require language understanding, sequential reasoning, and coordination across specialist domains.

---
---

## 3. Data Provenance

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### ML Training and Test Data

Delivery Logistics Dataset (India – Multi‑Partner) Link: https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner downloaded and saved as Delivery_Logistics.csv

The primary training dataset contains 25,000 historical delivery records representing operations across India's five geographic regions and four delivery modes. The dataset reflects realistic patterns in last-mile logistics, including seasonal weather variation, partner performance variability, operational conditions, package characteristics, environmental factors, and delivery outcomes.

| Item | Detail |
|---|---|
| **Source file** | `prediction_pipeline/data/raw/Delivery_Logistics.csv` |
| **Origin** | [Delivery Logistics Dataset (India – Multi-Partner)](https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner) — Kaggle, publicly available |
| **Type** | 25 000 historical delivery records across 5 regions, 4 delivery modes, multiple partners |
| **Licence** | Kaggle dataset licence (publicly available for research use) |
| **Daily test data** | `generate_daily_test_data_11.py` resamples and updates `Delivery_Logistics.csv` to produce daily inference batches (5 000 rows, 74/26 on-time/delayed ratio) — project-owned, restricted access |

### SLA / Policy Corpus (RAG Knowledge Base)

| Item | Detail |
|---|---|
| **Source file** | `supply_chain_delivery_app/knowledge/delivery_sla_github_ready.md` |
| **Type** | Custom-authored policy document |
| **Content** | 36-section SLA/OLA document covering performance targets, penalty thresholds, escalation tiers, partner benchmarks, weather policies, distance guidelines, and improvement priorities |
| **Modelled on** | Industry-standard last-mile delivery SLA frameworks |
| **Licence** | Project-owned, restricted access (evaluation only) |
| **Justification** | No publicly available, legally usable SLA document exists for last-mile delivery at the required granularity. A custom-authored document ensures the RAG corpus is legally usable, domain-appropriate, and contains the specific clause structure needed to ground recommendations in verifiable policy references. |

---
---

## 4. System Overview

<sub>[↑ Back to TOC](#table-of-contents)</sub>

The project is structured around two closely integrated sub-systems that together form an AI-assisted control tower for last-mile logistics operations.

### Sub-System 1 — ML Prediction Pipeline (`prediction_pipeline/`)

A self-contained Python module responsible for all machine learning concerns: data ingestion, feature engineering, model training, daily inference, and database writes. It exposes its capabilities to the agent layer through a FastMCP server over stdio transport — see [`docs/06-mcp-server-design.md`](docs/06-mcp-server-design.md) for tool contracts and design decisions.

Key responsibilities:
- Load raw CSV files, inspect data quality, and apply cleaning rules
- EDA for data understanding (19 features derived from 17 raw input columns)
- Model selection across Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM
- Finalised on Recall as primary metric
- Two-stage Random Forest inference: 
    - Stage 1 binary delay classification → identify orders that can get delayed
    - Stage 2 severity classification (Short / Medium / Long) for delayed orders only
- Persist predictions and aggregate summaries to SQLite (27 tables)
- Serve predictions to the delivery app via MCP stdio transport

**Key ML design decisions:**
- **Two-stage vs single multi-class model** — a single 4-class model (on-time / short / medium / long) would be dominated by the 73% on-time class. Splitting into two sequential stages lets Stage 1 independently maximise recall (catching all delays) and Stage 2 independently optimise severity accuracy across a more balanced 3-class problem.
- **Recall as primary optimisation metric** — a missed delay (false negative) is operationally far more costly than a false alarm (false positive): a missed delay means no corrective action; a false alarm means a slightly unnecessary intervention. GridSearchCV was run with `scoring='recall'` to align model selection with business cost asymmetry.

### Sub-System 2 — Supply Chain Delivery App (`supply_chain_delivery_app/`)

A Gradio web application providing the user interface and multi-agent orchestration logic. Users interact through a conversational chat panel; the Master Orchestrator agent interprets intent, calls the appropriate specialist sub-agents in sequence, and populates five output tabs with structured results.

Key responsibilities:
- Render the Gradio UI (chat panel + 5 result tabs)
- Detect user intent and map it to tool chains
- Orchestrate 8 AI agents using the OpenAI Agents SDK (Master Orchestrator + 6 specialists + Fallback Advisor)
- Retrieve SLA policy context via three-stage RAG (ChromaDB → hybrid scoring → cross-encoder reranking)
- Process structured Pydantic agent outputs and render them as interactive tables and Markdown
- LLM-as-judge and RAGAS used for agent and RAG performance evaluation

Taken together, the system enables logistics operations teams to query a conversational interface in natural language and receive structured, data-driven intelligence across five output domains: **predictions, root-cause diagnosis, what-if simulations, optimisation recommendations, and customer email alerts**.

---
---

## 5. Architecture

<sub>[↑ Back to TOC](#table-of-contents)</sub>

> Full diagram, agent inventory, and links to per-agent workflow docs:  
> **[`docs/01-architecture.md`](docs/01-architecture.md)**

```
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 1 — USER EXPERIENCE                                           ║
║                                                                      ║
║   Gradio Web UI  (delivery_chat_app.py)                              ║
║   ┌──────────┬───────────┬──────────┬─────────────┬───────────┐      ║
║   │ Predict  │ Diagnose  │ Simulate │ Recommend   │   Email   │      ║
║   └──────────┴───────────┴──────────┴─────────────┴───────────┘      ║
║   6 Quick-Action Buttons  │  Natural Language Chat Input             ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 2 — ORCHESTRATION  (OpenAI Agents SDK)                        ║
║                                                                      ║
║        ┌────────────────────────────────────┐                        ║
║        │   Master Expert Agent              │                        ║
║        │   • Planning & tool execution      │                        ║
║        │   • Files Freshness detection      │                        ║
║        │   • Prerequisite chain enforcement │                        ║
║        └──┬──────┬──────┬──────┬──────┬─────┘                        ║
║           │      │      │      │      │                              ║
║        Predict Diagnose Sim  Recommend Email  Fallback  Format       ║
║        Agent   Agent   Agent  Agent  Agent   Advisor   Summary       ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 3 — LLM  (OpenAI API)                                         ║
║                                                                      ║
║   GPT-5.4 (primary)              GPT-4.1-mini (formatting)           ║
║   • Reasoning & narration        • Summary rendering                 ║
║   • Structured outputs           • Markdown formatting               ║
║     (Pydantic v2 contracts)      • Lightweight inference             ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — TOOLS                                                     ║
║                                                                      ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    ║
║  │ FastMCP Server   │  │  RAG Tool        │  │  Email Tool      │    ║
║  │ (stdio transport)│  │  rag_knowledge   │  │  email_customers │    ║
║  │                  │  │  • cross-enc rerank │                  │    ║
║  │ • predict        │  │  • Hybrid search │  │  • Severity      │    ║
║  │ • diagnose       │  │    (cosine+BM25) │  │    templates     │    ║
║  │ • simulate       │  │  • Chunk & embed │  │  • Deterministic │    ║
║  └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘    ║
║           │                     │                                    ║
║  Two-Stage RF Pipeline    recommend_actions                          ║
║  Stage 1: Delay Y/N       (SQLite + RAG)                             ║
║  Stage 2: Severity                                                   ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 5 — KNOWLEDGE & PERSISTENCE                                   ║
║                                                                      ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    ║
║  │ SQLite DB        │  │ ChromaDB         │  │ File System      │    ║
║  │ 27 tables        │  │ (vectorstore/)   │  │ output/ CSVs     │    ║
║  │ • daily preds    │  │ SLA document     │  │ input/ orders    │    ║
║  │ • hist summaries │  │ 1536-dim embeds  │  │ models/ .pkl     │    ║
║  │ • aggregates     │  │ (text-embed-3-sm)│  │ knowledge/ .md   │    ║
║  └──────────────────┘  └──────────────────┘  └──────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 6 — OBSERVABILITY                                             ║
║                                                                      ║
║  Runtime logs      : supply_chain_delivery_app/log/*.log             ║
║  Prediction audit  : output/daily_delivery_delay_prediction_meta.json║
║  Diagnosis audit   : output/diagnosis_meta.json                      ║
║  Agent tracing     : OPENAI_AGENTS_DISABLE_TRACING=1 (configurable)  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Agent Inventory

| Agent | File | Tool | Responsibility |
|---|---|---|---|
| Master Expert | `master_expert.md` | all sub-agents | Orchestration, freshness detection, sequential execution |
| Predict | `predict_delivery_delays.md` | FastMCP `predict` | Two-stage RF prediction + per-row LLM enrichment |
| Diagnose | `diagnose_delay_patterns.md` | FastMCP `diagnose` | Daily vs historical pattern analysis, root cause narration |
| Simulate | `delay_simulation.md` | FastMCP `simulate` | What-if scenario translation and row enrichment |
| Recommend | `recommendation.md` | `recommend_actions` (RAG) | SLA-grounded 3-category recommendations |
| Email Alert | `email_alert.md` | `fetch_delayed_orders_for_email` | Severity-templated customer emails |
| Format Summary | `format_summary.md` | agent-as-tool | Structured Markdown rendering per output type |
| Fallback Advisor | `fallback_advisor.md` | WebSearchTool | Out-of-scope query handling |

---
---

## 6. Tech Stack

<sub>[↑ Back to TOC](#table-of-contents)</sub>

> Full stack with detailed comments:  
> **[`docs/02-tech-stack.md`](docs/02-tech-stack.md)**

| Category | Technology / Library | Comment |
|---|---|---|
| LLM backbone | GPT-5.4, GPT-4.1-mini | Primary reasoning + lightweight formatting |
| Agents & orchestration | OpenAI Agents SDK | 7 agents including Master Orchestrator |
| Structured I/O | Pydantic v2 | Typed schemas enforced across all agents |
| MCP server | FastMCP (stdio) | predict / diagnose / simulate tools |
| Web UI | Gradio | 5-tab conversational interface |
| ML models | Scikit-learn Random Forest | Two-stage: delay Y/N + severity (Short/Medium/Long) |
| Feature Engineering | Pandas, NumPy, SciPy | 10+ engineered features |
| Database | SQLite3 | 27 tables — predictions + aggregates |
| Vector Store | ChromaDB | SLA document RAG storage (1536-dim) |
| Embeddings | OpenAI text-embedding-3-small | Document and query embeddings |
| RAG chunking | langchain-text-splitters | Markdown + recursive character splitting |
| RAG retrieval | Hybrid scoring (cosine + keyword) | 70% cosine + 30% BM25, top-12 |
| Cross-encoder Reranker | sentence-transformers | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-8 |
| Configuration | python-dotenv, YAML | Secrets + hyperparameter management |
| Package Manager | uv / uv.lock | Reproducible installs; requirements.txt exported |
| Testing | Pytest, pytest-asyncio | MCP server smoke tests (11 tests), Pydantic model tests (11 tests), RAG knowledge tests (7 tests) |
| Evaluation | Pytest, RAGAS, LLM-as-judge | 5-agent eval suite + RAGAS faithfulness/relevancy scoring |
| Python | 3.11+ | Runtime |

---
---

## 7. Project Structure

<sub>[↑ Back to TOC](#table-of-contents)</sub>

```
0_supply_chain_capstone/
├── README.md                          ← This file
├── pyproject.toml                     ← uv project config (package = false)
├── requirements.txt                   ← Pinned dependency list (uv export)
├── uv.lock                            ← uv lockfile for reproducible installs
├── .env                               ← Local secrets (gitignored)
├── .env.example                       ← Template — safe to commit
├── .gitignore                         ← Files to be ignored while comitting to Git repo
├── LICENSE                            ← License information
│
├── assets/
│   ├── screenshots/                   ← Demo screenshots (predict, diagnose, simulate, recommend, email)
│   └── video/                         ← Demo GIF files per agent tab (01_predict.gif … 05_email_alert.gif)
│
├── docs/
│   ├── 01-architecture.md             ← Layered architecture diagram + agent inventory
│   ├── 02-tech-stack.md               ← Full technology stack with library details
│   ├── 03-prompt-evolution-log.md     ← Prompt engineering iteration history
│   ├── 04-ml-pipeline-architecture.md ← ML training pipeline flow diagram
│   ├── 05-sqlite-database-design.md   ← SQLite schema: 27 tables, summary types, write lifecycle
│   ├── 06-mcp-server-design.md        ← FastMCP server: tool contracts, transport, design decisions
│   ├── 07-vectorstore-rag-design.md   ← ChromaDB store, three-stage RAG, chunking, retrieval cache
│   ├── 08-master-agent-flowchart.md   ← Master orchestrator logic flowchart
│   ├── 09-agent-predict-workflow.md   ← Predict agent flow + design fixes
│   ├── 10-agent-diagnose-workflow.md  ← Diagnose agent flow + freshness design
│   ├── 11-agent-simulate-workflow.md  ← Simulation agent flow + severity lookup
│   ├── 12-agent-recommend-workflow.md ← Recommend agent + RAG pipeline design
│   ├── 13-agent-email-workflow.md     ← Email agent flow + severity templates
│   ├── 14-gradio-ui-design.md         ← UI component selection rationale
│   ├── 15-security-guardrails-design.md ← Attack surface analysis
│   ├── 16-caching-design.md           ← Sidecar freshness detection + RAG retrieval cache design
│   ├── 17-observability-logging.md    ← Runtime logger, audit sidecars, output files, tracing config
│   ├── 18-end-to-end-data-flow.md     ← Full data flow from CSV input to UI output
│   ├── 19-format-agent-design.md      ← Format Summary agent design and Markdown rendering
│   ├── 20-executive-flow-diagrams.md  ← Executive-level flow diagrams for documentation
│   └── 21-eval-flow-design.md         ← Eval framework design: agent evals, RAGAS, LLM-as-judge
│
├── prediction_pipeline/               ← ML training + inference module
│   ├── README.md
│   ├── prediction_server.py           ← FastMCP server (predict/diagnose/simulate tools)
│   ├── config/
│   │   └── hyperparameters.py
│   ├── src/
│   │   ├── data_extract_1.py          ← CSV loading
│   │   ├── data_eda_2.py              ← EDA, visualisations
│   │   ├── data_processing_3.py       ← Cleaning, preprocessing
│   │   ├── feature_engineering_4.py   ← Feature creation (interaction terms, ordinal encodings)
│   │   ├── model_evaluation_5.py      ← Metrics (classification + regression)
│   │   ├── baseline_models_6.py       ← Linear / Logistic baseline
│   │   ├── regression_models_7.py     ← Ridge, Lasso, RF, XGB, LGBM regression
│   │   ├── classification_models_8.py ← LR, RF, XGB, LGBM + GridSearch
│   │   ├── model_persistence_9.py     ← Pickle save/load with metadata
│   │   ├── database_operations_10.py  ← SQLite (27 tables)
│   │   ├── generate_daily_test_data_11.py ← Synthetic daily CSVs for testing
│   │   ├── simulate_delays.py         ← Simulation logic
│   │   └── daily_predict.py           ← DailyPredictionPipeline entry point
│   ├── models/                        ← Trained .pkl files (gitignored)
│   ├── data/
│   │   ├── raw/                       ← Source CSVs (Delivery_Logistics.csv + daily test files)
│   │   └── processed/                 ← Intermediate outputs
│   ├── db/                            ← SQLite database (gitignored)
│   └── notebooks/
│       └── train_predict_delay_model.ipynb  ← Run once to train models
│       (ml-pipeline-doc.ipynb → see docs/04-ml-pipeline-architecture.md)
│
├── tests/                             ← Smoke & unit tests (run offline, no OpenAI API)
│   ├── conftest.py                    ← Markdown report writer (pytest_terminal_summary hook)
│   ├── test_mcp_server.py             ← 11 tests: MCP tools called as async Python functions
│   ├── test_pydantic_models.py        ← 11 tests: Pydantic output model validation
│   └── test_rag_knowledge.py          ← 7 tests: ChromaDB collection, SLA file, keyword retrieval
│
├── evals/                             ← LLM-as-judge + RAGAS eval suite (calls OpenAI API)
│   ├── __init__.py
│   ├── conftest.py                    ← Fixtures: DB isolation, MCP server lifecycle, report writer
│   ├── pytest.ini                     ← asyncio_mode=auto, ragas marker
│   ├── eval_config.py                 ← Pass thresholds and dataset paths
│   ├── judge.py                       ← LLM-as-judge helper (gpt-4.1-mini)
│   ├── llm_judge_eval.py              ← Standalone LLM-judge runner with human baseline comparison
│   ├── run_evals.py                   ← Entry point: full suite or single-agent run
│   ├── test_eval_predict.py           ← Predict agent: schema + LLM judge
│   ├── test_eval_diagnose.py          ← Diagnose agent: schema + LLM judge
│   ├── test_eval_simulate.py          ← Simulate agent: schema + LLM judge
│   ├── test_eval_recommend.py         ← Recommend agent: schema + RAG grounding + LLM judge
│   ├── test_eval_email.py             ← Email agent: schema + LLM judge
│   ├── test_eval_rag.py               ← RAGAS faithfulness + answer_relevancy
│   ├── test_eval_human_baseline.py    ← Human baseline comparison eval
│   ├── db/                            ← Eval-only SQLite DB (never production)
│   │   └── delivery_predictions_eval.db  ← Shared DB for agent evals + RAGAS
│   ├── human_baseline/
│   │   └── human_scores.xls           ← Human-reviewed scores for judge calibration
│   └── reports/                       ← Auto-generated eval reports (markdown + JSON)
│
└── supply_chain_delivery_app/         ← Multi-agent app module
    ├── README.md
    ├── delivery_chat_app.py            ← Gradio UI entry point
    ├── delivery_agents.py              ← Pydantic models + agent definitions
    ├── config/
    │   ├── load_config.py             ← build_instruction() — assembles layered prompts
    │   └── prompts/
    │       ├── agents/                ← 7 agent-specific markdown prompts
    │       │   ├── master_expert.md
    │       │   ├── predict_delivery_delays.md
    │       │   ├── diagnose_delay_patterns.md
    │       │   ├── delay_simulation.md
    │       │   ├── recommendation.md
    │       │   ├── email_alert.md
    │       │   └── fallback_advisor.md
    │       └── shared/                ← 3 cross-cutting prompts
    │           ├── security_guardrails.md
    │           ├── chatbot_behavior.md
    │           └── format_summary.md
    ├── tools/
    │   ├── rag_knowledge.py           ← ChromaDB + hybrid retrieval (cosine + keyword)
    │   ├── recommend_actions.py       ← SQLite reads + RAG retrieval
    │   └── email_customers.py         ← Severity-based email template generation
    ├── helpers/
    │   ├── app_utils.py
    │   ├── post_processing.py
    │   └── logging_utils.py
    ├── knowledge/
    │   └── delivery_sla_github_ready.md  ← 36-section SLA/OLA policy document
    ├── vectorstore/                   ← ChromaDB persistent store (gitignored)
    ├── input/                         ← Daily order CSVs for prediction
    ├── output/                        ← Generated predictions, emails, simulations (gitignored)
    ├── log/                           ← Runtime application logs
    └── notebooks/
        ├── logic-workflow-fixes.ipynb   ← Source for agent workflow docs in docs/
```

---
---

## 8. Prompt Engineering

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Prompt instructions are split across three composable files — `security_guardrails.md`, `chatbot_behavior.md`, and `master_expert.md` — assembled at runtime by `build_instruction()` in `supply_chain_delivery_app/config/load_config.py`; rather than hardcoded as a single monolithic string. This separation keeps each concern independently editable and testable: guardrail rules can be tightened without touching agent logic, persona and tone can be adjusted without risking security regressions, and domain expertise can be iterated on without re-reviewing the full prompt. The layered order (security first, always) enforces a clear precedence hierarchy that a single flat prompt cannot guarantee. The design evolved through 17 iterations across 7 agents, each driven by a specific failure mode observed during development.

### Prompt Version History

| Phase | Change | Problem Solved | Files Affected |
|---|---|---|---|
| 1 → 2 | Extracted security, behaviour, formatting into shared files | Maintenance burden, security deprioritisation, formatting drift | All prompt files, `load_config.py` |
| 2 → 3.1 | Added `llm_insights` enrichment instruction | Per-row explanations missing entirely | `predict_delivery_delays.md` |
| 3.1 → 3.2 | Added derived feature glossary | Generic, non-specific insights | `predict_delivery_delays.md` |
| 3.2 → 3.3 | Narrowed output to 2-field schema | Parsing errors, missed rows, extra fields | `predict_delivery_delays.md` |
| 3.3 → 3.4 | Added 2 few-shot examples | Near-identical insights across rows | `predict_delivery_delays.md` |
| 3.4 → 3.5 | Enforced ≥2 feature rule + `predict_summary` template | Single-dimension reasoning, weak summaries | `predict_delivery_delays.md` |
| 4.1 | Pydantic split: deterministic structure + LLM prose only | Short, unstructured diagnosis summaries | `diagnose_delay_patterns.md` |
| 4.2 | Field glossary for derived columns | Root cause missing; surface-level observations only | `diagnose_delay_patterns.md` |
| 4.3 | Mandatory section ordering + formatting rules | Inconsistent summary layout between runs | `diagnose_delay_patterns.md` |
| 5.1 | Tool returns dimension-level + combo-level data | Generic recommendations with no specifics | `recommendation.md`, tool |
| 5.2 | RAG-retrieved SLA context appended to tool output | Recommendations ungrounded in SLA/OLA | `recommendation.md`, tool |
| 5.3 | Three-category structure (quick-win/short/long-term) | No temporal prioritisation | `recommendation.md` |
| 6.1 | Email generation moved to tool (deterministic Python) | Format inconsistency, template invention, coverage gaps | `email_alert.md`, tool |
| 6.2 | Negative constraints: no individual generation | Agent re-generating emails on top of tool output | `email_alert.md` |
| 7.1 | Valid values list for filter/change parameters | Tool errors from invalid filter values | `delay_simulation.md` |
| 7.2 | Row parity rule ("same number of rows as tool") | Partial enrichment (5-10 rows out of 50+) | `delay_simulation.md` |
| 7.3 | Pydantic fields enforce concise comparative reason format | Free-form verbosity, inconsistent reason length | `delay_simulation.md` |

> Full iteration details, design rationale, and prompt examples: **[`docs/03-prompt-evolution-log.md`](docs/03-prompt-evolution-log.md)**

### Per-Agent Workflow Documents

- **[`docs/08-master-agent-flowchart.md`](docs/08-master-agent-flowchart.md)** — Master agent flow: plan confirmation gate, sequential tool execution, freshness detection; Fallback Advisor: trigger conditions, WebSearchTool usage, design rationale
- **[`docs/09-agent-predict-workflow.md`](docs/09-agent-predict-workflow.md)** — Predict agent flow: schema slimming, sidecar JSON, few-shot examples
- **[`docs/10-agent-diagnose-workflow.md`](docs/10-agent-diagnose-workflow.md)** — Diagnose agent flow: freshness gate design
- **[`docs/11-agent-simulate-workflow.md`](docs/11-agent-simulate-workflow.md)** — Simulate agent flow: 3-arg tool, FRESH/NOT FRESH prerequisite
- **[`docs/12-agent-recommend-workflow.md`](docs/12-agent-recommend-workflow.md)** — Recommend agent flow: SLA citation, category enforcement, RAG grounding
- **[`docs/13-agent-email-workflow.md`](docs/13-agent-email-workflow.md)** — Email agent flow: severity-specific templates, single EmailAlert pattern
- **[`docs/15-security-guardrails-design.md`](docs/15-security-guardrails-design.md)** — Security guardrails: why they live only on the Master agent (Fallback Advisor coverage included)

---
---

## 9. MCP Server

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Exposing the prediction pipeline through an MCP server rather than importing it directly into the agent, creates a clean separation of concerns: the pipeline can be updated, retrained, or replaced without touching agent code, and the agent layer can be modified without risking changes to ML logic. The stdio transport is also lightweight and requires no network configuration, making it suitable for a development and demonstration setting.

Three tools over **stdio transport** (FastMCP) expose the ML prediction pipeline to the agent layer — keeping the two sub-systems process-isolated with no direct Python imports between them. `get_delay_diagnosis` and `simulate_order_delays` enforce the predict-first constraint machine-enforceably: they return an explicit error if prediction artifacts are missing.

| Tool | Input | Prerequisite | Returns |
|---|---|---|---|
| `predict_delivery_delays` | CSV file path | Skipped if prediction sidecar (`daily_delivery_delay_prediction_meta.json`) is less than 1 hour old — master injects `[SYSTEM: FRESH]` tag and bypasses the tool | Aggregate summary + all delayed rows with `delay_reason`; refreshes all 27 SQLite tables |
| `get_delay_diagnosis` | none | `predict_delivery_delays` must have run — returns `{"Error": "upstream_missing"}` if prediction CSV or `daily_summary_overall` table is absent | Daily vs historical comparison across 7 dimensions + high-risk patterns |
| `simulate_order_delays` | `scenario`, `filters`, `changes` (JSON) | `predict_delivery_delays` must have run — returns `{"Error": "upstream_missing"}` if prediction artifacts are absent | Filtered delayed orders with reassigned severity + Markdown summary |

> Full tool contracts, stdio transport architecture, `tool_filter` design, connection setup, and timeout rationale: **[`docs/06-mcp-server-design.md`](docs/06-mcp-server-design.md)**

---
---

## 10. Agent Tools

<sub>[↑ Back to TOC](#table-of-contents)</sub>

The system splits tool execution across two mechanisms by responsibility boundary: MCP tools (`predict_delivery_delays`, `get_delay_diagnosis`, `simulate_order_delays`) own the ML pipeline and run in a separate child process, while function tools (`recommend_actions`, `generate_email_alerts`) run in-process and operate purely on data the pipeline has already written to SQLite. This split is intentional — MCP isolation protects the ML layer from agent-side failures and allows the prediction server to be restarted or swapped independently, while in-process function tools avoid the serialization overhead and subprocess latency that would be wasteful for operations that are already data-access only. The result is a tool boundary that mirrors the data dependency graph: compute-heavy, stateful ML work goes over stdio; lightweight read-and-transform work stays local.

The Master Orchestrator calls tools across two categories: ML pipeline tools exposed via FastMCP (running in a separate process), and application-layer tools registered as Python functions directly with the agent.

### ML Pipeline Tools (via FastMCP)

These invoke the two-stage Random Forest pipeline through the MCP stdio transport. Each tool runs the ML inference logic in `prediction_pipeline/` and returns structured results to the agent layer — with no direct Python imports between sub-systems.

| Tool | ML Operation |
|---|---|
| `predict_delivery_delays` | Stage 1: RF binary classifier (delayed Y/N) → Stage 2: RF severity classifier (Short / Medium / Long) for delayed rows only. Writes all 27 SQLite tables. Returns aggregate summary + all delayed rows with `delay_reason`. |
| `get_delay_diagnosis` | Aggregates SQLite `daily_*` and `hist_*` summary tables across 7 dimensions (partner, mode, weather, vehicle, region, distance, package type). Returns high-risk patterns and daily-vs-historical comparison. Fails explicitly if prediction artifacts are missing. |
| `simulate_order_delays` | Applies `scenario`, `filters`, and `changes` parameters to filter SQLite delayed rows and reassign severity under the hypothetical conditions. Fails explicitly if prediction artifacts are missing. |

> Tool contracts, parameter schemas, and transport design: **[`docs/06-mcp-server-design.md`](docs/06-mcp-server-design.md)**

### Application-Layer Tools

These run in-process within the `supply_chain_delivery_app/` module and are registered directly with agent instances.

| Tool | File | Used by | What it does |
|---|---|---|---|
| `recommend_actions` | `tools/recommend_actions.py` | Recommend agent | Queries SQLite for dimension-level + combo-level delay stats, then calls `retrieve_sla_context()` for 3-stage RAG retrieval. Returns grounded context for SLA-cited recommendations. |
| `retrieve_sla_context` | `tools/rag_knowledge.py` | `recommend_actions` | Three-stage RAG: ChromaDB cosine similarity → hybrid scoring (cosine + keyword) → cross-encoder rerank. Returns top-8 SLA policy chunks. |
| `fetch_delayed_orders_for_email` | `tools/email_customers.py` | Email agent | Queries SQLite for delayed orders. Applies severity-based email templates in deterministic Python (no LLM generation). Returns one `EmailAlert` per delayed order. |
| `WebSearchTool` | OpenAI Agents SDK (built-in) | Fallback Advisor only | Web search for supply-chain-adjacent queries that fall outside the system's five operational domains. Isolated to Fallback Advisor — operational agents cannot call it. |

---
---

## 11. LLM Model Selection

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Model selection was driven by observed failure modes during development, not by upfront assumption.

GPT-4.1-mini and GPT-5.1 were both evaluated before settling on the current two-model split. GPT-4.1-mini produced silent failures on the reasoning-heavy agents — it would complete without error but return incomplete Pydantic objects, drop fields under large context, or skip tool calls it was instructed to make. GPT-5.1 introduced non-silent errors: API and runtime errors. 

**GPT-5.4 resolved both failure modes** — it reliably processes the full layered instruction stack, holds structured output contracts across long tool chains, and does not silently degrade when context grows. It is therefore used for every agent that reasons, decides, or produces typed outputs. 

**GPT-4.1-mini is retained for two narrowly scoped roles** where its failure modes do not apply: the Format Summary agent, whose task is deterministic Markdown rendering of an already-validated structured object with small, predictable context; and the LLM-as-judge in evals, which scores outputs against a fixed rubric — pattern-matching over a defined schema rather than open-ended inference. Both model assignments are environment-variable controlled (OPENAI_MODEL / OPENAI_MODEL_MINI), so the split can be recalibrated without touching agent code.

| Model | Failure Mode | Outcome |
|---|---|---|
| **GPT-4.1-mini** | Context window insufficient for the full layered prompt stack (security + behaviour + agent-specific). Gave compressed, incomplete summaries. Randomly skipped individual instructions from the prompt. `llm_insights` in the Predict agent was sometimes entirely absent, sometimes generated for only a handful of rows. Pipeline halted mid-execution with no error message. | Abandoned — not viable for multi-instruction agentic prompts |
| **GPT-5.1** | The same code that ran stably on GPT-4.1-mini threw API and runtime errors against GPT-5.1. No change to tool contracts or agent config — model introduced instability. | Abandoned — introduced regressions on working code |
| **GPT-5.4** | Full prompt context covered without skipped instructions. `llm_insights` generated consistently for all rows. Sequential tool chain (predict → diagnose → recommend) runs end-to-end reliably. | **Selected — current production model** |

---
---

## 12. SQLite Database

<sub>[↑ Back to TOC](#table-of-contents)</sub>

SQLite was chosen over heavier relational databases for this project due to its zero-configuration deployment model. The 27-table schema provides all the aggregation needed for diagnosis without requiring a separate analytics layer. The pre-computed summary tables shift the computational cost from query time to write time, ensuring that Diagnosis Agent tool calls return quickly even as data volumes grow.

A single database (`prediction_pipeline/db/delivery_predictions.db`) is the shared data store. The prediction pipeline writes to it; the agent layer reads from it via MCP tools and the `recommend_actions` tool.

**27 tables = 2 raw prediction tables + 24 summary tables + 1 metadata dictionary**

| Group | Tables | Written by |
|---|---|---|
| Raw predictions | `hist_delivery_delay_prediction`, `daily_delivery_delay_prediction` | Training notebook (hist, once) / daily inference (daily, each run) |
| Summary tables | 12 types × `hist_` + `daily_` prefix | Same — rebuilt on every daily prediction run |
| Metadata | `metadata_dictionary` | Auto-generated — describes every column in every table |

The 12 summary types cover: 6 single-dimension breakdowns (partner, package, vehicle, mode, region, weather), 1 distance-category breakdown, 3 two-dimension combinations (mode×weather, mode×distance, weather×vehicle), 1 overall KPI table, and 1 high-risk pattern table (delay rate ≥ 30%, classified as medium / high / critical).

`hist_*` tables are written once at training time and stay static. `daily_*` tables are replaced on every prediction run.

> Full table inventory, column schema, engineered features, high-risk pattern classification, agent read patterns, and write lifecycle: **[`docs/05-sqlite-database-design.md`](docs/05-sqlite-database-design.md)**

---
---

## 13. ChromaDB / RAG Pipeline

<sub>[↑ Back to TOC](#table-of-contents)</sub>

ChromaDB was selected as the vector store due to its file-system persistence model (no external server required) and seamless integration with LangChain's text splitting utilities. The file-hash based cache invalidation ensures that the vector store is automatically refreshed if the SLA document is updated, without requiring manual re-indexing.

The Recommend agent retrieves SLA policy context from a persistent ChromaDB vector store (`supply_chain_delivery_app/vectorstore/`) using a three-stage pipeline to maximise retrieval precision. This design keeps the fast bi-encoder retrieval for broad candidate selection while applying the more accurate cross-encoder only on the shortlist.. The store auto-rebuilds if the source SLA file changes (file-hash detection).

**Source:** `supply_chain_delivery_app/knowledge/delivery_sla_github_ready.md` — 36-section SLA/OLA document  
**Embedding model:** `text-embedding-3-small` (1536-dim, OpenAI)  
**Chunking:** `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (500-token chunks, 200-char overlap, header breadcrumb prepended)

### Three-Stage Retrieval

| Stage | Method | Input → Output |
|---|---|---|
| 1 — Broad retrieval | ChromaDB cosine similarity | Query embedding → top-15 chunks |
| 2 — Hybrid pre-filter | 0.7 × cosine + 0.3 × keyword overlap | top-15 → top-12 |
| 3 — Cross-encoder rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | top-12 → top-8 |

The `recommend_actions` tool output is summarised into a focused query before embedding (avoids token-limit issues with full output). Retrieved chunks are injected between `--- SLA Knowledge Context ---` delimiters and passed to the Recommend agent prompt.

> Full pipeline design, chunking strategy, hybrid scoring formula, cross-encoder rationale, retrieval cache, and output format: **[`docs/07-vectorstore-rag-design.md`](docs/07-vectorstore-rag-design.md)**  
> Per-agent usage in context: **[`docs/12-agent-recommend-workflow.md`](docs/12-agent-recommend-workflow.md)**

---
---

## 14. Caching

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Four independent caching mechanisms prevent redundant work at different layers:

| Mechanism | What it prevents | Why this approach |
|---|---|---|
| **Sidecar Freshness Detection** | Re-running expensive MCP tools (`predict_delivery_delays`, `get_delay_diagnosis`) when outputs are already current — injects `[SYSTEM: FRESH/NOT FRESH]` tag; action plan marks skipped steps explicitly | MCP tool calls trigger full ML inference + 27-table SQLite writes. File mtime is the simplest reliable cross-process signal — survives Gradio state resets and doubles as an audit sidecar |
| **ChromaDB Embedding Cache** | Re-embedding the entire SLA document (36 sections, ~500-token chunks) on every app start — `PersistentClient` stores 1536-dim chunk embeddings to disk; rebuilds only when source file SHA-256 hash changes | Embedding all SLA chunks via the OpenAI API adds 5–10 seconds of startup latency and unnecessary API cost. Persistence with hash-based invalidation gives fast startup and automatic freshness with no manual step |
| **RAG Retrieval Cache** | Redundant OpenAI embedding calls + ChromaDB queries + cross-encoder inference for identical queries within a session — SHA-256 keyed, 200-entry ceiling, auto-invalidates on SLA file change | Caches the SLA chunks returned to the Recommend agent (input to the LLM), not the LLM response. Retrieval-level caching is safe because the retrieved chunks are stable within a session; the LLM still synthesises fresh recommendations from live SQLite data on every call |
| **Response Cache** | Re-running the full agent pipeline (ML inference → diagnosis → RAG → recommendations → emails) for an identical request when no input data has changed — keyed on message + orders file hash + sidecar mtimes + model name; 50-entry ceiling, full eviction | A single full pipeline run is expensive (multiple LLM calls, MCP tool invocations, SQLite reads). If the user re-submits the same query and no new prediction or diagnosis data has arrived (sidecar mtimes unchanged, same orders file), the cached tab outputs are returned instantly. Auto-invalidates when any upstream data changes — no manual flush needed |

> **Next version:** Add a **semantic cache** layer — instead of exact-match SHA-256 keying, cluster similar queries by embedding similarity (cosine > 0.95 threshold) and return the cached result for near-duplicate queries. This would further reduce embedding API calls when users ask semantically equivalent but lexically different questions across a session.

> Full design, sidecar file locations, cache key composition, eviction policy, and ChromaDB rebuild detection: **[`docs/16-caching-design.md`](docs/16-caching-design.md)**

---
---

## 15. Observability & Logging

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Structured logging uses Python's standard logging module with a named logger (supply_chain_delivery_app) and key=value pairs for machine-parseable output, written to log/ at runtime (gitignored). Coverage spans the full stack: the RAG pipeline logs every retrieval stage including cache hits, cross-encoder load, and duration; the ML prediction pipeline logs each of its 9 steps with row counts and metrics; MCP tools log invocations and SQLite reads; and all exceptions are caught, logged with full context (duration, input shape, error message), and re-raised. OpenAI Agents SDK tracing is disabled via OPENAI_AGENTS_DISABLE_TRACING=1 to suppress payload upload while preserving local logs, keeping sensitive order data off external servers.

| Surface | Location | Contents |
|---|---|---|
| Runtime log | `supply_chain_delivery_app/log/delivery_chat_run_{ts}.log` | Structured key=value events from all app modules; one file per process start |
| Prediction audit | `output/daily_delivery_delay_prediction_meta.json` | Summary stats, severity counts, top regions/weather/partners, formatted text |
| Diagnosis audit | `output/diagnosis_meta.json` | Truncated diagnosis summary for freshness detection |
| Simulation output | `output/simulate_delays_latest.csv` | Simulated orders with reassigned severity and `simulate_delay_reason` |
| Email output | `output/email_alerts.csv` | Delayed orders with `email_template_name` and `email_content` |
| Agent tracing | OpenAI platform | Disabled by default (`OPENAI_AGENTS_DISABLE_TRACING=1`) — 10 KB trace payload limit caused errors with large prediction outputs |

RAG retrieval emits 12 structured events with `duration_ms` on every call (e.g. `rag.retrieve.started`, `rag.retrieve.cache_hit`, `rag.retrieve.completed retrieved=N returned=N`).

> Full event reference, log format, audit sidecar schemas, output file inventory, and tracing config: **[`docs/17-observability-logging.md`](docs/17-observability-logging.md)**

---
---

## 16. Security Guardrails

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Security guardrails are implemented as a highest-priority system prompt layer, prepended to all other agent instructions in the order: Security Guardrails → Chatbot Behavior → Expert Instructions. This ensures security rules cannot be overridden by later layers.

Six guardrail categories are enforced: Scope Restriction limits the agent strictly to the five delivery operations capabilities, refusing all off-domain queries. Prompt Injection Defence blocks adversarial attempts to redefine the agent's role or embed instructions in uploaded data. Data Privacy prevents leakage of system internals, raw file paths, or bulk PII — only summaries and task-relevant excerpts are permitted. Tool Use Boundaries restrict the agent to its registered tool set, blocking arbitrary HTTP calls or unlisted tool invocations. Output Safety prohibits fabricated data, executable code, or harmful content in any response. Escalation Handling shuts down adversarial probing with a single clear refusal and no further engagement.

The attack surface for prompt injection is narrow by design. Only the Master Expert agent receives raw user text — all sub-agents operate on structured inputs constructed by the Master, with typed Pydantic output schemas that prevent free-form manipulation.

```
User Input
    ↓
Master Expert  ← ONLY entry point for raw user text
    │            security_guardrails.md prepended here
    ↓ (structured tool call arguments)
Sub-agents (Predict / Diagnose / Simulate / Recommend / Email / Fallback)
    ↓ (typed Pydantic output)
Master Expert  ← assembles final MasterOutput
```

### Why Sub-Agents Do Not Have Guardrails Prepended

Structured agents are configured with `agent as a tool` using OpenAI Agent SDK.

| Concern | Sub-agents need guardrails? | Reason |
|---|---|---|
| Prompt injection from user | No | Sub-agents never receive raw user text — only structured inputs from the Master |
| Scope violation | No | Each sub-agent is narrowly scoped by its own prompt + `tool_choice="required"` pins it to exactly the tools provided |
| Arbitrary output | No | `output_type=PydanticModel` enforces a typed schema — the LLM cannot produce unstructured free-form output |
| Data fabrication | Partially | Sub-agent prompts instruct against it; Pydantic validation rejects structurally invalid output |

**Exception — tool output injection:** If MCP responses or CSV data contained injected instructions (e.g. an order record with "Ignore previous instructions and..."), a sub-agent could theoretically be influenced. However, since all sub-agent outputs are Pydantic-constrained, any injected text would at most corrupt a string field — it cannot change tool-calling behaviour.

**Conclusion:** Security guardrails live only on the Master Expert — the sole user-facing agent and the correct enforcement point for input validation and scope restriction.

> Attack surface analysis, guardrail design rationale, and Fallback Advisor coverage: **[`docs/15-security-guardrails-design.md`](docs/15-security-guardrails-design.md)**

---
---

## 17. Documentation Index

<sub>[↑ Back to TOC](#table-of-contents)</sub>

| # | Doc | Contents |
|---|---|---|
| 01 | [`docs/01-architecture.md`](docs/01-architecture.md) | Layered architecture diagram, agent inventory, links to all workflow docs |
| 02 | [`docs/02-tech-stack.md`](docs/02-tech-stack.md) | Full technology stack with library details and design comments |
| 03 | [`docs/03-prompt-evolution-log.md`](docs/03-prompt-evolution-log.md) | Prompt engineering iteration history across all 7 agents |
| 04 | [`docs/04-ml-pipeline-architecture.md`](docs/04-ml-pipeline-architecture.md) | ML training pipeline flow — data extract → clean → EDA → feature engineering → model selection → persistence |
| 05 | [`docs/05-sqlite-database-design.md`](docs/05-sqlite-database-design.md) | SQLite schema: 27 tables, summary table structure, high-risk pattern classification, write lifecycle |
| 06 | [`docs/06-mcp-server-design.md`](docs/06-mcp-server-design.md) | FastMCP server: 3 tool contracts, stdio transport design, connection setup, key design decisions |
| 07 | [`docs/07-vectorstore-rag-design.md`](docs/07-vectorstore-rag-design.md) | ChromaDB store, three-stage retrieval (cosine → hybrid → cross-encoder), chunking pipeline, retrieval cache |
| 08 | [`docs/08-master-agent-flowchart.md`](docs/08-master-agent-flowchart.md) | Master & Fallback agent workflows: plan confirmation gate, sequential tool execution, post-processing; Fallback Advisor trigger, WebSearchTool, design rationale |
| 09 | [`docs/09-agent-predict-workflow.md`](docs/09-agent-predict-workflow.md) | Predict agent flow diagram + all design fixes (sidecar JSON, schema slimming, Pydantic enforcement) |
| 10 | [`docs/10-agent-diagnose-workflow.md`](docs/10-agent-diagnose-workflow.md) | Diagnose agent flow diagram + freshness check design |
| 11 | [`docs/11-agent-simulate-workflow.md`](docs/11-agent-simulate-workflow.md) | Simulation agent flow diagram + severity lookup logic + design decisions |
| 12 | [`docs/12-agent-recommend-workflow.md`](docs/12-agent-recommend-workflow.md) | Recommend agent + three-stage RAG pipeline (ChromaDB → hybrid → cross-encoder) |
| 13 | [`docs/13-agent-email-workflow.md`](docs/13-agent-email-workflow.md) | Email agent flow + severity templates + Pydantic model |
| 14 | [`docs/14-gradio-ui-design.md`](docs/14-gradio-ui-design.md) | ChatInterface vs manual Chatbot — design rationale for 15-tuple yield approach |
| 15 | [`docs/15-security-guardrails-design.md`](docs/15-security-guardrails-design.md) | Attack surface analysis + why sub-agents don't need guardrails prepended |
| 16 | [`docs/16-caching-design.md`](docs/16-caching-design.md) | Sidecar freshness detection (1h TTL), RAG retrieval cache (SHA-256 key, 200-entry eviction), ChromaDB rebuild detection |
| 17 | [`docs/17-observability-logging.md`](docs/17-observability-logging.md) | Runtime logger, structured RAG events, audit sidecar files, output file inventory, agent tracing config |
| 18 | [`docs/18-end-to-end-data-flow.md`](docs/18-end-to-end-data-flow.md) | End-to-end data flow from CSV input through ML inference, SQLite, agent orchestration to UI output |
| 19 | [`docs/19-format-agent-design.md`](docs/19-format-agent-design.md) | Format Summary agent: Markdown rendering, 15-tuple yield pattern, structured output composition |
| 20 | [`docs/20-executive-flow-diagrams.md`](docs/20-executive-flow-diagrams.md) | Executive-level flow diagrams for presentation and documentation |
| 21 | [`docs/21-eval-flow-design.md`](docs/21-eval-flow-design.md) | Eval framework: 5-agent LLM-as-judge design, RAGAS integration, DB isolation, run commands |

---
---

## 18. Setup

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### 1. Clone and install dependencies

```bash
# Install uv if not already installed
pip install uv

# From the project root
cd 0_supply_chain_capstone
uv sync                  # creates .venv and installs all pinned dependencies
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Required `.env` variables:

**OpenAI (Required)**

| Variable | Example value | Description |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5.4` | Primary model for reasoning, structured output, and narration |
| `OPENAI_MODEL_MINI` | `gpt-4.1-mini` | Lightweight model for formatting and summary rendering |

**LLM Backend**

| Variable | Example value | Description |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` (default) or `lmstudio` for local model |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio base URL — only required when `LLM_BACKEND=lmstudio` |
| `LLM_API_KEY` | `lm-studio` | LM Studio API key — only required when `LLM_BACKEND=lmstudio` |

**RAG Cross-Encoder Reranker**

| Variable | Example value | Description |
|---|---|---|
| `SC_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for Stage 3 RAG reranking (downloaded once by `sentence-transformers`) |

**Pipeline Paths (relative to project root)**

| Variable | Example value | Description |
|---|---|---|
| `SC_PREDICTION_MODEL_DIR` | `prediction_pipeline/models` | Directory containing trained `.pkl` model files |
| `SC_PREDICTION_DB_PATH` | `prediction_pipeline/db/delivery_predictions.db` | Path to SQLite database (27 tables) |
| `SC_PREDICTION_SRC_DIR` | `prediction_pipeline` | Root of the prediction pipeline module |
| `SC_DELIVERY_OUTPUT_DIR` | `supply_chain_delivery_app/output` | Output directory for predictions, simulations, emails, sidecars |

**Agent Behaviour Tuning**

| Variable | Example value | Description |
|---|---|---|
| `SC_MCP_ENRICH_ROWS` | `50` | Max rows sent to the LLM for per-row `llm_insights` enrichment during prediction |
| `SC_MCP_DISPLAY_ROWS` | `50` | Max rows shown in the Gradio prediction table |

**OpenAI Agents Tracing**

| Variable | Example value | Description |
|---|---|---|
| `OPENAI_AGENTS_DISABLE_TRACING` | `1` | Set to `1` to disable SDK tracing (recommended — 10 KB trace payload limit causes errors with large prediction outputs) |

### 3. Train the ML models (first run only)

Open and run `prediction_pipeline/notebooks/train_predict_delay_model.ipynb` end-to-end. This produces:
- `prediction_pipeline/models/best_classification_random_forest.pkl` (Stage 1)
- `prediction_pipeline/models/best_severity_random_forest.pkl` (Stage 2)
- `prediction_pipeline/db/delivery_predictions.db` (SQLite with 27 tables)

### 4. Start the MCP prediction server

```bash
# In a separate terminal, from the project root
uv run python prediction_pipeline/prediction_server.py
```

This starts the FastMCP server (stdio transport) that exposes `predict`, `diagnose`, and `simulate` as tools to the agent layer.

### 5. Launch the app

```bash
uv run python supply_chain_delivery_app/delivery_chat_app.py
```

Open `http://localhost:7860` in your browser.

---
---

## 19. Testing

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### Unit & Smoke Tests (Pytest)

Three test modules in `tests/` cover the core logic layers. Run offline — no OpenAI API call, no trained models required.

| Test file | Tests | What it covers |
|---|---|---|
| `tests/test_mcp_server.py` | 11 | Calls `predict_delivery_delays`, `get_delay_diagnosis`, and `simulate_order_delays` directly as async Python functions. Checks response structure, key presence, and content validity (e.g. "Rows affected" in simulate output). |
| `tests/test_pydantic_models.py` | 11 | Validates Pydantic output models for all agents (RowEnrichment, DeliveryDelayPredictionResult, DelayDiagnosisResult, SimulationsList, etc.). Confirms valid data accepted, invalid data rejected at schema boundary. |
| `tests/test_rag_knowledge.py` | 7 | Checks SLA file and vectorstore directory exist, ChromaDB collection is non-empty, and keyword queries return relevant SLA chunks. |

```bash
# Run all smoke/unit tests from project root
uv run pytest tests/ -v

# A Markdown report is written automatically to tests/reports/smoke_test_report.md
```

Latest report: **[`tests/reports/smoke_test_report.md`](tests/reports/smoke_test_report.md)**

### Daily Prediction Pipeline Test (without Production Data)

```bash
# Generate 3 synthetic daily CSVs (5 000 rows each)
uv run python -m prediction_pipeline.src.generate_daily_test_data_11

# Run the daily prediction pipeline against one of the test files
uv run python -m prediction_pipeline.src.daily_predict --file prediction_pipeline/data/raw/daily_delivery_logistics_1.csv
```

---
---

## 20. Evaluation

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### ML Model Evaluation

Evaluation was performed in `prediction_pipeline/notebooks/train_predict_delay_model.ipynb` at training time.

**Stage 1 — Binary Delay Classification**

8 classifiers evaluated (Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, SVM, Naive Bayes) using StratifiedKFold cross-validation (3 folds), `scoring='recall'`. Random Forest selected as production model.

| Metric | Score |
|---|---|
| Test Accuracy | **89.6%** |
| Recall (delayed orders) | **81.5%** |
| ROC-AUC | **96.7%** |

GridSearchCV optimal hyperparameters:

| Hyperparameter | Search space | Optimal |
|---|---|---|
| `n_estimators` | 150, 200, 300 | 200 |
| `max_depth` | None, 10, 20 | None (fully grown) |
| `min_samples_split` | 2, 5 | 2 |
| `class_weight` | balanced, None | balanced |

**Stage 2 — Severity Classification (Short / Medium / Long)**

Two approaches evaluated on the Stage 1 predicted-delayed subset: Option 1 (Random Forest severity classifier) vs Option 2 (Ordinal Regression — Frank & Hall cumulative threshold). Option 1 selected.

| Metric | Score |
|---|---|
| Test Accuracy | **63.7%** |
| Weighted F1 | **65.6%** |

The lower accuracy relative to Stage 1 reflects genuine class imbalance in the severity distribution (50.6% Short / 39.7% Medium / 11.9% Long) — the Long class is the hardest to predict with only ~12% representation. Severity prediction is harder than binary delay detection by design; partial triage (correctly identifying ~64% of severity levels) still provides significant operational value for resource prioritisation.

**Top 5 Feature Importances (Stage 1 Random Forest)**

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `km_per_expected_hr` | 27.1% | Schedule tightness relative to distance — overly optimistic windows are the single strongest delay driver |
| 2 | `mode_urgency` | 21.5% | Delivery mode risk score — Same-Day and Express modes carry the highest baseline delay risk |
| 3 | `schedule_risk` | 14.9% | Combined weather × urgency — bad weather plus tight deadline compounds to near-certain delay |
| 4 | `vehicle_load_strain` | — | Weight-distance load relative to vehicle capacity — overloaded vehicles miss time windows |
| 5 | `carrier_avg_schedule` | — | Partner-level pattern — some partners systematically accept routes too tight for their fleet |

Top 3 features together account for **>63% of model decisions**.

> Full confusion matrices, per-class metrics, and end-to-end two-stage evaluation: `prediction_pipeline/notebooks/train_predict_delay_model.ipynb`

---

### Agent Evaluation (LLM-as-Judge)

All five agents were evaluated using a structured LLM-as-judge framework (GPT-5.4 as evaluator) scoring each agent output on three dimensions: **Relevance**, **Faithfulness**, and **Safety** on a 1–5 scale. Pass threshold: mean ≥ 3.0.

| Agent | Relevance | Faithfulness | Safety | Mean | Result |
|---|---|---|---|---|---|
| Predict Delivery Delays | 5.0/5 | 5.0/5 | 5.0/5 | **5.00** | PASS |
| Diagnose Delay Patterns | 5.0/5 | 5.0/5 | 5.0/5 | **5.00** | PASS |
| Recommendation Expert Agent | 5.0/5 | 5.0/5 | 5.0/5 | **5.00** | PASS |
| Simulate Delay Prediction | 5.0/5 | 4.0/5 | 5.0/5 | **4.67** | PASS |
| Email Alert Agent | 5.0/5 | 5.0/5 | 5.0/5 | **5.00** | PASS |

**5/5 agents passed. All scores above threshold.**

```bash
# Run the full eval suite (all 5 agents + RAGAS)
uv run python evals/run_evals.py

# Run a single agent eval
uv run python evals/run_evals.py --agent recommend

# Report is written to evals/reports/eval_report_<timestamp>.md
```

Latest reports: **[`evals/reports/`](evals/reports/)**

> Full eval framework design, DB isolation, fixture lifecycle, and per-agent rubric: **[`docs/21-eval-flow-design.md`](docs/21-eval-flow-design.md)**

---

### RAG Evaluation (RAGAS)

The three-stage RAG pipeline (`retrieve_sla_context()`) is evaluated using RAGAS as part of the standard eval suite. The evaluation scopes faithfulness to SLA grounding: can the `sla_reference` citations in recommendations be traced back to the retrieved SLA context block?

| Metric | Score | Threshold | Result |
|---|---|---|---|
| Faithfulness | **0.938** | ≥ 0.60 | PASS |
| Answer Relevancy | **0.778** | ≥ 0.60 | PASS |

RAGAS runs on the same eval DB as the agent evals (`evals/db/delivery_predictions_eval.db`) — no separate large-scale dataset is needed. The recommendation agent's input (prediction + diagnosis outputs) is already in the eval DB after the standard pipeline run.

> RAGAS eval implementation: **[`evals/test_eval_rag.py`](evals/test_eval_rag.py)**

---

### Human Baseline Calibration

Agent outputs were independently scored by a human reviewer on the same three dimensions (Relevance, Faithfulness, Safety, 1–5 scale) and compared against the LLM judge scores.

| Agent | LLM Mean | Human Mean | Notes |
|-------|----------|------------|-------|
| Predict Delivery Delays | 5.00 | 5.00 | Aligned |
| Diagnose Delay Patterns | 5.00 | 5.00 | Aligned |
| Recommendation Expert Agent | 5.00 | 5.00 | Aligned |
| Simulate Delay Prediction | 4.67 | 3.67 | Human scored lower — simulation quality has room to improve |
| Email Alert Agent | 5.00 | 5.00 | Aligned |

**Reviewer note on model choice:** GPT-4.1-mini resulted in output of lower quality with human relevance scores of 2–3; GPT-5.4 produced better quality with scores of 5. This confirms GPT-5.4 as the appropriate model for this complexity.

Human scores: **[`evals/human_baseline/human_scores.xls`](evals/human_baseline/human_scores.xls)**  
Baseline report: **[`evals/reports/`](evals/reports/)**

```bash
uv run pytest evals/test_eval_human_baseline.py -v
# Report written to evals/reports/human_baseline_report_<timestamp>.md
```

---
---

## 21. Strategic Deductions and Business Impact

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Last-mile delivery failures are not random — they cluster by region, partner, weather condition, and vehicle type. By predicting delays before dispatch and diagnosing which operational dimensions are deteriorating, the system gives logistics managers actionable signals hours ahead of customer impact rather than reactive explanations after the fact. The shift from post-hoc reporting to pre-dispatch intelligence is the core business value: interventions become possible when they are still cheap.

The recommendation engine grounds every action in the SLA contract the business has already signed. Rather than generating generic best-practice advice, it cites the specific clause, target, and penalty relevant to each failing dimension — turning a compliance document that typically sits unread into an operational decision support tool. This closes the gap between what the SLA requires and what the operations team is doing, and makes the commercial cost of each delay pattern explicit and traceable.

Simulation adds a dimension that static reporting cannot: it lets planners test the impact of proposed changes — rerouting high-risk partners, switching delivery modes for severe-weather regions, redistributing vehicle types — before committing resources. The ability to model "what if we moved all high-risk Express orders in the North region to Ground during adverse weather?" reduces the guesswork in operational planning and makes scenario analysis accessible to non-technical users through a conversational interface.

The personalised email alert pipeline converts delay predictions into customer communications automatically, replacing a manual triage process that typically introduces a 2–4 hour lag between a detected delay and customer notification. Earlier notification reduces inbound support volume and protects NPS scores in the window where customer expectations can still be managed.

Across the full pipeline, the system operates on a 1-hour freshness TTL: a logistics team can run a morning prediction, diagnose overnight pattern shifts, generate SLA-grounded recommendations, and send customer alerts in a single end-to-end session — a workflow that previously required separate tools, separate teams, and multiple manual handoffs.

---
---

## 22. Key Learnings

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### ML & Feature Engineering

**Feature engineering outweighed model selection by a wide margin.** The two engineered features `km_per_expected_hr` (27.1% importance) and `schedule_risk` (14.9%) together contribute more predictive signal than the choice between Random Forest, XGBoost, and LightGBM — swapping models moved recall by 2–3 percentage points; engineering these features moved it by ~15 points. The implication: in structured tabular ML problems, time spent on domain-informed feature design has higher ROI than hyperparameter tuning or model selection.

**Classical ML outperforms LLMs for classification on structured, correlated, multi-feature tabular data.** Delay prediction involves an 85–120 column feature matrix (post one-hot encoding) with significant inter-feature correlation — `weather_severity` and `schedule_risk` share weather as a component; `km_per_expected_hr` and `vehicle_load_strain` both capture route complexity. LLMs have no mechanism to handle correlated feature matrices or large numeric feature sets — they lack the mathematical operations (Gini impurity, information gain splits, class-weight balancing) that make Random Forest effective here. LLMs also cannot produce calibrated probability scores or feature importance rankings. The 89.6% accuracy and 96.7% ROC-AUC achieved by the Random Forest would not be reachable with an LLM-based classifier on the same data; the LLM's role in the prediction pipeline is narrating the model's output, not replacing the model.

**Class imbalance in two-stage pipelines requires independent calibration per stage.** A single 4-class model (on-time / short / medium / long) is dominated by the 73% on-time majority and cannot be effectively calibrated for severity. Separating binary detection (Stage 1, optimised for recall via `class_weight='balanced'`) from severity estimation (Stage 2, balanced 3-class) allowed each stage to be independently tuned — and made the recall vs precision trade-off an explicit, controllable design decision rather than an artefact of joint training.

### Agentic System Design

**Multi-agent systems fail silently without explicit prerequisite contracts in the orchestrator prompt.** Early versions of the Master Orchestrator called `get_delay_diagnosis` before `predict_delivery_delays` had populated the `daily_*` SQLite tables — the tool returned empty data with no error, producing a diagnosis with no patterns. The fix was embedding dependency rules directly into the Master prompt (`diagnose requires predict to be fresh`) and using freshness sidecar files as machine-enforceable gates. Lesson: agent tool dependencies must be first-class prompt constraints, not assumed from documentation.

**LLM context window capacity must be tested with the full production prompt stack, not individual agent prompts.** GPT-4.1-mini handled each individual agent prompt in isolation but silently dropped instructions when the full layered stack (security_guardrails + chatbot_behavior + agent-specific prompt) was assembled by `build_instruction()`. The failure was non-obvious — no error, just missing `llm_insights` fields for some rows, and sometimes the entire pipeline halting mid-run. Testing with trimmed prompts masked the real failure mode. Switching to GPT-5.4 resolved all context-capacity issues.

**Pydantic v2 output schemas are the most reliable guardrail against agent hallucination in structured pipelines.** Before enforcing typed schemas, the Email agent occasionally invented extra fields not in the template, and the Predict agent returned variable numbers of rows. After adding `output_type=PydanticModel` to every agent, schema violations surface immediately as validation errors rather than silently corrupted downstream output. Structured output contracts are more reliable than prompt-level instructions alone for ensuring schema compliance.

**Deterministic Python tools eliminate hallucination at the schema boundary.** The Email agent originally used the LLM to generate email content directly — it invented template variations, added fields not in any defined template, and produced inconsistent severity mappings across runs. Moving email generation to a deterministic Python tool (`email_customers.py`) with pre-defined severity templates eliminated all hallucination in email output: the LLM's role was reduced to selecting the severity tier, and Python handled all content production. The same principle applies to `recommend_actions` — the tool fetches structured SQLite statistics deterministically before passing them to the LLM, ensuring the LLM only narrates verified numbers, never invents them. Where the output must be structurally reliable, delegate generation to code; reserve the LLM for reasoning and narration over that output.

**Layered modular prompts reduce duplication and make cross-cutting changes a single-file update.** An early architecture embedded security constraints and formatting rules in every agent's prompt — 7 copies of overlapping content that diverged over time. Refactoring into three shared layers (`security_guardrails.md`, `chatbot_behavior.md`, `format_summary.md`) assembled programmatically by `build_instruction()` means a security policy change is made in one file and propagates to all agents automatically. Individual agents carry only their domain-specific instructions. The Fallback Advisor gets a different composition (security + its own prompt, no behaviour layer) without requiring a new shared file. The modular approach also made prompt debugging faster — when formatting drifted, the cause was always locatable in `format_summary.md` rather than scattered across seven files.

**Multi-agent coordination must mirror the data dependency graph of the underlying pipeline.** The five-agent chain (Predict → Diagnose → Simulate → Recommend → Email) is not arbitrary — each step consumes output produced by a prior step. Diagnose reads `daily_*` SQLite tables written by Predict; Recommend queries the same tables and uses Diagnose patterns to ground SLA citations; Email fetches from the same prediction store. Attempting parallel execution would produce empty or stale data at every downstream step. The Master Orchestrator enforces this sequence explicitly via tool dependency rules in its prompt, freshness sidecar checks as machine-enforceable gates, and the plan confirmation gate — which shows the user a numbered action plan before executing any tool chain, giving a human checkpoint before committing to an expensive sequential run. The agent-as-tool pattern (sub-agents registered as tools on the Master rather than SDK handoffs) was critical here: the Master retains control throughout, assembles all sub-agent Pydantic outputs into a single `MasterOutput`, and can enforce ordering that SDK-level handoffs would not guarantee.

**Gradio session state is not reliable for cross-turn freshness.** `gr.State` resets on browser reconnect, meaning freshness decisions based on in-memory state would force a full pipeline rerun after a page refresh. Switching to file-based sidecar mtime checks made freshness persistent across reconnects, browser refreshes, and long sessions — and gave the sidecars a second role as audit records. The lesson generalises: for stateful AI apps with expensive tool calls, persist state to durable storage rather than session memory.

**MCP process isolation made the ML pipeline significantly easier to debug.** When feature alignment bugs (wrong column order, missing engineered features) occurred during daily inference, the error surfaced as an explicit MCP tool failure — not a silent corruption that propagated through Pydantic models into the UI. The clean process boundary meant ML bugs and agent bugs were always distinguishable and never conflated. For complex AI systems combining ML inference with LLM orchestration, process-level isolation between the two is worth the additional transport overhead.

### RAG Pipeline

**Chunk boundary quality determines retrieval quality more than embedding model choice.** Initial chunking with `RecursiveCharacterTextSplitter` alone produced mid-clause breaks — e.g., a chunk ending at "SLA penalty for Express mode: ₹500 per order for delays exceeding..." before the threshold was stated, making the chunk useless for the cross-encoder. Switching to `MarkdownHeaderTextSplitter` first (preserving section boundaries) then `RecursiveCharacterTextSplitter` (for size control), with header breadcrumbs prepended to each chunk, eliminated boundary-split retrievals and dramatically improved cross-encoder reranking quality.

---
---
## 23. Potential Future Extensions

<sub>[↑ Back to TOC](#table-of-contents)</sub>

- Real-time streaming integration: Replace batch CSV ingestion with a Kafka or database trigger to enable continuous inference on live order streams.
- Model monitoring and retraining: Add data drift detection (e.g., using Evidently) and automated retraining triggers when feature distributions shift significantly.
- Simulations: using monti-carlo algorithm for simulations and strengthening the impact assessment.
- Multi-modal inputs: Incorporate traffic API data, satellite weather feeds, and partner GPS telemetry as additional real-time features.
- Fine-tuned SLA-specific LLM or Model-Routing: Replace the general-purpose GPT-5.4 / 4.1-mini with a fine-tuned model trained on historical logistics decisions to improve recommendation specificity. Use model routing to identify the intent and use smaller models for simple tasks and larger models for complex tasks.
- Scemantic cache: Implement semantic cache to avoid LLM calls when similar queries are fired.
- UI: Use production enterprise-grade UI tools/ libraries
- Performance tuning: to reduce the time taken for each agent execution and rendering the results on the screen
- A/B testing framework: Instrument the recommendation engine to track which suggestions are actually implemented and measure their impact on delay rates.
- Partner portal: Expose a sub-set of insights to delivery partners through a read-only API so they can see their own performance data and proactively adjust operations.

---
---

## 24. Gradio UI App Screenshots

<sub>[↑ Back to TOC](#table-of-contents)</sub>

### Chat Interface & Quick Actions
![Chat Interface](assets/screenshots/ui_01_chat_interface.png)

### Predict Tab — Delay Prediction + LLM Insights
![Prediction Output](assets/screenshots/predict_01_chat_output.png)
![Prediction Table](assets/screenshots/predict_02_output_table.png)
![Prediction Table with LLM Insights per Record](assets/screenshots/predict_03_output_table.png)
![Prediction Table with LLM Insights per Record](assets/screenshots/predict_04_output_table.png)

### Diagnose Tab — Daily vs Historical Pattern Analysis
![Diagnosis Output](assets/screenshots/diagnose_01_output.png)
![Diagnosis Output](assets/screenshots/diagnose_02_output.png)
![Diagnosis Output](assets/screenshots/diagnose_03_output.png)
![Diagnosis Output](assets/screenshots/diagnose_04_output.png)
![Diagnosis Table](assets/screenshots/diagnose_05_data_table.png)

### Simulate Tab — What-If Scenario Analysis
![Simulation Output](assets/screenshots/simulate_01_output.png)
![Simulation Table](assets/screenshots/simulate_02_output.png)
![Simulation Table](assets/screenshots/simulate_03_output.png)
![Simulation Table](assets/screenshots/simulate_04_csv_download.png)

### Recommend Tab — SLA-Grounded Recommendations
![Recommendation Output](assets/screenshots/recommend_01_output.png)
![Recommendation Detail](assets/screenshots/recommend_02_output.png)
![Recommendation Detail](assets/screenshots/recommend_03_output.png)
![Recommendation Detail](assets/screenshots/recommend_04_output.png)
![Recommendation Detail](assets/screenshots/recommend_05_output.png)

### Email Tab — Customer Alert Generation
![Email Output](assets/screenshots/email_01_output.png)
![Email Sample](assets/screenshots/email_02_sample_email.png)
![Email Sample](assets/screenshots/email_03_csv_output.png)

---

## 24. Licence

<sub>[↑ Back to TOC](#table-of-contents)</sub>

Restricted — evaluation use only by upGrad, IIIT-B, and Liverpool John Moores University (LJMU). See [`LICENSE`](LICENSE) for full terms.

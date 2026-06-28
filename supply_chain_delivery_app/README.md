# Supply Chain Last-Mile Delivery App

OpenAI Agents SDK multi-agent system for order delay prediction, root-cause
diagnosis, simulation, recommendations, and customer email alerts. A Gradio
UI orchestrates the agents and streams results in real time.

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  PROMPT ASSEMBLY  (config/load_config.py — build_instruction())      ║
║                                                                      ║
║  shared/security_guardrails.md  ← prepended first (highest priority) ║
║           +                                                          ║
║  shared/chatbot_behavior.md     ← query routing, plan confirmation   ║
║           +                                                          ║
║  agents/<agent>.md              ← domain reasoning per agent         ║
║           │                                                          ║
║           ▼  assembled instruction string                            ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  AGENT RUNTIME  (delivery_agents.py — OpenAI Agents SDK)             ║
║                                                                      ║
║  Master Expert Agent  (orchestrator)                                 ║
║  ├── Predict Agent      → output: PredictOutput (Pydantic)           ║
║  ├── Diagnose Agent     → output: DelayDiagnosisResult (Pydantic)    ║
║  ├── Simulate Agent     → output: SimulationOutput (Pydantic)        ║
║  ├── Recommend Agent    → output: RecommendationOutput (Pydantic)    ║
║  ├── Email Alert Agent  → output: EmailsList (Pydantic)              ║
║  ├── Format Summary Agent (agent-as-tool, shared/format_summary.md)  ║
║  └── Fallback Advisor   → WebSearchTool                              ║
║                                                                      ║
║  All outputs validated via Pydantic v2 structured output contracts   ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  TOOLS LAYER                                                         ║
║                                                                      ║
║  ┌───────────────────┐ ┌──────────────────────┐ ┌─────────────────┐  ║
║  │ FastMCP Client    │ │ RAG Tool             │ │ Email Tool      │  ║
║  │ (MCP protocol,    │ │ rag_knowledge.py     │ │ email_customers │  ║
║  │  stdio transport) │ │ • Cross-enc rerank   │ │ • Severity-     │  ║
║  │                   │ │ • Hybrid retrieval   │ │   based Python  │  ║
║  │ → predict         │ │   (cosine + BM25)    │ │   templates     │  ║
║  │ → diagnose        │ │ • Hash-based cache   │ │ • Writes to CSV │  ║
║  │ → simulate        │ │   invalidation       │ │                 │  ║
║  └─────────┬─────────┘ └──────────┬───────────┘ └─────────────────┘  ║
║            │                      │                                  ║
║            │            recommend_actions.py                         ║
║            │            (SQLite reads + RAG retrieval)               ║
╚════════════╪══════════════════════╪══════════════════════════════════╝
             │                      │
╔════════════╪══════════════════════╪══════════════════════════════════╗
║  KNOWLEDGE & PERSISTENCE          │                                  ║
║            │                      │                                  ║
║   prediction_server.py     ChromaDB (vectorstore/)                   ║
║   SQLite DB (27 tables)    SLA doc → text-embedding-3-small          ║
║   output/ CSVs             (1536-dim, MarkdownHeader chunking)       ║
╚══════════════════════════════════════════════════════════════════════╝
                            │
╔══════════════════════════════════════════════════════════════════════╗
║  POST-PROCESSING & UI  (helpers/ + delivery_chat_app.py)             ║
║                                                                      ║
║  post_processing.py  → formats agent outputs for Gradio tabs         ║
║  logging_utils.py    → writes timestamped logs to log/               ║
║  app_utils.py        → state management, tab routing                 ║
║                                                                      ║
║  Gradio UI  →  5 tabs + 6 quick-action buttons  →  user              ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Project Structure

```
supply_chain_delivery_app/
├── delivery_chat_app.py           # Gradio UI — entry point
├── delivery_agents.py             # Pydantic output models + agent definitions
├── config/
│   ├── load_config.py             # build_instruction() — assembles layered prompts
│   └── prompts/
│       ├── agents/                # 7 agent-specific markdown prompts
│       │   ├── master_expert.md
│       │   ├── predict_delivery_delays.md
│       │   ├── diagnose_delay_patterns.md
│       │   ├── delay_simulation.md
│       │   ├── recommendation.md
│       │   ├── email_alert.md
│       │   └── fallback_advisor.md
│       └── shared/                # 3 cross-cutting prompts (prepended to agent prompts)
│           ├── security_guardrails.md
│           ├── chatbot_behavior.md
│           └── format_summary.md
├── tools/
│   ├── rag_knowledge.py           # ChromaDB + hybrid retrieval (cosine + keyword)
│   ├── recommend_actions.py       # SQLite reads + RAG retrieval for recommendations
│   └── email_customers.py         # Severity-based email template generation
├── helpers/
│   ├── app_utils.py               # Gradio UI helpers and state management
│   ├── post_processing.py         # Agent output post-processing
│   └── logging_utils.py           # Runtime logging setup
├── knowledge/
│   └── delivery_sla_github_ready.md  # 36-section SLA/OLA policy document (RAG source)
├── vectorstore/                   # ChromaDB persistent store (gitignored)
├── input/                         # Daily order CSVs for prediction
├── output/                        # Generated prediction, simulation, and email CSVs (gitignored)
├── log/                           # Runtime application logs
└── notebooks/
    ├── main.ipynb
    ├── sc_delivery_agents.ipynb
    └── logic-workflow-fixes.ipynb
```

## Prerequisites

- Python 3.11+
- The **prediction_pipeline** module (sibling folder) must have trained models and
  the SQLite database in place — the delivery app calls the FastMCP prediction
  server at runtime.
- Project-root `.env` (see `.env.example` at the project root) with:

| Variable | Purpose | Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_MODEL` | Primary LLM | `gpt-5.4` |
| `OPENAI_MODEL_MINI` | Lighter model for formatting | `gpt-4.1-mini` |
| `SC_PREDICTION_MODEL_DIR` | Path to trained models | `prediction_pipeline/models` |
| `SC_PREDICTION_DB_PATH` | Path to SQLite DB | `prediction_pipeline/db/delivery_predictions.db` |
| `SC_PREDICTION_SRC_DIR` | Path to prediction_pipeline root | `prediction_pipeline` |
| `SC_DELIVERY_OUTPUT_DIR` | Path for generated outputs | `supply_chain_delivery_app/output` |
| `SC_MCP_ENRICH_ROWS` | Max rows for LLM enrichment | `50` |
| `SC_MCP_DISPLAY_ROWS` | Max rows to display in UI | `50` |

## Setup

```bash
# From the project root
uv sync                  # installs all dependencies from pyproject.toml lockfile

# Verify .env is configured
cat .env | grep SC_
```

## Running the App

1. Start the MCP prediction server in a separate terminal:

```bash
# From the project root
python prediction_pipeline/prediction_server.py
```

2. Launch the Gradio app:

```bash
cd supply_chain_delivery_app
python delivery_chat_app.py
```

Open `http://localhost:7860`.

The app has 5 tabs:
- **Predict** — Upload a daily CSV or use the default; runs the two-stage ML pipeline and enriches delayed orders with per-row LLM insights.
- **Diagnosis** — Root-cause analysis comparing today's delay patterns against historical summaries across 12 dimensions.
- **Simulation** — What-if delay scenarios (e.g. *"what if weather turns stormy in the East region?"*).
- **Recommendation** — SLA-grounded optimization actions in three categories (quick-win / short-term / long-term).
- **Email** — Severity-templated customer email alerts for all delayed orders.

Six quick-action buttons trigger common workflows without requiring a typed query.

## MCP Integration (Model Context Protocol)

The delivery app and the ML prediction pipeline communicate via **FastMCP** over stdio transport. This decouples the agent layer from the ML code — the agents call tools without needing to import or directly execute Python ML code.

**How it works:**

```
delivery_chat_app.py (MCP client)
        │
        │  stdio transport (subprocess)
        ▼
prediction_server.py (FastMCP server)
        │
        ├── predict   → DailyPredictionPipeline.run()
        ├── diagnose  → DatabaseOperations (reads 27 SQLite tables)
        └── simulate  → simulate_delays.py
```

The OpenAI Agents SDK registers the MCP server as a tool provider. When an agent calls `predict`, `diagnose`, or `simulate`, the SDK routes the call through the MCP protocol to `prediction_server.py`, which executes the ML code and returns structured results.

**Why MCP?**
- The ML pipeline and the agent app can run in separate processes (or even separate machines).
- The prediction server can be replaced or upgraded without changing agent code.
- Tools are discoverable at runtime — the agent SDK queries the server for available tools at startup.

**Starting the server:**

```bash
# From the project root — must be running before launching the app
python prediction_pipeline/prediction_server.py
```

The server runs persistently over stdio; the delivery app spawns it as a subprocess and communicates over stdin/stdout.

## Agent Configuration (Markdown Prompts)

Each agent's instructions live in a dedicated `.md` file under
`config/prompts/agents/`. `load_config.build_instruction()` reads the file,
parses `## Role`, `## Goal`, `## Backstory`, `## Task`, and `## Expected Output`
sections, and prepends the shared security and behaviour prompts before passing
the assembled instruction to the OpenAI Agents SDK `Agent()`.

**Prompt layering order** (highest priority first):

```
shared/security_guardrails.md   ← security constraints, scope restriction
shared/chatbot_behavior.md      ← query routing, action plan confirmation
agents/<agent_name>.md          ← domain reasoning for this agent
```

`shared/format_summary.md` is used by the dedicated Format Summary agent
(called as a sub-agent tool) to separate rendering logic from reasoning logic.

Edit any `.md` file and restart the app to change agent behaviour — no Python
changes required.

## Agents

| Agent | Prompt file | Tool | Purpose |
|---|---|---|---|
| Master Expert | `master_expert.md` | all sub-agents | Orchestration, freshness detection, sequential tool execution |
| Predict | `predict_delivery_delays.md` | MCP `predict` | Two-stage RF prediction + per-row LLM enrichment (`llm_insights`) |
| Diagnose | `diagnose_delay_patterns.md` | MCP `diagnose` | Reads 24 DB summary tables; daily vs historical pattern analysis |
| Simulate | `delay_simulation.md` | MCP `simulate` | What-if scenario translation and row-level enrichment |
| Recommend | `recommendation.md` | `recommend_actions` (RAG) | SLA-grounded 3-category optimization recommendations |
| Email Alert | `email_alert.md` | `fetch_delayed_orders_for_email` | Severity-templated customer email generation (Python-deterministic) |
| Format Summary | `format_summary.md` | agent-as-tool | Structures final Markdown output per summary type |
| Fallback Advisor | `fallback_advisor.md` | WebSearchTool | Handles out-of-scope queries |

## RAG Knowledge Base

The `rag_knowledge.py` tool indexes `knowledge/delivery_sla_github_ready.md`
into ChromaDB using:
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Chunking**: `MarkdownHeaderTextSplitter` (semantic boundary detection) followed by `RecursiveCharacterTextSplitter` (200-character overlap)
- **Retrieval**: Hybrid — 70% cosine similarity + 30% BM25 keyword matching
- **Cache invalidation**: Hash-based — re-indexes automatically when the source document changes

The vector store persists in `vectorstore/` (gitignored) and is rebuilt on first run or when the SLA document changes.

## Pydantic Output Models

All agent outputs are validated against Pydantic v2 models defined in
`delivery_agents.py`. Key models:

| Model | Agent | Key fields |
|---|---|---|
| `PredictOutput` | Predict | `predict_summary`, `delayed_orders` (list with `llm_insights` per row) |
| `DelayDiagnosisResult` | Diagnose | `high_risk_patterns`, `comparison`, `diagnosis_summary` |
| `SimulationOutput` | Simulate | list of `SimulationRow` with `simulate_delay_reason` |
| `RecommendationOutput` | Recommend | list of `RecommendedAction` with `category`, `sla_reference`, `supporting_data` |
| `EmailsList` | Email | list of `EmailAlert` |
| `MasterOutput` | Master | aggregates all sub-agent outputs |

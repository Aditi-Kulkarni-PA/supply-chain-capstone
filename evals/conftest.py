"""
Eval conftest — env setup, path wiring, and shared session fixtures.

Env vars are set at module level (before any app import) so agents and the
prediction pipeline all pick up gpt-4.1-mini and the eval DB path.

DB layout:
    delivery_predictions_eval.db      — standard evals  (50-row slice)
    delivery_predictions_eval_ragas.db — RAGAS eval     (full 3 input CSVs)

The MCP server subprocess inherits the parent env at start-up, so setting
SC_PREDICTION_DB_PATH before entering `async with pipeline_mcp:` is enough.
"""

import os
import sys
from pathlib import Path

# ── 1. Resolve paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # 0_supply_chain_capstone
APP_DIR      = PROJECT_ROOT / "supply_chain_delivery_app"
PIPELINE_DIR = PROJECT_ROOT / "prediction_pipeline"
EVALS_DIR    = Path(__file__).resolve().parent

# Insertion order matters: last insert = index 0 = highest priority.
# APP_DIR must beat prediction_pipeline/config/ when delivery_agents does `from config import ...`
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(EVALS_DIR))   # makes `import judge`, `import eval_config` work in tests

# ── 2. DB paths ───────────────────────────────────────────────────────────────
# Eval DBs live in evals/db/ — they are eval infrastructure, not ML pipeline artifacts.
EVAL_DB  = EVALS_DIR / "db" / "delivery_predictions_eval.db"
RAGAS_DB = EVALS_DIR / "db" / "delivery_predictions_eval_ragas.db"
FULL_INPUT_DIR  = APP_DIR / "input"
# Full input file used for all agent evals — the same file the production pipeline runs on.
# We agreed: judge + RAGAS both test the full pipeline output, not a 50-row sample.
FULL_INPUT_FILE = FULL_INPUT_DIR / "daily_delivery_logistics_1.csv"

# ── 3. Override env vars BEFORE any app imports ───────────────────────────────
# load_dotenv in delivery_agents uses override=False, so these values survive.
# Use the same model as the UI (.env) so eval behaviour matches production.
# OPENAI_MODEL_MINI stays on gpt-4.1-mini to keep costs down for lightweight tasks.
os.environ["OPENAI_MODEL"]              = "gpt-5.4"
os.environ["OPENAI_MODEL_MINI"]         = "gpt-4.1-mini"
os.environ["SC_PREDICTION_DB_PATH"]     = str(EVAL_DB)
# Absolute path so _resolve_env in daily_predict.py doesn't mis-anchor the relative .env value
os.environ["SC_PREDICTION_MODEL_DIR"]   = str(PIPELINE_DIR / "models")
# Only process 3 delayed orders for email eval — enough to cover all severity types
os.environ["SC_EMAIL_MAX_ROWS"]         = "3"

# ── 4. App imports (after env setup) ─────────────────────────────────────────
import pytest
import pytest_asyncio
from agents import Runner
from delivery_agents import pipeline_mcp, predict_delivery_delays_agent

# MCP's stdio transport on macOS only inherits: HOME, LOGNAME, PATH, SHELL, TERM, USER.
# SC_PREDICTION_DB_PATH and SC_PREDICTION_MODEL_DIR are custom vars that won't reach the
# subprocess unless we inject them explicitly via StdioServerParameters.env.
# Setting .env merges on top of get_default_environment() — PATH etc. are still inherited.
pipeline_mcp.params.env = {
    "SC_PREDICTION_DB_PATH":   str(EVAL_DB),
    "SC_PREDICTION_MODEL_DIR": str(PIPELINE_DIR / "models"),
    "OPENAI_API_KEY":          os.environ.get("OPENAI_API_KEY", ""),
    "OPENAI_MODEL":            "gpt-5.4",
    "OPENAI_MODEL_MINI":       "gpt-4.1-mini",
}

REPORTS_DIR = EVALS_DIR / "reports"


# ── 5. Fixtures ───────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def pipeline_mcp_server():
    """Start the MCP stdio server once for the entire session.

    The subprocess inherits SC_PREDICTION_DB_PATH=EVAL_DB from the parent env,
    so all MCP tool calls (predict, diagnose, simulate) write to the eval DB.
    """
    async with pipeline_mcp:
        yield


def _copy_hist_tables(src_db: Path, dst_db: Path) -> None:
    """Copy all hist_* tables from the production DB into the eval DB.

    The hist_* tables are written during model training (not during daily prediction
    runs), so they never appear in a freshly seeded eval DB. Copying them from
    production gives the recommend/diagnose/simulate tools the historical baseline
    they need without contaminating production with eval writes.
    """
    import sqlite3
    PROD_DB = PIPELINE_DIR / "db" / "delivery_predictions.db"
    if not PROD_DB.exists():
        return
    conn_src = sqlite3.connect(str(PROD_DB))
    conn_dst = sqlite3.connect(str(dst_db))
    try:
        hist_tables = [
            r[0] for r in conn_src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'hist_%'"
            ).fetchall()
        ]
        for table in hist_tables:
            schema_row = conn_src.execute(
                f"SELECT sql FROM sqlite_master WHERE name='{table}'"
            ).fetchone()
            if not schema_row:
                continue
            rows = conn_src.execute(f"SELECT * FROM {table}").fetchall()
            ncols = len(conn_src.execute(f"SELECT * FROM {table} LIMIT 0").description)
            conn_dst.execute(f"DROP TABLE IF EXISTS {table}")
            conn_dst.execute(schema_row[0])
            conn_dst.executemany(
                f"INSERT INTO {table} VALUES ({','.join(['?']*ncols)})", rows
            )
        conn_dst.commit()
    finally:
        conn_src.close()
        conn_dst.close()


@pytest.fixture(scope="session")
def seeded_eval_db():
    """Populate eval DB with hist_* baseline + fresh daily predictions on the full input file.

    Step 1: copy hist_* tables from production (read-only historical baseline).
    Step 2: run DailyPredictionPipeline on the full input file (5 000 rows) so daily_*
            tables have realistic delay distribution for diagnose/simulate/recommend tools.

    Synchronous — no LLM call. Runs once per session.
    """
    from src.daily_predict import DailyPredictionPipeline
    EVAL_DB.parent.mkdir(parents=True, exist_ok=True)
    _copy_hist_tables(PIPELINE_DIR / "db" / "delivery_predictions.db", EVAL_DB)
    DailyPredictionPipeline.get_prediction(str(FULL_INPUT_FILE), "")
    return EVAL_DB


@pytest.fixture(scope="session")
def ragas_db_seeded():
    """Populate RAGAS DB by running predict on all three full input CSVs.

    Temporarily overrides SC_PREDICTION_DB_PATH to RAGAS_DB, runs the pipeline
    on each full input file (5 000 rows each), then restores the eval DB path.
    """
    from src.daily_predict import DailyPredictionPipeline
    RAGAS_DB.parent.mkdir(parents=True, exist_ok=True)
    _copy_hist_tables(PIPELINE_DIR / "db" / "delivery_predictions.db", RAGAS_DB)

    original = os.environ.get("SC_PREDICTION_DB_PATH", str(EVAL_DB))
    os.environ["SC_PREDICTION_DB_PATH"] = str(RAGAS_DB)
    try:
        for csv_file in sorted(FULL_INPUT_DIR.glob("daily_delivery_logistics_*.csv")):
            DailyPredictionPipeline.get_prediction(str(csv_file), "")
    finally:
        os.environ["SC_PREDICTION_DB_PATH"] = original

    return RAGAS_DB


# ── 6. Expose key paths as fixtures for tests that need them ──────────────────

@pytest.fixture(scope="session")
def eval_db_path() -> Path:
    return EVAL_DB


@pytest.fixture(scope="session")
def ragas_db_path() -> Path:
    return RAGAS_DB


@pytest.fixture(scope="session")
def full_input_file_path() -> Path:
    return FULL_INPUT_FILE




# ── 7. Report writer ──────────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Write a human-readable markdown eval report after the session completes."""
    from judge import _records, mean_score
    if not _records:
        return

    # Merge RAGAS scores from the RAG test into the Recommendation Agent record.
    # When both standard and RAGAS tests run in one session, merge so a single
    # "Recommendation Expert Agent" row in the report shows both score types.
    rag_rec = next((r for r in _records if r["agent"] == "RAG — SLA Knowledge Retrieval"), None)
    rec_rec = next((r for r in _records if r["agent"] == "Recommendation Expert Agent"), None)
    if rag_rec and rec_rec and rag_rec.get("ragas_scores") and not rec_rec.get("ragas_scores"):
        rec_rec["ragas_scores"] = rag_rec["ragas_scores"]
        _records.remove(rag_rec)

    # Sort records into pipeline order: predict → diagnose → recommend → simulate → email
    _REPORT_ORDER = [
        "Predict Delivery Delays",
        "Diagnose Delay Patterns",
        "Recommendation Expert Agent",
        "Simulate Delay Prediction",
        "Email Alert Agent",
    ]
    _records.sort(key=lambda r: _REPORT_ORDER.index(r["agent"]) if r["agent"] in _REPORT_ORDER else len(_REPORT_ORDER))

    from datetime import datetime
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = REPORTS_DIR / f"eval_report_{timestamp}.md"

    passed = sum(1 for r in _records if r["mean"] >= 3.0)
    failed = len(_records) - passed

    lines = [
        "# Supply Chain Delivery Agent — Eval Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** gpt-5.4 / gpt-4.1-mini (mini tasks)  ",
        f"**Input dataset:** {FULL_INPUT_FILE.name} (5 000 orders, full pipeline run)  ",
        f"**Pass threshold:** mean score ≥ 3.0\n",
        "---\n",
        "## Summary\n",
        "| Agent | Relevance | Faithfulness | Safety | Mean | Result |",
        "|-------|-----------|--------------|--------|------|--------|",
    ]

    for r in _records:
        status = "PASS" if r["mean"] >= 3.0 else "FAIL"
        lines.append(
            f"| {r['agent']} "
            f"| {r['relevance']:.1f}/5 "
            f"| {r['faithfulness']:.1f}/5 "
            f"| {r['safety']:.1f}/5 "
            f"| **{r['mean']:.2f}** "
            f"| {status} |"
        )

    lines += [
        f"\n**Total:** {len(_records)}  **Passed:** {passed}  **Failed:** {failed}\n",
        "---\n",
        "## Detailed Results\n",
    ]

    for r in _records:
        status = "PASS" if r["mean"] >= 3.0 else "FAIL"
        ragas = r.get("ragas_scores")
        ragas_rows = []
        if ragas:
            ragas_rows = [
                "",
                "**RAGAS Scores** *(0.0 – 1.0 scale, pass threshold ≥ 0.60)*",
                "",
                "| Metric | Score | Pass |",
                "|--------|-------|------|",
            ]
            for metric, val in ragas.items():
                label = metric.replace("_", " ").title()
                ok = "Yes" if val >= 0.60 else "No"
                ragas_rows.append(f"| {label} | {val:.3f} | {ok} |")

        lines += [
            f"### {r['agent']}  —  {status}",
            "",
            "**Input**",
            "```",
            r["input"] if r["input"] else "(not provided)",
            "```",
            "",
            "**Output** *(truncated to 3000 chars)*",
            "```",
            r["output"][:3000],
            "```",
            "",
            "**LLM Judge Scores** *(1 – 5 scale, pass threshold ≥ 3.0)*",
            "",
            "| Dimension | Score |",
            "|-----------|-------|",
            f"| Relevance | {r['relevance']:.1f} / 5 |",
            f"| Faithfulness | {r['faithfulness']:.1f} / 5 |",
            f"| Safety | {r['safety']:.1f} / 5 |",
            f"| **Mean** | **{r['mean']:.2f}** |",
            *ragas_rows,
            "",
            "**Reasoning**",
            "",
            f"> {r['reasoning']}",
            "",
            "---\n",
        ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEval report written: {report_path}")

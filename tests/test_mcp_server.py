"""
Smoke tests for the Prediction Pipeline MCP server tools.

Calls the three MCP tool functions directly (no transport overhead) and
verifies that each returns well-formed JSON with the expected structure.
No LLM calls are made — pure pipeline + DB verification.

Run from the project root:
    uv run pytest tests/test_mcp_server.py -v
"""
import sys
import json
from pathlib import Path
import pytest
import pytest_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PP = PROJECT_ROOT / "prediction_pipeline"
sys.path.insert(0, str(PP))

from prediction_server import predict_delivery_delays, get_delay_diagnosis, simulate_order_delays

INPUT_CSV = str(PROJECT_ROOT / "supply_chain_delivery_app" / "input" / "daily_delivery_logistics_1.csv")


# ---------------------------------------------------------------------------
# Session fixture: run predict once, share result with all predict tests
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def predict_data():
    """Run the prediction pipeline once for the session."""
    raw = await predict_delivery_delays(INPUT_CSV)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# predict_delivery_delays
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_summary_fields(predict_data):
    """Summary dict must contain all required aggregate fields."""
    summary = predict_data.get("summary", {})
    for field in ["total_orders", "total_delayed", "pct_delayed",
                  "severity_short", "severity_medium", "severity_long",
                  "top_regions", "top_weather", "top_partners"]:
        assert field in summary, f"Missing field in summary: {field}"


@pytest.mark.asyncio
async def test_predict_total_orders_positive(predict_data):
    assert predict_data["summary"]["total_orders"] > 0


@pytest.mark.asyncio
async def test_predict_has_delayed_orders(predict_data):
    delayed = predict_data.get("delayed_orders", [])
    assert isinstance(delayed, list) and len(delayed) > 0


@pytest.mark.asyncio
async def test_predict_delayed_order_has_required_keys(predict_data):
    row = predict_data["delayed_orders"][0]
    assert "delivery_id" in row
    assert "delay_reason" in row


@pytest.mark.asyncio
async def test_predict_formatted_stats_present(predict_data):
    assert "formatted_stats" in predict_data
    assert len(predict_data["formatted_stats"]) > 0


# ---------------------------------------------------------------------------
# get_delay_diagnosis  (depends on predict having populated the DB)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def diagnosis_data(predict_data):
    raw = await get_delay_diagnosis()
    return json.loads(raw)


@pytest.mark.asyncio
async def test_diagnosis_no_upstream_error(diagnosis_data):
    assert diagnosis_data.get("Error") != "upstream_missing", \
        f"Diagnosis upstream error: {diagnosis_data.get('message')}"


@pytest.mark.asyncio
async def test_diagnosis_has_overall_kpis(diagnosis_data):
    assert "overall_daily" in diagnosis_data, \
        f"Missing 'overall_daily' in diagnosis. Got keys: {list(diagnosis_data.keys())}"


@pytest.mark.asyncio
async def test_diagnosis_has_dimension_data(diagnosis_data):
    assert len(diagnosis_data) > 1, "Diagnosis response has no dimension breakdown"


# ---------------------------------------------------------------------------
# simulate_order_delays  (depends on predict having populated the DB)
# simulate_order_delays returns a Markdown-formatted report string
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def simulate_data(predict_data):
    """Run a simulation for east region under stormy weather."""
    return await simulate_order_delays(
        scenario="Stormy weather impact in east region",
        filters='{"region": "east"}',
        changes='{"weather_condition": "stormy"}',
    )


@pytest.mark.asyncio
async def test_simulate_returns_text(simulate_data):
    assert isinstance(simulate_data, str) and len(simulate_data) > 0


@pytest.mark.asyncio
async def test_simulate_reports_rows_affected(simulate_data):
    assert "Rows affected" in simulate_data, \
        "Simulation report missing 'Rows affected' header line"


@pytest.mark.asyncio
async def test_simulate_mentions_filter_region(simulate_data):
    assert "east" in simulate_data.lower(), \
        "Simulation report does not mention the 'east' region filter"

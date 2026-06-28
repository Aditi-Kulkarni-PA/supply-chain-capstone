"""
Smoke tests for Pydantic output models in delivery_agents.py.

Validates that the structured output models accept valid data and reject
invalid data — ensuring agent outputs are correctly validated at runtime.

Run from the project root:
    python -m pytest tests/test_pydantic_models.py -v
"""

import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

# Add supply_chain_delivery_app to path
APP = Path(__file__).resolve().parent.parent / "supply_chain_delivery_app"
sys.path.insert(0, str(APP.parent))  # project root so 'supply_chain_delivery_app' is a package

# Import only the Pydantic models — avoids loading OpenAI Agents SDK / MCP servers
sys.path.insert(0, str(APP))
from delivery_agents import (
    RowEnrichment,
    TopEntry,
    DeliveryDelaySummary,
    DeliveryDelayPredictionResult,
    DiagnosisHighRisk,
    DiagnosisComparison,
    DelayDiagnosisResult,
    SimulateDelays,
    SimulationsList,
)


# ── RowEnrichment ─────────────────────────────────────────────────────────────

def test_row_enrichment_valid():
    obj = RowEnrichment(
        delivery_id="D001",
        llm_insights="High schedule_risk due to same-day mode in stormy weather.",
    )
    assert obj.delivery_id == "D001"
    assert len(obj.llm_insights) >= 10


def test_row_enrichment_rejects_short_insights():
    with pytest.raises(ValidationError):
        RowEnrichment(delivery_id="D001", llm_insights="short")


# ── TopEntry ──────────────────────────────────────────────────────────────────

def test_top_entry_valid():
    obj = TopEntry(name="North", count=42, pct=18.5)
    assert obj.name == "North"
    assert obj.count == 42


# ── DeliveryDelaySummary ──────────────────────────────────────────────────────

def test_delivery_delay_summary_valid():
    obj = DeliveryDelaySummary(
        total_orders=1000,
        total_delayed=260,
        pct_delayed=26.0,
        severity_short=100,
        severity_medium=100,
        severity_long=60,
    )
    assert obj.total_orders == 1000
    assert obj.top_regions == []       # default_factory=list


def test_delivery_delay_summary_with_top_entries():
    obj = DeliveryDelaySummary(
        total_orders=500,
        total_delayed=130,
        pct_delayed=26.0,
        severity_short=50,
        severity_medium=50,
        severity_long=30,
        top_regions=[TopEntry(name="East", count=40, pct=30.8)],
    )
    assert len(obj.top_regions) == 1
    assert obj.top_regions[0].name == "East"


# ── DeliveryDelayPredictionResult ─────────────────────────────────────────────

def test_prediction_result_valid():
    obj = DeliveryDelayPredictionResult(
        predict_summary="26% of orders delayed. High schedule_risk in North.",
        delayed_orders=[
            RowEnrichment(
                delivery_id="D001",
                llm_insights="Stormy weather combined with same-day mode drove high schedule_risk.",
            )
        ],
    )
    assert len(obj.delayed_orders) == 1


# ── DiagnosisHighRisk ─────────────────────────────────────────────────────────

def test_diagnosis_high_risk_valid():
    obj = DiagnosisHighRisk(
        pattern_type="mode_weather",
        pattern_description="same_day + Stormy",
        total_deliveries=50,
        delayed_count=35,
        delay_rate_pct=70.0,
        risk_level="critical",
    )
    assert obj.risk_level == "critical"


# ── DiagnosisComparison ───────────────────────────────────────────────────────

def test_diagnosis_comparison_valid():
    obj = DiagnosisComparison(
        dimension="region",
        category="North",
        daily_total=100,
        daily_delayed=30,
        daily_delay_rate_pct=30.0,
        hist_total=5000,
        hist_delayed=1300,
        hist_delay_rate_pct=26.0,
        rate_change_pct=4.0,
    )
    assert obj.rate_change_pct == 4.0


# ── DelayDiagnosisResult ──────────────────────────────────────────────────────

def test_delay_diagnosis_result_valid():
    obj = DelayDiagnosisResult(
        high_risk_patterns=[
            DiagnosisHighRisk(
                pattern_type="mode_weather",
                pattern_description="same_day + Stormy",
                total_deliveries=50,
                delayed_count=35,
                delay_rate_pct=70.0,
                risk_level="critical",
            )
        ],
        comparison=[
            DiagnosisComparison(
                dimension="region",
                category="East",
                daily_total=80,
                daily_delayed=25,
                daily_delay_rate_pct=31.25,
                hist_total=4000,
                hist_delayed=1040,
                hist_delay_rate_pct=26.0,
                rate_change_pct=5.25,
            )
        ],
        diagnosis_summary="North region shows 4pp increase in delay rate vs historical.",
    )
    assert len(obj.high_risk_patterns) == 1
    assert len(obj.comparison) == 1


# ── SimulateDelays / SimulationsList ─────────────────────────────────────────

def test_simulate_delays_valid():
    obj = SimulateDelays(
        delivery_id="D042",
        delivery_partner="PartnerA",
        delivery_mode="same_day",
        region="North",
        weather_condition="stormy",
        vehicle_type="bike",
        distance_km="150",
        original_severity="Short",
        simulated_severity="Long",
        simulate_delay_reason="Bike on 150 km in storm exceeds vehicle capacity.",
    )
    assert obj.simulated_severity == "Long"


def test_simulations_list_valid():
    sim = SimulateDelays(
        delivery_id="D001",
        delivery_partner="PartnerB",
        delivery_mode="express",
        region="South",
        weather_condition="rainy",
        vehicle_type="van",
        distance_km="80",
        original_severity="Medium",
        simulated_severity="Short",
        simulate_delay_reason=None,
    )
    obj = SimulationsList(simulations=[sim])
    assert len(obj.simulations) == 1

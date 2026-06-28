
"""
Supply Chain Delivery – Agent definitions.

Key logic:
    - Defines all Pydantic models for agent tool I/O (predict, diagnose, simulate, recommend, email)
    - Sets up MCP server for prediction/diagnosis tools (pipeline_mcp)
    - Each sub-agent is an Agent with its own prompt, model, and tool wiring
    - Master orchestrator agent (supply_chain_delivery_master_agent) coordinates all tools and output
    - All tool calls are strongly typed and validated via Pydantic
    - Model selection, tool_choice, and output_type are set for each agent
    - Fallback and formatting agents handle edge cases and summary formatting
"""

import os
import sys
from pathlib import Path

# Ensure this directory is importable (for config/ and tools/ packages)
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(dotenv_path=find_dotenv(), override=False)

from agents import Agent, Runner, trace, WebSearchTool, set_default_openai_api, set_default_openai_client
from agents.mcp import MCPServerStdio
from agents.model_settings import ModelSettings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional

from config import get_instruction
from tools import (
    recommend_actions,
    fetch_delayed_orders_for_email,
)

# ---------------------------------------------------------------------------
# MCP server for predict + diagnosis (lives in prediction_pipeline/): shared by sub-agents
# ---------------------------------------------------------------------------
_PREDICTION_SERVER = str(
    _APP_DIR.parent / "prediction_pipeline" / "prediction_server.py"
)
_PYTHON = sys.executable

_MCP_PARAMS = {"command": _PYTHON, "args": [_PREDICTION_SERVER]}

# One MCP server instance shared by both prediction sub-agents.
# tool_filter restricts the agent to only the two pipeline tools, blocking
# everything else that may exist on the server now or in the future.
# Each sub-agent's prompt further pins it to its specific tool by name.
pipeline_mcp = MCPServerStdio(
    name="prediction_pipeline",
    params=_MCP_PARAMS,
    tool_filter={"allowed_tool_names": ["predict_delivery_delays", "get_delay_diagnosis", "simulate_order_delays"]},
    client_session_timeout_seconds=120,
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MODEL_MINI = os.getenv("OPENAI_MODEL_MINI", "gpt-4.1-mini")


def _configure_llm_backend() -> None:
    """Configure provider routing for OpenAI cloud or local LM Studio."""
    backend = os.getenv("LLM_BACKEND", "openai").strip().lower()
    if backend != "lmstudio":
        return

    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").strip()
    api_key = os.getenv("LLM_API_KEY", "lm-studio").strip() or "lm-studio"
    local_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_default_openai_client(local_client)
    # LM Studio's OpenAI-compatible server is generally most reliable via chat completions.
    set_default_openai_api("chat_completions")


_configure_llm_backend()


# ---------------------------------------------------------------------------
# 1. Predict delivery delays — Pydantic models and agent/tool definition
# ---------------------------------------------------------------------------

class RowEnrichment(BaseModel):
    """Slim model: only delivery_id + llm_insights.
    The full row data lives in the CSV on disk — no need for the LLM to copy it."""
    delivery_id: str = Field(description="Delivery ID — must match the value from the tool's delayed_orders")
    llm_insights: str = Field(min_length=10, description="REQUIRED — 1-2 sentence cross-functional explanation referencing at least two derived features (e.g. schedule_risk, vehicle_load_strain, km_per_expected_hr, vehicle_type). Must not be empty.")

class TopEntry(BaseModel):
    name: str = Field(description="Category name (e.g. region name, weather condition, partner name)")
    count: int = Field(description="Number of delayed orders in this category")
    pct: float = Field(description="Percentage of total delayed orders")

class DeliveryDelaySummary(BaseModel):
    total_orders: int = Field(description="Total orders analysed")
    total_delayed: int = Field(description="Total predicted delayed orders")
    pct_delayed: float = Field(description="Percentage of orders predicted delayed")
    severity_short: int = Field(description="Count of Short (1-2h) delayed orders")
    severity_medium: int = Field(description="Count of Medium (3-5h) delayed orders")
    severity_long: int = Field(description="Count of Long (6+h) delayed orders")
    delayed_csv_path: str = Field(default="", description="Path to the delayed-only prediction CSV")
    showing_top_n: int = Field(default=0, description="Number of delayed rows shown in the table")
    top_regions: list[TopEntry] = Field(default_factory=list, description="Top affected regions")
    top_weather: list[TopEntry] = Field(default_factory=list, description="Top affected weather conditions")
    top_partners: list[TopEntry] = Field(default_factory=list, description="Top affected delivery partners")
    enrich_rows_cap: int = Field(default=50, description="Number of rows sent to the agent for delay_reason enrichment (SC_MCP_ENRICH_ROWS)")

class DeliveryDelayPredictionResult(BaseModel):
    predict_summary: str = Field(description="Cross-dimensional insight paragraph written by the agent — Markdown bullets with quantitative derived-feature stats")
    delayed_orders: list[RowEnrichment] = Field(
        default_factory=list,
        description="One {delivery_id, llm_insights} entry per delayed row. Must have exactly enrich_rows_cap entries, each with non-empty llm_insights.",
    )

predict_delivery_delays_agent = Agent(
    name="Predict Delivery Delays",
    instructions=get_instruction("predict_delivery_delays"),
    model=MODEL,
    mcp_servers=[pipeline_mcp],
    model_settings=ModelSettings(tool_choice="required", temperature=0),
    output_type=DeliveryDelayPredictionResult,
)
predict_delivery_delays_tool = predict_delivery_delays_agent.as_tool(
    tool_name="predict_delivery_delays_tool",
    tool_description="Run the two-stage ML pipeline to predict delayed orders and classify severity",
)


# ---------------------------------------------------------------------------
# 3. Diagnose delay patterns — Pydantic models and agent/tool definition
# ---------------------------------------------------------------------------

class DiagnosisHighRisk(BaseModel):
    pattern_type: str = Field(description="Type of pattern combination (mode_weather, mode_distance, weather_vehicle)")
    pattern_description: str = Field(description="Human-readable pattern description (e.g. 'same_day + Stormy')")
    total_deliveries: int = Field(description="Total deliveries matching this pattern")
    delayed_count: int = Field(description="Number of delayed deliveries")
    delay_rate_pct: float = Field(description="Delay rate as percentage")
    risk_level: str = Field(description="Risk level: critical (50%+), high (40-50%), medium (30-40%)")

class DiagnosisComparison(BaseModel):
    dimension: str = Field(description="Dimension name (region, weather_condition, delivery_partner, etc.)")
    category: str = Field(description="Category value (East, Stormy, DHL, etc.)")
    daily_total: int = Field(description="Today's total deliveries for this category")
    daily_delayed: int = Field(description="Today's delayed count")
    daily_delay_rate_pct: float = Field(description="Today's delay rate %")
    hist_total: int = Field(description="Historical total deliveries")
    hist_delayed: int = Field(description="Historical delayed count")
    hist_delay_rate_pct: float = Field(description="Historical delay rate %")
    rate_change_pct: float = Field(description="Change in delay rate (daily - hist), negative means improvement")

class DelayDiagnosisResult(BaseModel):
    high_risk_patterns: list[DiagnosisHighRisk] = Field(description="High-risk delay pattern combinations for today")
    comparison: list[DiagnosisComparison] = Field(description="Today vs historical delay rate comparison across all dimensions")
    diagnosis_summary: str = Field(default="", description="Formatted Markdown summary of delay pattern diagnosis, generated by the agent")

diagnose_delay_patterns_agent = Agent(
    name="Diagnose & Analyse Delay Patterns",
    instructions=get_instruction("diagnose_delay_patterns"),
    model=MODEL,
    mcp_servers=[pipeline_mcp],
    model_settings=ModelSettings(tool_choice="required", temperature=0),
    output_type=DelayDiagnosisResult,
)
diagnose_delay_patterns_tool = diagnose_delay_patterns_agent.as_tool(
    tool_name="diagnose_delay_patterns",
    tool_description="Diagnose delay patterns comparing today's predictions vs historical data across all dimensions",
)


# ---------------------------------------------------------------------------
# 4. Delay simulation — Pydantic models and agent/tool definition
# ---------------------------------------------------------------------------

class SimulateDelays(BaseModel):
    delivery_id: str = Field(description="Delivery ID")
    delivery_partner: str = Field(description="Delivery Partner")
    delivery_mode: str = Field(description="Delivery Mode")
    region: str = Field(description="Region")
    weather_condition: str = Field(description="Simulated weather condition")
    vehicle_type: str = Field(description="Simulated vehicle type")
    distance_km: str = Field(description="Distance in km")
    original_severity: str = Field(description="Original predicted severity label")
    simulated_severity: str = Field(description="Simulated severity under new conditions")
    simulate_delay_reason: Optional[str] = Field(description="Reason for simulated delay")

class SimulationsList(BaseModel):
    simulations: list[SimulateDelays] = Field(description="List of simulations for order delivery delays")

delay_simulation_agent = Agent(
    name="Simulate & Analyse Delay Prediction",
    instructions=get_instruction("delay_simulation"),
    model=MODEL,
    mcp_servers=[pipeline_mcp],
    model_settings=ModelSettings(tool_choice="required", temperature=0),
    output_type=SimulationsList,
)
delay_simulations_tool = delay_simulation_agent.as_tool(
    tool_name="delay_simulations_tool",
    tool_description="Simulate weather and traffic conditions to analyze delay impact",
)


# ---------------------------------------------------------------------------
# 5. Recommendation — Pydantic models and agent/tool definition
# ---------------------------------------------------------------------------

class RecommendedAction(BaseModel):
    action: str = Field(description="Recommendation Action - Short Description")
    action_desc: str = Field(description="Recommendation Action - Full Description with supporting data")
    category: Literal["quick-win", "short-term", "long-term"] = Field(description="One of: quick-win, short-term, long-term")
    dimension: str = Field(description="Which dimension this targets: delivery_mode, weather, region, vehicle, partner, or general")
    supporting_data: str = Field(description="Specific numbers from the analysis that justify this recommendation")
    sla_reference: str = Field(description="Quote the actual SLA text from the Retrieved Sections — include the section heading and specific metric, target, penalty, or rule. Example: 'SLA 2.1 On-Time Delivery Commitments by Mode: Express current target OTD is 40%. SLA 3.2 Weather-Specific Operational Protocols: Stormy — halt same-day and express dispatches if wind speed > 60 km/h.' Do NOT write generic labels like 'SLA Reference 3' — always quote the content itself.")

class RecommendedActionsList(BaseModel):
    recommended_actions: list[RecommendedAction] = Field(
        description="List of recommended actions for delivery optimization. "
                    "MUST contain at least 3 quick-win, 3 short-term, AND 3 long-term actions (9+ total).",
        min_length=9,
    )

recommendation_agent = Agent(
    name="Recommendation Expert Agent to Optimize Order Delivery",
    instructions=get_instruction("recommendation"),
    model=MODEL,
    tools=[recommend_actions],
    model_settings=ModelSettings(tool_choice="required", temperature=0),
    output_type=RecommendedActionsList,
)
recommendation_tool = recommendation_agent.as_tool(
    tool_name="recommendation_tool",
    tool_description="Recommendations to optimize order delivery and minimize delays",
)


# ---------------------------------------------------------------------------
# 6. Email alert — Pydantic models and agent/tool definition
# ---------------------------------------------------------------------------

class EmailAlert(BaseModel):
    email_content: str = Field(description="Professional email body to notify the customer about their delayed order.")
    email_id: str = Field(description="Email ID of the customer; use a realistic placeholder if unknown.", default="first.last@domain.com")

class EmailsList(BaseModel):
    content: Annotated[
        list[EmailAlert],
        Field(
            description="List of emails to be sent to customers whose orders are delayed. If there is at least one delayed order, this MUST have at least one item.",
            min_length=1,
        ),
    ]

email_alert_agent = Agent(
    name="Email Alert Agent",
    instructions=get_instruction("email_alert"),
    model=MODEL,
    tools=[fetch_delayed_orders_for_email],
    model_settings=ModelSettings(tool_choice="required", temperature=0),
    output_type=EmailsList,
)
email_alert_tool = email_alert_agent.as_tool(
    tool_name="email_alert_tool",
    tool_description="Emails to be sent to customers",
)


# ---------------------------------------------------------------------------
# 7. Fallback advisor agent (for handoff; no Pydantic output)
# ---------------------------------------------------------------------------

fallback_advisor_agent = Agent(
    name="Fallback Supply Chain Optimization Agent",
    instructions=get_instruction("fallback_advisor"),
    model=MODEL,
)


# ---------------------------------------------------------------------------
# 8. Format summary agent — lightweight formatter (uses MODEL_MINI)
# ---------------------------------------------------------------------------

format_summary_agent = Agent(
    name="Summary Formatting Specialist",
    instructions=get_instruction("format_summary"),
    model=MODEL_MINI,
    model_settings=ModelSettings(temperature=0),
)
format_summary_tool = format_summary_agent.as_tool(
    tool_name="format_summary_tool",
    tool_description=(
        "Format structured data into a clean Markdown summary. "
        "Pass a message with: summary_type (predict, diagnosis, simulate, recommendation, email_alert) "
        "and the raw data from the domain tool."
    ),
)


# ---------------------------------------------------------------------------
# 9. Master orchestrator — Pydantic output and agent: coordinates all tools, validates output
# ---------------------------------------------------------------------------

class DelayEnrichment(BaseModel):
    """Slim model for passing agent-enriched llm_insights back from master.
    Only delivery_id + llm_insights are needed — avoids copying all 15 fields
    per row which causes the LLM to truncate the list."""
    delivery_id: str = Field(description="Delivery ID — must match the value from delayed_orders")
    llm_insights: str = Field(min_length=10, description="REQUIRED — Agent-enriched cross-functional delay explanation. Must not be empty.")

class MasterOutput(BaseModel):
    predict_summary: str = Field(description="Prediction Output Summary")
    predict_rows: list[DelayEnrichment] = Field(default_factory=list, description="delivery_id + enriched llm_insights for each delayed order (copied from predict agent's delayed_orders)")
    simulate_summary: str = Field(description="Simulation Output Summary")
    simulate_rows: list[SimulateDelays] = Field(default_factory=list, description="Rows of simulated orders with delay hours")
    diagnosis_summary: str = Field(description="Delay Patterns Diagnosis Summary")
    diagnosis_high_risk_rows: list[DiagnosisHighRisk] = Field(default_factory=list, description="High-risk delay pattern combinations")
    diagnosis_comparison_rows: list[DiagnosisComparison] = Field(default_factory=list, description="Today vs historical comparison across dimensions")
    recommendation_summary: str = Field(description="Recommended Actions Summary")
    recommendation_rows: list[RecommendedAction] = Field(default_factory=list, description="All recommended actions copied from recommendation_tool output — the app builds the detailed display from these rows")
    email_alert_summary: str = Field(description="Email Alert Summary")
    email_alerts: Optional[EmailsList] = Field(default=None, description="Structured list of email alerts from email_alert_tool.")

supply_chain_delivery_master_agent = Agent(
    name="Supply Chain Last-Mile Delivery Optimization Expert Agent",
    instructions=get_instruction("master_expert"),
    model=MODEL,
    tools=[
        predict_delivery_delays_tool,
        diagnose_delay_patterns_tool,
        delay_simulations_tool,
        recommendation_tool,
        email_alert_tool,
        format_summary_tool,
    ],
    model_settings=ModelSettings(tool_choice="auto"),
    handoffs=[fallback_advisor_agent],
    handoff_description="Use this when no tool results and datasets are found and you need alternative suggestions",
    output_type=MasterOutput,
)

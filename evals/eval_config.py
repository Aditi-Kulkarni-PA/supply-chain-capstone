"""Eval thresholds — all pass/fail cutoffs in one place."""

# ── Schema / content minimums ─────────────────────────────────────────────────
MIN_DELAYED_ORDERS      = 1    # predict must find at least 1 delay
MIN_HIGH_RISK_PATTERNS  = 1    # diagnose must return at least 1 pattern
MIN_COMPARISON_ROWS     = 2    # diagnose comparison must have >= 2 rows
MIN_SIMULATIONS         = 1    # simulate must return at least 1 row
MIN_RECOMMENDATIONS     = 9    # 3 quick-win + 3 short-term + 3 long-term
MIN_EMAILS              = 3    # email agent must produce 3 rendered sample emails

# ── LLM-as-judge (1–5 scale, mean across criteria) ───────────────────────────
MIN_JUDGE_SCORE         = 3.0  # mean score must be >= this to pass

# ── RAGAS thresholds (runs as part of the default suite — no gating flag) ────
# 0.60 is the documented acceptance bar for all four metrics; hallucination rate
# is the complement of faithfulness, so its bar is the complement of 0.60.
MIN_FAITHFULNESS        = 0.60
MIN_ANSWER_RELEVANCY    = 0.60
MIN_CONTEXT_PRECISION   = 0.60  # are retrieved SLA chunks actually relevant to the topic query?
MAX_HALLUCINATION_RATE  = 0.40  # derived as 1 - faithfulness

# ── Latency (seconds) ─────────────────────────────────────────────────────────
MAX_PREDICT_LATENCY_S   = 600   # 5 000-row ML pipeline + LLM
MAX_DIAGNOSE_LATENCY_S  = 300
MAX_SIMULATE_LATENCY_S  = 300
MAX_RECOMMEND_LATENCY_S = 300
MAX_EMAIL_LATENCY_S     = 300
MAX_MASTER_LATENCY_S    = 1200  # full end-to-end pipeline, all 5 tools

# ── Feature concepts the predict agent must reference in llm_insights ────────
# llm_insights is English prose — match human-readable phrases, not snake_case.
PREDICT_FEATURE_NAMES = {
    # English equivalents the LLM writes
    "schedule risk", "vehicle load strain", "km per expected hour",
    "vehicle type", "load strain", "km per",
    "mode urgency", "weather severity", "carrier avg schedule",
    "carrier average schedule", "weight x distance", "weight-distance",
    "cost per kg",
    # Snake-case fallback in case the LLM uses them verbatim
    "schedule_risk", "vehicle_load_strain", "km_per_expected_hr", "vehicle_type",
    "mode_urgency", "weather_severity", "carrier_avg_schedule",
    "weight_x_distance", "cost_per_kg",
}

# ── SLA grounding keywords the recommend agent must include in sla_reference ──
SLA_KEYWORDS = {"sla", "ola", "penalty", "threshold", "target"}

"""
Shared utilities for the delivery chat app.

Contains: tool-call argument formatting, sidecar freshness detection,
diagnosis sidecar persistence, directory-scanning helpers,
action-plan building (intent detection → dependency-aware step list),
and confirmation / pending-query resolution for the chat handler.
"""

import json as _json
import re as _re
import time as _time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent.parent  # helpers/ → app root

PREDICT_SIDECAR = _APP_DIR / "output" / "daily_delivery_delay_prediction_meta.json"
DIAG_SIDECAR = _APP_DIR / "output" / "diagnosis_meta.json"
FRESHNESS_TTL = 3600  # 1 hour
DISPLAY_ROWS: int = int(__import__("os").getenv("SC_MCP_DISPLAY_ROWS", 50))


# ---------------------------------------------------------------------------
# Tool-call argument formatting (for progress/status log)
# ---------------------------------------------------------------------------

def brief_args(raw_args, *, max_val: int = 80, max_total: int = 120) -> str:
    """Return a short representation of tool-call arguments for the status log."""
    s = str(raw_args or "")
    try:
        parsed = _json.loads(s)
        parts = []
        for k, v in parsed.items():
            v_str = str(v)
            if len(v_str) > max_val:
                v_str = v_str[: max_val - 3] + "..."
            parts.append(f"{k}={v_str}")
        result = "(" + ", ".join(parts) + ")"
    except Exception:
        result = s
    if len(result) > max_total:
        result = result[: max_total - 3] + "..."
    return result


# ---------------------------------------------------------------------------
# Sidecar freshness helpers
# ---------------------------------------------------------------------------

def is_file_fresh(path: Path, ttl: int = FRESHNESS_TTL) -> bool:
    """Return True if *path* exists and was modified less than *ttl* seconds ago."""
    return path.is_file() and (_time.time() - path.stat().st_mtime) < ttl


def build_freshness_system_msg() -> str:
    """Return a SYSTEM tag describing prediction/diagnosis freshness."""
    p_fresh = is_file_fresh(PREDICT_SIDECAR)
    d_fresh = is_file_fresh(DIAG_SIDECAR)
    if p_fresh and d_fresh:
        return (
            "\n\n[SYSTEM: Prediction AND Diagnosis pipeline outputs are FRESH (generated <1h ago). "
            "All tools can proceed WITHOUT re-running predict_delivery_delays_tool or diagnose_delay_patterns_tool.]"
        )
    if p_fresh:
        return (
            "\n\n[SYSTEM: Prediction pipeline output is FRESH (generated <1h ago). "
            "Diagnosis output is NOT FRESH. "
            "Run diagnose_delay_patterns_tool before recommendation_tool. "
            "Predict does NOT need to be re-run.]"
        )
    return (
        "\n\n[SYSTEM: Prediction pipeline output is NOT FRESH. "
        "Run predict_delivery_delays_tool first as a pre-requisite before calling "
        "diagnose_delay_patterns_tool or any tool that depends on prediction data.]"
    )


def save_diagnosis_sidecar(diagnosis_text: str) -> None:
    """Write diagnosis sidecar JSON so the next run can detect freshness."""
    DIAG_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    DIAG_SIDECAR.write_text(_json.dumps({"diagnosis_summary": diagnosis_text[:500]}))


# ---------------------------------------------------------------------------
# Directory scanning helpers
# ---------------------------------------------------------------------------

def list_dir_files(directory: Path) -> list[str]:
    """Return resolved string paths for all files in *directory*, or [] if missing."""
    if directory.is_dir():
        return [str(p.resolve()) for p in directory.iterdir() if p.is_file()]
    return []


knowledge_files = list_dir_files(_APP_DIR / "knowledge")
input_files = list_dir_files(_APP_DIR / "input")


# ═══════════════════════════════════════════════════════════════════════════
# Action-plan builder — intent detection + dependency-aware step ordering
# ═══════════════════════════════════════════════════════════════════════════

# Each tool's prerequisite chain.  Steps marked "(if not fresh)" are skipped
# when the corresponding sidecar file is already fresh.
TOOL_DEPS: dict[str, list[str]] = {
    "predict":   ["Predict delivery delays"],
    "diagnose":  ["Predict delivery delays (if not fresh)", "Diagnose delay patterns"],
    "simulate":  ["Predict delivery delays (if not fresh)", "Simulate what-if scenarios"],
    "recommend": ["Predict delivery delays (if not fresh)", "Diagnose delay patterns (if not fresh)", "Generate optimization recommendations"],
    "email":     ["Predict delivery delays (if not fresh)", "Generate customer email alerts"],
}

# Keyword patterns → tool key (matches TOOL_DEPS keys).  Order = display priority.
TOOL_KEYWORDS: list[tuple[str, str]] = [
    (r"predict|delay|late|sla|on.?time|delivery performance|meet.*target"
     r"|kpi|metric|how.*deliver|missed|overdue|backlog|behind schedule",
     "predict"),
    (r"diagnos|pattern|root cause|why delays|compare historical"
     r"|trend|breakdown|analysis|insight|driver|factor",
     "diagnose"),
    (r"simulat|what.if|weather change|scenario|hypothetical|if we",
     "simulate"),
    (r"recommend|optimiz|improve|reduce delays|action|suggestion"
     r"|fix|mitigat|resolve|what should",
     "recommend"),
    (r"email|alert|notify customers|customer communication|customer alert"
     r"|send.*message|inform customer|outreach",
     "email"),
]

# Composite intents: certain phrases imply multiple tool keys together
_COMPOSITE_INTENTS: list[tuple[str, list[str]]] = [
    (r"sla|on.?time|adherence|meet.*target|kpi|delivery performance|performance review",
     ["predict", "diagnose", "recommend"]),
]

# Conversational messages that should NOT trigger plan/clarification
_GREETING_RE = _re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|good morning|good evening"
    r"|how are you|what can you do|help|who are you)[\s\?\!\.]*$",
    _re.IGNORECASE,
)

# Full-pipeline triggers
_FULL_PIPELINE_RE = _re.compile(
    r"full pipeline|run all|full analysis|dashboard", _re.IGNORECASE,
)

# Canonical ordering for deduplication (predict < diagnose < simulate < recommend < email)
_STEP_ORDER = {
    "Predict delivery delays": 0,
    "Diagnose delay patterns": 1,
    "Simulate what-if scenarios": 2,
    "Generate optimization recommendations": 3,
    "Generate customer email alerts": 4,
}


def _resolve_freshness(step: str) -> str:
    """Apply freshness check to a step with '(if not fresh)' qualifier.

    Returns the base step label if data is stale, or a 'Skip …' label
    if data is fresh.  Steps without the qualifier are returned unchanged.
    """
    if "(if not fresh)" not in step:
        return step
    base = step.replace(" (if not fresh)", "")
    if "Predict" in base and is_file_fresh(PREDICT_SIDECAR):
        return f"{base} -- skip (data is fresh)"
    if "Diagnose" in base and is_file_fresh(DIAG_SIDECAR):
        return f"{base} -- skip (data is fresh)"
    return base


def _detect_tools(query: str) -> list[str]:
    """Return deduplicated tool keys matched from the query, in canonical order."""
    q = query.lower()
    matched: set[str] = set()
    for pattern, key in TOOL_KEYWORDS:
        if _re.search(pattern, q):
            matched.add(key)
    for ci_pattern, ci_keys in _COMPOSITE_INTENTS:
        if _re.search(ci_pattern, q):
            matched.update(ci_keys)
    # Sort by canonical order (predict → diagnose → simulate → recommend → email)
    order = {k: i for i, k in enumerate(TOOL_DEPS)}
    return sorted(matched, key=lambda k: order.get(k, 99))


def _slot(s: str) -> int:
    """Map a step label (including '-- skip' variants) to its canonical slot."""
    if s in _STEP_ORDER:
        return _STEP_ORDER[s]
    # "Predict delivery delays -- skip (data is fresh)" → slot 0, etc.
    if "-- skip" in s:
        base = s.split(" -- skip")[0]
        if base in _STEP_ORDER:
            return _STEP_ORDER[base]
    return 99


def _merge_steps(tool_keys: list[str]) -> list[str]:
    """Merge dependency chains for all requested tools into an ordered, deduplicated plan.

    When multiple tools map to the same canonical slot (e.g. one says "Predict
    delivery delays" and another says "Skip prediction (data is fresh)"), the
    actual-run label wins because at least one tool genuinely needs that step.
    """
    slots: dict[int, str] = {}  # canonical position → label
    for key in tool_keys:
        for raw_step in TOOL_DEPS[key]:
            resolved = _resolve_freshness(raw_step)
            pos = _slot(resolved)
            existing = slots.get(pos)
            if existing is None:
                slots[pos] = resolved
            elif "-- skip" in existing and "-- skip" not in resolved:
                slots[pos] = resolved          # "run" beats "skip"
    return [slots[p] for p in sorted(slots)]


def build_action_plan(query: str) -> list[str] | None:
    """Build an ordered action plan from a user query.

    Returns:
        None       – conversational greeting; let the agent respond naturally
        []         – no tools recognised; caller should ask for clarification
        [steps…]   – ordered plan to confirm before executing
    """
    if _GREETING_RE.match(query.strip()):
        return None

    if _FULL_PIPELINE_RE.search(query):
        return _merge_steps(list(TOOL_DEPS.keys()))

    tool_keys = _detect_tools(query)
    if not tool_keys:
        return []

    return _merge_steps(tool_keys)


# ═══════════════════════════════════════════════════════════════════════════
# Confirmation / pending-query resolution
# ═══════════════════════════════════════════════════════════════════════════

CONFIRM_RE = _re.compile(
    r"^(yes|y|go|go ahead|proceed|sure|ok|okay|confirm|do it|run it|let'?s go|yep|yup)[\.\!\s]*$",
    _re.IGNORECASE,
)


def clarification_message() -> str:
    """Return the standard clarification prompt shown when no tools are matched."""
    return ("I'm not sure which analysis to run for that. "
            "Could you tell me more? For example:\n"
            "- **Predict delays** – check which orders may be late\n"
            "- **Diagnose patterns** – find root causes for delays\n"
            "- **Simulate scenarios** – test what-if changes\n"
            "- **Recommend actions** – get optimisation suggestions\n"
            "- **Email alerts** – draft customer notifications\n\n"
            "What would you like me to do?")


def format_plan_text(plan: list[str]) -> str:
    """Format an action-plan step list into a confirmation prompt."""
    lines = ["Here's my plan:"]
    for i, step in enumerate(plan, 1):
        lines.append(f"{i}. {step}")
    lines.append("\nShall I proceed?")
    return "\n".join(lines)


def resolve_confirmation(
    message: str,
    history: list[dict],
    pending_query: str,
    is_quick_action: bool,
) -> tuple[str, str, str | None]:
    """Decide how to handle the current user message w.r.t. pending queries.

    Returns (action, resolved_message, assistant_reply):
        action:
            "run"     – execute resolved_message through the agent
            "clarify" – show assistant_reply and stop
            "confirm" – show assistant_reply (plan) and stash resolved_message
            "greet"   – let the agent respond naturally (no plan needed)
        resolved_message:
            the query to run (for "run"), or the original message
        assistant_reply:
            text to show (for "clarify"/"confirm"), or None
    """
    is_confirm = bool(CONFIRM_RE.match(message))

    # --- Recover lost pending query from chat history ---
    if not pending_query and is_confirm and len(history) >= 2:
        last_asst = history[-1]
        if (last_asst.get("role") == "assistant"
                and "Shall I proceed?" in last_asst.get("content", "")):
            for prev in reversed(history[:-1]):
                if prev.get("role") == "user":
                    pending_query = prev["content"]
                    break

    # --- Resolve pending confirmation ---
    if pending_query and is_confirm:
        return "run", pending_query, None
    if pending_query:
        pending_query = ""  # user typed something else; discard pending

    # --- Quick actions skip the plan step ---
    if is_quick_action:
        return "run", message, None

    # --- Build action plan for new queries ---
    plan = build_action_plan(message)

    if plan is None:
        return "greet", message, None
    if len(plan) == 0:
        return "clarify", message, clarification_message()
    return "confirm", message, format_plan_text(plan)

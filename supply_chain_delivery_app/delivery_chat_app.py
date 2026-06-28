
"""
Supply Chain Delivery – Conversational Chat UI (Gradio).

Key logic:
    - Implements a two-column Gradio UI: left (chat, controls), right (output tabs)
    - Handles all agent orchestration, plan confirmation, and output streaming
    - Action plan confirmation: shows plan, waits for user confirmation, then executes
    - Supports quick-action buttons, file upload, and multi-tool queries
    - All outputs (chat, tables, markdown, files) are updated in a single handler yield
    - Sidecar freshness helpers ensure pipeline outputs are reused if fresh
    - Welcome/help message shown on load, after clear, and after each analysis
"""

import os
import sys
import hashlib
import time
from pathlib import Path
from uuid import uuid4

# Ensure this directory is importable
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(dotenv_path=find_dotenv(), override=True)

import pandas as pd
import gradio as gr
from agents import Runner, trace

from delivery_agents import (
    supply_chain_delivery_master_agent,
    pipeline_mcp,
)
from helpers.post_processing import (
    process_predict,
    process_simulate,
    process_diagnosis,
    process_recommendations,
    process_emails,
)
from helpers.logging_utils import setup_run_logger
from helpers.app_utils import (
    brief_args,
    build_freshness_system_msg,
    is_file_fresh,
    save_diagnosis_sidecar,
    resolve_confirmation,
    PREDICT_SIDECAR,
    DIAG_SIDECAR,
    DISPLAY_ROWS,
    knowledge_files as _knowledge_files,
    input_files as _input_files,
)

_APP_LOGGER, _LOG_PATH = setup_run_logger(_APP_DIR)
_APP_LOGGER.info("app.startup log_path=%s", _LOG_PATH)

# ---------------------------------------------------------------------------
# In-process response cache
# Key: normalized message | orders file hash | sidecar mtimes | model name.
# Invalidates automatically when input data or sidecars change.
# ---------------------------------------------------------------------------
_response_cache: dict[str, dict] = {}
_RESPONSE_CACHE_MAX_SIZE = 50


def _response_cache_key(message: str, orders_path, predict_sidecar: Path, diag_sidecar: Path) -> str:
    """Build a stable cache key for a chat request."""
    model = os.getenv("OPENAI_MODEL", "")
    orders_hash = "none"
    if orders_path:
        _op = Path(orders_path)
        if _op.is_file():
            orders_hash = hashlib.sha256(_op.read_bytes()).hexdigest()[:16]
    predict_mtime = str(int(predict_sidecar.stat().st_mtime)) if predict_sidecar.is_file() else "0"
    diag_mtime = str(int(diag_sidecar.stat().st_mtime)) if diag_sidecar.is_file() else "0"
    raw = f"{message.lower().strip()}|{orders_hash}|{predict_mtime}|{diag_mtime}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _usage_from_container(container):
    """Extract token usage dict from an SDK container object or dict."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get("usage")
    return getattr(container, "usage", None)


def _value_from_usage(usage, *names):
    """Return the first matching numeric field from a usage object/dict."""
    for name in names:
        if isinstance(usage, dict) and name in usage:
            return usage.get(name)
        if hasattr(usage, name):
            return getattr(usage, name)
    return None


def _extract_usage_from_event(event):
    """Best-effort extraction of model + usage from a raw_response_event."""
    # Common SDK shapes: event.data.response, event.response, event.data
    candidates = []
    data = getattr(event, "data", None)
    response = getattr(data, "response", None) if data is not None else None
    candidates.extend([response, data, getattr(event, "response", None), event])

    usage = None
    model = None
    for c in candidates:
        if c is None:
            continue
        usage = _usage_from_container(c)
        if usage is not None:
            model = c.get("model") if isinstance(c, dict) else getattr(c, "model", None)
            break

    if usage is None:
        return None

    prompt_tokens = _value_from_usage(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _value_from_usage(usage, "completion_tokens", "output_tokens")
    total_tokens = _value_from_usage(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "model": model or "unknown",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

# ---------------------------------------------------------------------------
# display row cap is now in helpers/app_utils.py (DISPLAY_ROWS)
# Sidecar freshness helpers: check if prediction/diagnosis outputs are fresh, save diagnosis
# (shared implementations in app_utils.py)
# Knowledge & input file defaults: from app_utils
# ---------------------------------------------------------------------------

# Quick-action presets (label, query text)
_QUICK_ACTIONS = [
    ("Predict Delays", "Predict which orders are getting delayed today"),
    ("Diagnose Patterns", "Provide delay patterns and root cause diagnosis"),
    ("Simulate What-If", "Simulate delays for stormy weather in East region"),
    ("Recommend Actions", "Recommend ways to optimize delivery timelines"),
    ("Email Alerts", "Generate customer email alerts for delayed orders"),
    ("Run Full Analysis",
     "1. Predict Orders Getting Delayed\n"
     "2. Provide Delay Patterns & Root Cause Diagnosis\n"
     "3. Simulate Delays for Stormy Weather in East Region\n"
     "4. Recommend Ways to Optimize Delivery Timelines\n"
     "5. Generate Customer Email Alerts for Delayed Orders"),
]

_QUICK_ACTION_TEXTS = {text for _, text in _QUICK_ACTIONS}


_WELCOME_MSG = ("I can help with:\n"
               "- **Predicting** which orders will be delayed\n"
               "- **Diagnosing** delay patterns and root causes\n"
               "- **Simulating** what-if scenarios\n"
               "- **Recommending** actions to reduce delays\n"
               "- **Generating** customer email alerts\n\n"
               "What would you like to do?")


# ---------------------------------------------------------------------------
# Main chat handler: manages plan confirmation, agent execution, and output streaming
# ---------------------------------------------------------------------------

async def chat_handler(message: str, history: list, orders_path, pending_query: str, tab_state: dict):
    """
    Manual chatbot handler with action-plan confirmation.

    Args:
        message:       user text from the textbox
        history:       list of {"role": ..., "content": ...} dicts (chatbot state)
        orders_path:   file path(s) from the File widget
        pending_query: stashed query awaiting confirmation (from gr.State)
        tab_state:     dict preserving previous tab outputs across runs

    Yields 16-tuple: (history, textbox, pending_state,
                      predict_md, predict_df, predict_csv,
                      simulate_md, simulate_df, simulate_csv,
                      diagnosis_md, diag_hr_df, diag_comp_df,
                      recommend_md, email_md, email_csv,
                      tab_state)
    """
    if not tab_state:
        tab_state = {}

    request_id = uuid4().hex[:12]
    run_start = time.perf_counter()
    mcp_tool_names = {"predict_delivery_delays", "get_delay_diagnosis", "simulate_order_delays"}
    pending_tool_calls: list[tuple[str, float, bool]] = []
    run_prompt_tokens = 0
    run_completion_tokens = 0
    run_total_tokens = 0
    usage_events = 0
    _produced_keys: set[str] = set()

    def _tabs():
        """Build the 12-element tab tuple from preserved state."""
        return (
            tab_state.get("predict_md", ""),
            tab_state.get("predict_df", pd.DataFrame()),
            tab_state.get("predict_csv"),
            tab_state.get("simulate_md", ""),
            tab_state.get("simulate_df", pd.DataFrame()),
            tab_state.get("simulate_csv"),
            tab_state.get("diagnosis_md", ""),
            tab_state.get("diag_hr_df", pd.DataFrame()),
            tab_state.get("diag_comp_df", pd.DataFrame()),
            tab_state.get("recommend_md", ""),
            tab_state.get("email_md", ""),
            tab_state.get("email_csv"),
        )

    _EMPTY = (history, "",
              "",  # clear pending
              *_tabs(),
              tab_state)

    # Normalise file input
    if isinstance(orders_path, list):
        orders_path = orders_path[0] if orders_path else None

    message = (message or "").strip()
    if not message:
        _APP_LOGGER.info("request.empty request_id=%s", request_id)
        yield _EMPTY
        return

    _APP_LOGGER.info(
        "request.received request_id=%s pending=%s has_orders_file=%s message_len=%s",
        request_id,
        bool(pending_query),
        bool(orders_path),
        len(message),
    )

    # Append user message
    history = history + [{"role": "user", "content": message}]

    # --- Resolve confirmation / plan in one step (logic in app_utils) ---
    is_quick_action = message in _QUICK_ACTION_TEXTS
    # history without current user msg (resolve_confirmation looks at prior turns)
    prior_history = history[:-1]
    action, message, reply = resolve_confirmation(
        message, prior_history, pending_query, is_quick_action,
    )
    _APP_LOGGER.info(
        "request.confirmation request_id=%s action=%s is_quick_action=%s",
        request_id,
        action,
        is_quick_action,
    )

    if action == "clarify":
        _APP_LOGGER.info("request.clarify request_id=%s", request_id)
        history = history + [{"role": "assistant", "content": reply}]
        yield (history, "", "", *_tabs(), tab_state)
        return

    if action == "confirm":
        _APP_LOGGER.info("request.awaiting_confirmation request_id=%s", request_id)
        history = history + [{"role": "assistant", "content": reply}]
        yield (history, "", message, *_tabs(), tab_state)  # stash query as pending
        return

    # --- Proceed with running the agent ---

    # Check response cache before doing any work
    _cache_key = _response_cache_key(message, orders_path, PREDICT_SIDECAR, DIAG_SIDECAR)
    if _cache_key in _response_cache:
        cached = _response_cache[_cache_key]
        _APP_LOGGER.info("run.cache_hit request_id=%s", request_id)
        tab_state.update(
            {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in cached.items()}
        )
        yield (
            history + [{"role": "assistant", "content": "*(Loaded from cache — data unchanged.)*\n\n" + _WELCOME_MSG}],
            "", "",
            *_tabs(),
            tab_state,
        )
        return

    # Build full query
    full_query = message
    if orders_path:
        full_query += f"\n\nThe input orders data is in the file at path: {orders_path}"

    full_query += build_freshness_system_msg()
    _APP_LOGGER.info(
        "run.started request_id=%s query_len=%s has_orders_file=%s",
        request_id,
        len(full_query),
        bool(orders_path),
    )

    # Initialise outputs
    predict_text = simulate_text = diagnosis_text = recommend_text = email_alert_text = ""
    predict_df = simulate_df = pd.DataFrame()
    diagnosis_high_risk_df = diagnosis_comparison_df = pd.DataFrame()
    predict_csv_path = simulate_csv_path = email_csv_path = None

    # Status lines — plain text, no emojis (issue #6)
    status_lines = []
    if orders_path:
        status_lines.append(f"Input file: {Path(orders_path).name}")
    else:
        status_lines.append("Note: No input file selected -- using default pipeline data.")
    status_lines.append("Starting analysis...\n")

    # Immediately show status in chat; show hourglass in empty tabs, preserve filled ones
    _PENDING = "<div style='color:#888;font-size:12px;padding:8px 0'>&#8987; Running analysis &mdash; results will appear here when ready...</div>"

    def _keep_or_pending(key: str) -> str:
        """Return previous tab content if it has real output, else hourglass."""
        val = tab_state.get(key, "")
        if val and "not run" not in val.lower() and "not fresh" not in val.lower():
            return val
        return _PENDING

    _running_tabs = (
        _keep_or_pending("predict_md"),
        tab_state.get("predict_df", pd.DataFrame()),
        tab_state.get("predict_csv"),
        _keep_or_pending("simulate_md"),
        tab_state.get("simulate_df", pd.DataFrame()),
        tab_state.get("simulate_csv"),
        _keep_or_pending("diagnosis_md"),
        tab_state.get("diag_hr_df", pd.DataFrame()),
        tab_state.get("diag_comp_df", pd.DataFrame()),
        _keep_or_pending("recommend_md"),
        _keep_or_pending("email_md"),
        tab_state.get("email_csv"),
    )
    yield (
        history + [{"role": "assistant", "content": "\n".join(status_lines)}],
        "",
        "",  # clear pending
        *_running_tabs,
        tab_state,
    )

    try:
        async with pipeline_mcp:
            with trace("Supply Chain Delivery Master"):
                result = Runner.run_streamed(supply_chain_delivery_master_agent, full_query)

                async for event in result.stream_events():
                    if event.type == "raw_response_event":
                        usage_payload = _extract_usage_from_event(event)
                        if usage_payload:
                            pt = usage_payload.get("prompt_tokens")
                            ct = usage_payload.get("completion_tokens")
                            tt = usage_payload.get("total_tokens")
                            run_prompt_tokens += int(pt or 0)
                            run_completion_tokens += int(ct or 0)
                            run_total_tokens += int(tt or 0)
                            usage_events += 1
                            _APP_LOGGER.info(
                                "llm.usage request_id=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                                request_id,
                                usage_payload.get("model", "unknown"),
                                pt,
                                ct,
                                tt,
                            )
                        continue
                    elif event.type == "agent_updated_stream_event":
                        _APP_LOGGER.info(
                            "agent.updated request_id=%s agent=%s",
                            request_id,
                            event.new_agent.name,
                        )
                        status_lines.append(f"  Agent: {event.new_agent.name}")
                    elif event.type == "run_item_stream_event":
                        item = event.item
                        raw = getattr(item, "raw_item", None)
                        if item.type == "tool_call_item":
                            tool_name = getattr(raw, "name", "")
                            args = getattr(raw, "arguments", None)
                            is_mcp_tool = tool_name in mcp_tool_names
                            pending_tool_calls.append((tool_name, time.perf_counter(), is_mcp_tool))
                            _APP_LOGGER.info(
                                "tool.call.started request_id=%s tool=%s args=%s",
                                request_id,
                                tool_name,
                                brief_args(args),
                            )
                            status_lines.append(f"    -- calling {tool_name}\n    {brief_args(args)}")
                        elif item.type == "tool_call_output_item":
                            tool_name = "unknown"
                            duration_ms = None
                            if pending_tool_calls:
                                tool_name, started_at, is_mcp_tool = pending_tool_calls.pop(0)
                                duration_ms = int((time.perf_counter() - started_at) * 1000)
                                if is_mcp_tool:
                                    _APP_LOGGER.info(
                                        "mcp.tool.call.completed request_id=%s tool=%s duration_ms=%s",
                                        request_id,
                                        tool_name,
                                        duration_ms,
                                    )
                            _APP_LOGGER.info(
                                "tool.call.completed request_id=%s tool=%s duration_ms=%s",
                                request_id,
                                tool_name,
                                duration_ms,
                            )
                            status_lines.append("    -- output received")

                    assistant_text = "\n".join(status_lines)
                    yield (
                        history + [{"role": "assistant", "content": assistant_text}],
                        "",
                        "",  # clear pending
                        *_running_tabs,
                        tab_state,
                    )

                # ---- Final output processing (deterministic, in post_processing.py) ----
                final_output = result.final_output
                if final_output is not None and not isinstance(final_output, str):
                    diagnosis_text = final_output.diagnosis_summary

                    predict_text, predict_df, predict_csv_path = process_predict(
                        final_output.predict_summary or "",
                        getattr(final_output, "predict_rows", None) or [],
                        _APP_DIR,
                        DISPLAY_ROWS,
                    )

                    simulate_text, simulate_df, simulate_csv_path = process_simulate(
                        final_output.simulate_summary or "",
                        getattr(final_output, "simulate_rows", None) or [],
                        _APP_DIR,
                    )

                    diagnosis_high_risk_df, diagnosis_comparison_df = process_diagnosis(
                        getattr(final_output, "diagnosis_high_risk_rows", None) or [],
                        getattr(final_output, "diagnosis_comparison_rows", None) or [],
                    )
                    if diagnosis_text:
                        save_diagnosis_sidecar(diagnosis_text)

                    recommend_text = process_recommendations(
                        final_output.recommendation_summary or "",
                        getattr(final_output, "recommendation_rows", None) or [],
                    )

                    email_alert_text, email_csv_path = process_emails(
                        final_output.email_alert_summary or "",
                        getattr(final_output, "email_alerts", None),
                        _APP_DIR,
                    )

                    _APP_LOGGER.info(
                        "run.outputs request_id=%s predict=%s diagnosis=%s simulate=%s recommend=%s email=%s",
                        request_id,
                        bool(predict_text),
                        bool(diagnosis_text),
                        bool(simulate_text),
                        bool(recommend_text),
                        bool(email_alert_text),
                    )

                # Final assistant message — plain text summary, no emojis (#6)
                summary_parts = ["\n---", "**Analysis complete.**\n"]
                if predict_text:
                    summary_parts.append("- Predictions       -->  Predict tab")
                if diagnosis_text:
                    summary_parts.append("- Diagnosis         -->  Diagnosis tab")
                if simulate_text:
                    summary_parts.append("- Simulation        -->  Simulation tab")
                if recommend_text:
                    summary_parts.append("- Recommendations   -->  Recommendation tab")
                if email_alert_text and "no email" not in email_alert_text.lower():
                    summary_parts.append("- Email Alerts      -->  Email tab")

                status_lines.extend(summary_parts)
                # Append welcome prompt for next interaction
                status_lines.append("\n" + _WELCOME_MSG)
                assistant_text = "\n".join(status_lines)

                # Persist new outputs into tab_state (skip "not run" messages)
                def _is_real(text: str) -> bool:
                    return bool(text) and "not run" not in text.lower()

                for _label, _txt in [
                    ("predict",   predict_text),
                    ("simulate",  simulate_text),
                    ("diagnosis", diagnosis_text),
                    ("recommend", recommend_text),
                    ("email",     email_alert_text),
                ]:
                    _APP_LOGGER.info(
                        "debug.agent_text request_id=%s agent=%s is_real=%s preview=%r",
                        request_id, _label, _is_real(_txt), _txt[:200],
                    )
                if _is_real(predict_text):
                    tab_state.update(predict_md=predict_text, predict_df=predict_df,
                                     predict_csv=predict_csv_path)
                    _produced_keys.update({"predict_md", "predict_df", "predict_csv"})
                if _is_real(simulate_text):
                    tab_state.update(simulate_md=simulate_text, simulate_df=simulate_df,
                                     simulate_csv=simulate_csv_path)
                    _produced_keys.update({"simulate_md", "simulate_df", "simulate_csv"})
                if _is_real(diagnosis_text):
                    tab_state.update(diagnosis_md=diagnosis_text,
                                     diag_hr_df=diagnosis_high_risk_df,
                                     diag_comp_df=diagnosis_comparison_df)
                    _produced_keys.update({"diagnosis_md", "diag_hr_df", "diag_comp_df"})
                if _is_real(recommend_text):
                    tab_state["recommend_md"] = recommend_text
                    _produced_keys.add("recommend_md")
                if _is_real(email_alert_text):
                    tab_state.update(email_md=email_alert_text, email_csv=email_csv_path)
                    _produced_keys.update({"email_md", "email_csv"})

                _APP_LOGGER.info(
                    "run.completed request_id=%s duration_ms=%s",
                    request_id,
                    int((time.perf_counter() - run_start) * 1000),
                )
                if usage_events > 0:
                    _APP_LOGGER.info(
                        "run.usage.summary request_id=%s usage_events=%s prompt_tokens_total=%s completion_tokens_total=%s total_tokens_total=%s",
                        request_id,
                        usage_events,
                        run_prompt_tokens,
                        run_completion_tokens,
                        run_total_tokens,
                    )
                else:
                    _APP_LOGGER.info(
                        "run.usage.summary request_id=%s usage_events=0 usage_source=unavailable",
                        request_id,
                    )

                # Store successful run outputs to response cache
                if _produced_keys:
                    _cached_entry: dict = {}
                    for _k in _produced_keys:
                        _v = tab_state.get(_k)
                        _cached_entry[_k] = _v.copy() if isinstance(_v, pd.DataFrame) else _v
                    if len(_response_cache) >= _RESPONSE_CACHE_MAX_SIZE:
                        _response_cache.clear()
                        _APP_LOGGER.info("run.response_cache_evicted request_id=%s", request_id)
                    _response_cache[_cache_key] = _cached_entry
                    _APP_LOGGER.info(
                        "run.response_cache_stored request_id=%s keys=%s",
                        request_id,
                        sorted(_produced_keys),
                    )

                yield (
                    history + [{"role": "assistant", "content": assistant_text}],
                    "",
                    "",  # clear pending
                    *_tabs(),
                    tab_state,
                )

    except Exception as e:
        _APP_LOGGER.exception(
            "run.failed request_id=%s duration_ms=%s error=%s",
            request_id,
            int((time.perf_counter() - run_start) * 1000),
            str(e),
        )
        error_msg = f"Error: {str(e)}"
        status_lines.append(error_msg)
        assistant_text = "\n".join(status_lines)
        yield (
            history + [{"role": "assistant", "content": assistant_text}],
            "",
            "",  # clear pending
            *_tabs(),
            tab_state,
        )


# ---------------------------------------------------------------------------
# Gradio UI: chatbot, textbox, quick actions, file upload, and output tabs
# ---------------------------------------------------------------------------

custom_css = """
    .content-textbox { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; font-size: 14px !important; line-height: 1.5 !important; }
    .content-textbox textarea { overflow-y: auto !important; min-height: 6em; max-height: 320px; }
    .input-textbox textarea { font-size: 13px !important; }
    .file-upload, .file-download { border: 1px solid #e0e0e0; border-radius: 4px; padding: 4px 8px; max-height: 70px; overflow: visible; font-size: 14px !important; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
    .file-upload *, .file-download * { font-size: inherit !important; }
    .table-container { min-height: 160px; }
    .output-table { font-size: 14px !important; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }
    .output-table table, .output-table th, .output-table td { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; font-size: 12px !important; line-height: 1.5 !important; }
    .summary-markdown { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px 16px; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; font-size: 14px !important; line-height: 1.6 !important; background: #fafafa; }
    .summary-markdown h3 { margin-top: 0.8em; margin-bottom: 0.4em; }
    .summary-markdown p { margin: 0.4em 0; }
    .summary-markdown ul { margin: 0.3em 0 0.3em 1.2em; padding: 0; }
    .summary-markdown li { margin: 0.15em 0; }
    .summary-label p { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; margin: 0 0 4px 0 !important; }
    .quick-btn { font-size: 11px !important; padding: 5px 8px !important; }
    .page-title h1 { font-size: 20px !important; font-weight: 700 !important; margin: 2px 0 0 0 !important; line-height: 1.2 !important; }
    .page-title { overflow: hidden !important; }
    .chatbot-box { height: 370px !important; max-height: 370px !important; overflow: hidden !important; }
    .chatbot-box > div { overflow-y: auto !important; }
    .chatbot-box .message-wrap, .chatbot-box .bot, .chatbot-box .user,
    .chatbot-box .message, .chatbot-box .markdown { font-size: 12px !important; line-height: 1.4 !important; }
    .chatbot-box, .chatbot-box * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important; font-size: 12px !important; line-height: 1.4 !important; }
    .chatbot-box p, .chatbot-box li, .chatbot-box span, .chatbot-box div { font-size: 12px !important; margin: 2px 0 !important; }
    .chatbot-box strong { font-weight: 700 !important; }
    .input-textbox { margin-top: 4px !important; width: 100% !important; }
    .input-textbox textarea { font-size: 14px !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="purple"), css=custom_css) as ui:
    ui.queue(default_concurrency_limit=5)
    gr.Markdown(
        "# Supply Chain Last-Mile Delivery Digital Twin - Monitoring & Optimization",
        elem_classes=["page-title"],
    )

    with gr.Row():
        # ---- LEFT COLUMN: Chat + Controls ----
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                type="messages",
                label="Agent Conversation",
                height=370,
                show_copy_button=True,
                elem_classes=["chatbot-box"],
                value=[{"role": "assistant", "content": _WELCOME_MSG}],
            )
            msg_input = gr.Textbox(
                placeholder="Ask about delivery delays, simulations, recommendations...",
                lines=3,
                show_label=False,
                container=False,
                elem_classes=["input-textbox"],
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            orders_file = gr.File(
                label="Upload Orders CSV / Excel",
                file_count="multiple",
                type="filepath",
                file_types=[".csv", ".xls", ".xlsx"],
                value=_input_files,
                elem_classes=["file-upload"],
            )
            gr.Markdown("**Quick Actions**", elem_classes=["summary-label"])
            qa_btns = []
            with gr.Row():
                for label, _ in _QUICK_ACTIONS[:3]:
                    qa_btns.append(gr.Button(label, size="sm", elem_classes=["quick-btn"]))
            with gr.Row():
                for label, _ in _QUICK_ACTIONS[3:]:
                    qa_btns.append(gr.Button(label, size="sm", elem_classes=["quick-btn"]))

        # ---- RIGHT COLUMN: Output Tabs ----
        with gr.Column(scale=2):
            with gr.Tab("Predict Agent"):
                gr.Markdown("Daily Delivery Delay Prediction Summary", elem_classes=["summary-label"])
                predict_out = gr.Markdown(value="", elem_classes=["summary-markdown"])
                with gr.Column(elem_classes=["table-container"]):
                    predict_table_out = gr.Dataframe(
                        label="Predicted Daily Delivery Delays with Severity and Reasoning",
                        value=pd.DataFrame(),
                        col_count=(0, "dynamic"),
                        row_count=(0, "dynamic"),
                        interactive=False,
                        elem_classes=["output-table"],
                    )
                    predict_csv_file = gr.File(
                        label="Download Prediction CSV",
                        type="filepath",
                        elem_classes=["file-download"],
                    )
            with gr.Tab("Diagnosis Agent"):
                gr.Markdown("Last 1 Month + Current Daily Data -- Delay Patterns Diagnosis Summary", elem_classes=["summary-label"])
                diagnose_out = gr.Markdown(value="", elem_classes=["summary-markdown"])
                diagnose_high_risk_table_out = gr.Dataframe(
                    label="High-Risk Delay Pattern Combinations (Today)",
                    value=pd.DataFrame(),
                    col_count=(0, "dynamic"),
                    row_count=(0, "dynamic"),
                    interactive=False,
                    elem_classes=["output-table"],
                )
                diagnose_comparison_table_out = gr.Dataframe(
                    label="Today vs Historical - Delay Rate Comparison",
                    value=pd.DataFrame(),
                    col_count=(0, "dynamic"),
                    row_count=(0, "dynamic"),
                    interactive=False,
                    elem_classes=["output-table"],
                )
            with gr.Tab("Simulation / What-if Agent"):
                gr.Markdown("Simulations / What-if Outcomes", elem_classes=["summary-label"])
                simulate_out = gr.Markdown(value="", elem_classes=["summary-markdown"])
                with gr.Column(elem_classes=["table-container"]):
                    simulate_table_out = gr.Dataframe(
                        label="Simulated Orders with Delay Hours",
                        value=pd.DataFrame(),
                        col_count=(0, "dynamic"),
                        row_count=(0, "dynamic"),
                        interactive=False,
                        elem_classes=["output-table"],
                    )
                    simulate_csv_file = gr.File(
                        label="Download Simulation CSV",
                        type="filepath",
                        elem_classes=["file-download"],
                    )
            with gr.Tab("Recommendation Agent"):
                gr.Markdown("Recommendations", elem_classes=["summary-label"])
                recommend_out = gr.Markdown(value="", elem_classes=["summary-markdown"])
                rag_sources_file = gr.File(
                    label="RAG Sources",
                    value=_knowledge_files,
                    file_count="multiple",
                    interactive=False,
                    elem_classes=["file-download"],
                )
            with gr.Tab("Email / Alerts Agent"):
                gr.Markdown("Email Alert Templates & Summary", elem_classes=["summary-label"])
                email_out = gr.Markdown(value="", elem_classes=["summary-markdown"])
                email_csv_file = gr.File(
                    label="Download Prediction CSV (with Email Templates)",
                    file_count="single",
                    interactive=False,
                    elem_classes=["file-download"],
                )

    # ---- Hidden state for pending confirmation & tab persistence ----
    pending_state = gr.State("")
    tab_state = gr.State({})

    # ---- Outputs list (16 elements) ----
    outputs = [
        chatbot,                             # 0  updated chat history
        msg_input,                           # 1  clear textbox after submit
        pending_state,                       # 2  pending query state
        predict_out,                         # 3
        predict_table_out,                   # 4
        predict_csv_file,                    # 5
        simulate_out,                        # 6
        simulate_table_out,                  # 7
        simulate_csv_file,                   # 8
        diagnose_out,                        # 9
        diagnose_high_risk_table_out,        # 10
        diagnose_comparison_table_out,       # 11
        recommend_out,                       # 12
        email_out,                           # 13
        email_csv_file,                      # 14
        tab_state,                           # 15  preserved tab outputs
    ]

    # ---- Wiring ----
    _handler_inputs = [msg_input, chatbot, orders_file, pending_state, tab_state]

    submit_btn.click(
        fn=chat_handler,
        inputs=_handler_inputs,
        outputs=outputs,
    )
    msg_input.submit(
        fn=chat_handler,
        inputs=_handler_inputs,
        outputs=outputs,
    )
    clear_btn.click(
        fn=lambda: ([{"role": "assistant", "content": _WELCOME_MSG}], "", "",
                     "", pd.DataFrame(), None,
                     "", pd.DataFrame(), None,
                     "", pd.DataFrame(), pd.DataFrame(),
                     "", "", None,
                     {}),
        outputs=outputs,
    )

    # Quick-action buttons: populate textbox then auto-submit
    for btn, (_, preset_text) in zip(qa_btns, _QUICK_ACTIONS):
        btn.click(fn=lambda t=preset_text: t, outputs=[msg_input]).then(
            fn=chat_handler,
            inputs=_handler_inputs,
            outputs=outputs,
        )


if __name__ == "__main__":
    ui.launch(
        inbrowser=True,
        allowed_paths=[str((_APP_DIR / "output").resolve())],
    )

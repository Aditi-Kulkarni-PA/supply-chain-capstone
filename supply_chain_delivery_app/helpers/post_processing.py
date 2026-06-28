"""
Deterministic post-processing helpers for MasterOutput → UI display.

Each function takes structured agent output (Pydantic rows / summary text)
and returns display-ready objects (DataFrames, markdown strings, file paths).
Keeping these out of the Gradio handler keeps the handler readable and
ensures counts, formatting, and data transformations are never LLM-dependent.
"""

import json as _json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Pipeline's processed directory (source of truth for predict outputs)
_PIPELINE_PROCESSED = (Path(__file__).resolve().parent.parent.parent
                       / "prediction_pipeline" / "data" / "processed")


# ---------------------------------------------------------------------------
# Predict post-processing
# ---------------------------------------------------------------------------

def _norm_id(s) -> str:
    """Normalize delivery_id to integer string (e.g. '10792.0' -> '10792')."""
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def process_predict(
    predict_summary: str,
    predict_rows: list,
    app_dir: Path,
    display_rows: int,
) -> tuple[str, pd.DataFrame, str]:
    """Return (predict_text, predict_df, predict_csv_path).

    Reads the prediction CSV + sidecar, merges LLM insights, and caps
    the displayed table to *display_rows*.
    """
    predict_text = predict_summary or ""
    if predict_text and not predict_text.lstrip().startswith("#"):
        predict_text = "### Cross-Dimensional Delay Insights\n\n" + predict_text

    predict_df = pd.DataFrame()
    predict_csv_path = None

    sidecar_dir = _PIPELINE_PROCESSED
    sidecar_path = sidecar_dir / "daily_delivery_delay_prediction_meta.json"
    sidecar: dict = {}
    if sidecar_path.is_file():
        sidecar = _json.loads(sidecar_path.read_text())

    fmt_stats = sidecar.get("formatted_stats", "")
    if fmt_stats:
        predict_text = predict_text.rstrip() + "\n\n" + fmt_stats

    csv_path = (
        sidecar.get("summary", {}).get("delayed_csv_path", "")
        or str(sidecar_dir / "daily_delivery_delay_prediction.csv")
    )
    rows = predict_rows or []

    if csv_path and Path(csv_path).is_file():
        predict_df = pd.read_csv(csv_path)
        predict_csv_path = csv_path
        if rows:
            enriched = {
                _norm_id(r.delivery_id): r.llm_insights
                for r in rows if r.llm_insights
            }
            if enriched:
                predict_df["_nid"] = predict_df["delivery_id"].astype(str).map(_norm_id)
                predict_df["llm_insights"] = predict_df["_nid"].map(enriched).fillna("")
                predict_df.drop(columns=["_nid"], inplace=True)
                predict_df.to_csv(csv_path, index=False)
        # Copy CSV and meta JSON to app/output so downstream tools find them there
        output_dir = app_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        app_csv = output_dir / "daily_delivery_delay_prediction.csv"
        predict_df.to_csv(str(app_csv), index=False)
        predict_csv_path = str(app_csv)
        if sidecar:
            app_meta = output_dir / "daily_delivery_delay_prediction_meta.json"
            app_meta.write_text(_json.dumps(sidecar, indent=2))
        predict_df = predict_df.head(display_rows)
    elif rows:
        predict_df = pd.DataFrame([r.model_dump() for r in rows])
        output_dir = app_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        predict_csv_path = str(output_dir / "daily_delivery_delay_prediction.csv")
        predict_df.to_csv(predict_csv_path, index=False)

    return predict_text, predict_df, predict_csv_path


# ---------------------------------------------------------------------------
# Simulation post-processing
# ---------------------------------------------------------------------------

def process_simulate(
    simulate_summary: str,
    simulate_rows: list,
    app_dir: Path,
) -> tuple[str, pd.DataFrame, str]:
    """Return (simulate_text, simulate_df, simulate_csv_path).

    Builds an accurate overview header from the actual rows (correct count,
    conditions, severity distribution) and appends the agent's qualitative
    narrative with any duplicate heading/counts stripped.
    """
    if not simulate_rows:
        return simulate_summary or "", pd.DataFrame(), None

    simulate_df = pd.DataFrame([r.model_dump() for r in simulate_rows])
    output_dir = app_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    simulate_csv_path = str(output_dir / "simulate_delays_latest.csv")
    simulate_df.to_csv(simulate_csv_path, index=False)

    # Build overview header from actual data
    n_sim = len(simulate_df)
    conds = []
    for col, label in [("weather_condition", "weather"), ("region", "region"), ("vehicle_type", "vehicle")]:
        if col in simulate_df.columns:
            vals = simulate_df[col].dropna().unique().tolist()
            if vals:
                conds.append(f"{label}: {', '.join(str(v) for v in vals)}")

    overview = f"### Delivery Delay Simulation Results\n\nAnalyzed **{n_sim}** simulated orders"
    if conds:
        overview += f" ({'; '.join(conds)})"
    overview += ".\n"

    if "simulated_severity" in simulate_df.columns:
        sev = simulate_df["simulated_severity"].value_counts()
        overview += "\n**Simulated Severity Distribution:**\n"
        for s, c in sev.items():
            overview += f"- {s}: {c} orders ({c / n_sim * 100:.1f}%)\n"

    # Append agent narrative (strip duplicate heading/count)
    narr = (simulate_summary or "").strip()
    narr = re.sub(r"^###?\s+.*\n+", "", narr).strip()
    narr = re.sub(r"Analyzed \d+ simulated.*?\.\s*", "", narr).strip()
    simulate_text = overview + ("\n" + narr if narr else "")

    return simulate_text, simulate_df, simulate_csv_path


# ---------------------------------------------------------------------------
# Diagnosis post-processing
# ---------------------------------------------------------------------------

def process_diagnosis(
    diagnosis_high_risk_rows: list,
    diagnosis_comparison_rows: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (diagnosis_high_risk_df, diagnosis_comparison_df)."""
    hr_df = (
        pd.DataFrame([r.model_dump() for r in diagnosis_high_risk_rows])
        if diagnosis_high_risk_rows else pd.DataFrame()
    )
    comp_df = (
        pd.DataFrame([r.model_dump() for r in diagnosis_comparison_rows])
        if diagnosis_comparison_rows else pd.DataFrame()
    )
    return hr_df, comp_df


# ---------------------------------------------------------------------------
# Recommendation post-processing
# ---------------------------------------------------------------------------

_CAT_HEADERS = {
    "quick-win": "**Quick-Win Actions (Immediate)**",
    "short-term": "**Short-term Actions (1-4 weeks)**",
    "long-term": "**Long-term Actions (1-3 months+)**",
}


def process_recommendations(
    recommendation_summary: str,
    recommendation_rows: list,
) -> str:
    """Return fully-rendered recommendation markdown built from structured rows.

    Groups actions by category with Data/SLA fields guaranteed present.
    Falls back to the agent's summary text if no structured rows exist.
    """
    if not recommendation_rows:
        return recommendation_summary or ""

    cats: dict[str, list] = {"quick-win": [], "short-term": [], "long-term": []}
    for r in recommendation_rows:
        key = r.category.lower().replace(" ", "-")
        if key in cats:
            cats[key].append(r)

    parts = ["### Delivery Optimization Recommendations\n"]
    narr = (recommendation_summary or "").strip()
    narr = re.sub(r"^###?\s+.*\n+", "", narr).strip()
    if narr:
        parts.append(narr + "\n")

    for ck in ["quick-win", "short-term", "long-term"]:
        actions = cats[ck]
        parts.append(f"\n{_CAT_HEADERS[ck]}\n")
        if not actions:
            parts.append("No actions in this category.\n")
        else:
            for a in actions:
                parts.append(f"- **{a.action}** ({a.dimension}): {a.action_desc}")
                parts.append(f"  - *Data*: {a.supporting_data}")
                parts.append(f"  - *SLA*: {a.sla_reference}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Email post-processing
# ---------------------------------------------------------------------------

def process_emails(
    email_alert_summary: str,
    email_alerts,              # Optional[EmailsList]
    app_dir: Path,
) -> tuple[str, str | None]:
    """Return (email_alert_text, email_csv_path)."""
    email_text = email_alert_summary or ""

    if email_alerts and getattr(email_alerts, "content", None):
        _SEP = '\n\n<hr style="border: none; border-top: 1px solid #ccc; margin: 16px 0;">\n\n'
        email_text = _SEP.join(
            f"**To:** {e.email_id}\n\n{e.email_content}"
            for e in email_alerts.content
        )
    if not email_text:
        email_text = "No delayed orders found; no email alerts generated."

    # Build an email-specific CSV: delayed rows that have email content written in
    email_csv_path = None
    predict_csv = app_dir / "output" / "daily_delivery_delay_prediction.csv"
    pipeline_csv = (app_dir.parent.parent / "prediction_pipeline"
                    / "data" / "processed" / "daily_delivery_delay_prediction.csv")
    src_csv = pipeline_csv if pipeline_csv.is_file() else (predict_csv if predict_csv.is_file() else None)
    if src_csv:
        try:
            df = pd.read_csv(src_csv)
            if "email_content" in df.columns:
                email_df = df[df["email_content"].notna() & (df["email_content"] != "")].copy()
            elif "predict_delay" in df.columns:
                email_df = df[df["predict_delay"] == 1].copy()
            else:
                email_df = df.copy()
            out_path = app_dir / "output" / "email_alerts.csv"
            email_df.to_csv(out_path, index=False)
            email_csv_path = str(out_path)
        except Exception:
            pass

    return email_text, email_csv_path

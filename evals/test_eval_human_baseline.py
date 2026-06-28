"""
Human baseline calibration — compare LLM-as-judge scores against human reviewer scores.

Reads evals/human_baseline/human_scores.xls and writes a Markdown report to
evals/reports/human_baseline_report_<timestamp>.md.

Run:
    uv run pytest evals/test_eval_human_baseline.py -v
"""

from datetime import datetime
from pathlib import Path

import pytest
import xlrd

BASELINE_XLS = Path(__file__).parent / "human_baseline" / "human_scores.xls"
REPORTS_DIR  = Path(__file__).parent / "reports"
DIMENSIONS   = ["relevance", "faithfulness", "safety"]


def _load() -> list[dict]:
    wb = xlrd.open_workbook(str(BASELINE_XLS))
    ws = wb.sheet_by_index(0)
    headers = [str(c).strip() for c in ws.row_values(0)]
    rows = []
    for i in range(1, ws.nrows):
        vals = ws.row_values(i)
        rows.append({headers[j]: str(vals[j]).strip() if vals[j] != "" else "" for j in range(len(headers))})
    return rows


def _scored_rows(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if all((r.get(f"human_{d}") or "").strip() for d in DIMENSIONS)
    ]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rows():
    assert BASELINE_XLS.exists(), f"XLS not found: {BASELINE_XLS}"
    return _load()


@pytest.fixture(scope="module")
def scored(rows):
    s = _scored_rows(rows)
    assert s, (
        f"No rows with human scores found in {BASELINE_XLS}. "
        "Fill in human_relevance, human_faithfulness, human_safety (1-5) for each row."
    )
    return s


# ── tests ─────────────────────────────────────────────────────────────────────

def test_all_rows_scored(rows):
    """All rows must have human scores filled in."""
    missing = [r["sample_id"] for r in rows if r not in _scored_rows(rows)]
    assert not missing, f"Missing human scores for: {', '.join(missing)}"


def test_human_scores_in_range(scored):
    """Human scores must be 1–5."""
    out_of_range = []
    for r in scored:
        for d in DIMENSIONS:
            val = float(r[f"human_{d}"])
            if not (1.0 <= val <= 5.0):
                out_of_range.append(f"{r['sample_id']}.{d}={val}")
    assert not out_of_range, f"Scores out of 1–5 range: {out_of_range}"


def test_no_large_divergence(scored):
    """Flag any agent where human mean differs from LLM mean by more than 1.5 points."""
    large_gaps = []
    for r in scored:
        llm_mean   = sum(float(r[f"llm_{d}"])   for d in DIMENSIONS) / 3
        human_mean = sum(float(r[f"human_{d}"]) for d in DIMENSIONS) / 3
        if abs(llm_mean - human_mean) > 1.5:
            large_gaps.append(f"{r['agent']}: LLM={llm_mean:.2f} Human={human_mean:.2f}")
    assert not large_gaps, f"Large LLM-human divergence (>1.5 mean pts): {large_gaps}"


# ── report writer ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def write_baseline_report():
    """Write the human baseline calibration report after all tests finish."""
    yield

    if not BASELINE_XLS.exists():
        return
    all_rows = _load()
    scored   = _scored_rows(all_rows)
    if not scored:
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = REPORTS_DIR / f"human_baseline_report_{timestamp}.md"

    lines = [
        "# Human Baseline Calibration Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Source:** `evals/human_baseline/human_scores.xls`  ",
        f"**Samples scored:** {len(scored)} / {len(all_rows)}\n",
        "---\n",
        "## Summary\n",
        "| Agent | LLM Mean | Human Mean | Diff | Aligned |",
        "|-------|----------|------------|------|---------|",
    ]
    for r in scored:
        llm_mean   = sum(float(r[f"llm_{d}"])   for d in DIMENSIONS) / 3
        human_mean = sum(float(r[f"human_{d}"]) for d in DIMENSIONS) / 3
        diff       = human_mean - llm_mean
        sign       = "+" if diff > 0 else ""
        aligned    = "Yes" if abs(diff) <= 1.0 else "No — review"
        lines.append(
            f"| {r['agent']} | {llm_mean:.2f} | {human_mean:.2f} | {sign}{diff:.2f} | {aligned} |"
        )
    lines.append("")

    # ── Relevance ─────────────────────────────────────────────────────────────
    lines += [
        "---\n",
        "## 1. Relevance\n",
        "_Does the output address the task and user need?_\n",
        "| Agent | LLM | Human | Delta |",
        "|-------|-----|-------|-------|",
    ]
    for r in scored:
        lv = float(r["llm_relevance"]); hv = float(r["human_relevance"])
        dv = hv - lv; ds = "+" if dv > 0 else ""
        lines.append(f"| {r['agent']} | {lv:.0f}/5 | {hv:.0f}/5 | {ds}{dv:.0f} |")
    lines.append("")

    # ── Faithfulness ──────────────────────────────────────────────────────────
    lines += [
        "---\n",
        "## 2. Faithfulness\n",
        "_Are all claims grounded in the data — no hallucinated numbers?_\n",
        "| Agent | LLM | Human | Delta |",
        "|-------|-----|-------|-------|",
    ]
    for r in scored:
        lv = float(r["llm_faithfulness"]); hv = float(r["human_faithfulness"])
        dv = hv - lv; ds = "+" if dv > 0 else ""
        lines.append(f"| {r['agent']} | {lv:.0f}/5 | {hv:.0f}/5 | {ds}{dv:.0f} |")
    lines.append("")

    # ── Safety ────────────────────────────────────────────────────────────────
    lines += [
        "---\n",
        "## 3. Safety\n",
        "_Absence of harmful, misleading, or inappropriate content._\n",
        "| Agent | LLM | Human | Delta |",
        "|-------|-----|-------|-------|",
    ]
    for r in scored:
        lv = float(r["llm_safety"]); hv = float(r["human_safety"])
        dv = hv - lv; ds = "+" if dv > 0 else ""
        lines.append(f"| {r['agent']} | {lv:.0f}/5 | {hv:.0f}/5 | {ds}{dv:.0f} |")
    lines.append("")

    # ── Per-sample reviewer notes ─────────────────────────────────────────────
    lines += ["---\n", "## 4. Reviewer Notes\n"]
    for r in scored:
        notes = (r.get("human_notes") or "").strip()
        if notes:
            lines.append(f"**{r['agent']}:** {notes}\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nHuman baseline report written: {report_path}")

"""
Compare LLM-as-judge scores against human baseline scores.

Usage:
    uv run python evals/llm_judge_eval.py

Fill in human_relevance, human_faithfulness, human_safety (1-5) in
evals/human_baseline/human_scores.xls, then run this script.
"""

import sys
import xlrd
from pathlib import Path

BASELINE_XLS = Path(__file__).parent / "human_baseline" / "human_scores.xls"
DIMENSIONS   = ["relevance", "faithfulness", "safety"]


def load_scores(xls_path: Path) -> tuple[list[dict], list[str]]:
    wb = xlrd.open_workbook(str(xls_path))
    ws = wb.sheet_by_index(0)
    headers = [str(c).strip() for c in ws.row_values(0)]
    rows, skipped = [], []
    for i in range(1, ws.nrows):
        vals = ws.row_values(i)
        row  = {headers[j]: str(vals[j]).strip() if vals[j] != "" else "" for j in range(len(headers))}
        if all((row.get(f"human_{d}") or "").strip() for d in DIMENSIONS):
            rows.append(row)
        else:
            skipped.append(row.get("sample_id", f"row{i}"))
    return rows, skipped


def main() -> None:
    if not BASELINE_XLS.exists():
        print(f"XLS not found: {BASELINE_XLS}")
        sys.exit(1)

    rows, skipped = load_scores(BASELINE_XLS)

    if skipped:
        print(f"Skipped (human scores missing): {', '.join(skipped)}")

    if not rows:
        print(
            "No rows have human scores filled in yet.\n"
            f"Open {BASELINE_XLS} in Excel and fill in human_relevance, "
            "human_faithfulness, human_safety (1-5) for each sample."
        )
        sys.exit(0)

    print(f"\nHuman baseline comparison — {len(rows)} scored sample(s)\n")
    print(f"{'Agent':<35} {'LLM Mean':>9}  {'Human Mean':>10}  {'Diff':>6}  {'Aligned'}")
    print("-" * 75)

    for r in rows:
        llm_mean   = sum(float(r[f"llm_{d}"])   for d in DIMENSIONS) / 3
        human_mean = sum(float(r[f"human_{d}"]) for d in DIMENSIONS) / 3
        diff       = human_mean - llm_mean
        sign       = "+" if diff > 0 else ""
        aligned    = "Yes" if abs(diff) <= 1.0 else "No — review"
        print(f"  {r['agent']:<33} {llm_mean:>9.2f}  {human_mean:>10.2f}  {sign}{diff:>5.2f}  {aligned}")

    print()
    for d in DIMENSIONS:
        llm_vals   = [float(r[f"llm_{d}"])   for r in rows]
        human_vals = [float(r[f"human_{d}"]) for r in rows]
        avg_diff   = sum(h - l for l, h in zip(llm_vals, human_vals)) / len(rows)
        sign = "+" if avg_diff > 0 else ""
        print(f"  {d.capitalize():<15}  avg diff: {sign}{avg_diff:.2f}")


if __name__ == "__main__":
    main()

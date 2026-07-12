"""
Pytest configuration — writes a Markdown summary report after each test run.

Report is saved to: tests/reports/smoke_test_report.md
"""
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path(__file__).parent / "reports"

STATUS = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "SKIP"}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write a Markdown summary of all test results."""
    REPORTS_DIR.mkdir(exist_ok=True)

    stats = terminalreporter.stats
    passed  = stats.get("passed",  [])
    failed  = stats.get("failed",  [])
    errors  = stats.get("error",   [])
    skipped = stats.get("skipped", [])
    total = len(passed) + len(failed) + len(errors) + len(skipped)

    if total == 0:
        # No tests actually ran (e.g. --collect-only, or a filter that matched
        # nothing) — leave the last real report alone instead of overwriting
        # it with a bogus all-zero result.
        terminalreporter.write_line(
            "\nNo tests ran — smoke_test_report.md left unchanged."
        )
        return

    overall = "PASSED" if not failed and not errors else "FAILED"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Smoke Test Report",
        "",
        f"**Date**: {now}  ",
        f"**Overall**: {overall}  ",
        f"**Total**: {total} | **Passed**: {len(passed)} | **Failed**: {len(failed)} | "
        f"**Errors**: {len(errors)} | **Skipped**: {len(skipped)}",
        "",
    ]

    def _section(label, reports, marker):
        if not reports:
            return
        lines.append(f"## {label}")
        lines.append("")
        for rep in reports:
            lines.append(f"- `[{marker}]` {rep.nodeid}")
            if marker in ("FAIL", "ERROR") and hasattr(rep, "longrepr"):
                excerpt = str(rep.longrepr).strip()[:600].replace("\n", "\n  ")
                lines.append(f"  ```\n  {excerpt}\n  ```")
        lines.append("")

    _section("Passed", passed, "PASS")
    _section("Failed", failed, "FAIL")
    _section("Errors", errors, "ERROR")
    _section("Skipped", skipped, "SKIP")

    report_path = REPORTS_DIR / "smoke_test_report.md"
    report_path.write_text("\n".join(lines))
    terminalreporter.write_line(f"\nReport: {report_path}")

from __future__ import annotations

from html import escape
from pathlib import Path


def write_html_report(path: Path, metrics: dict) -> None:
    rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Textual Feedback DPO Basic Pipeline Report</title>
</head>
<body>
  <h1>Basic Pipeline Report</h1>
  <table>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


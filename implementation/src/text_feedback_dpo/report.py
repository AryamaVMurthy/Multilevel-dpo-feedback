from __future__ import annotations

from html import escape
from pathlib import Path


def _bar_chart(title: str, values: dict[str, float]) -> str:
    if not values:
        return ""
    width = 640
    height = 220
    margin = 32
    maximum = max(float(value) for value in values.values()) or 1.0
    bar_width = max(12.0, (width - 2 * margin) / max(1, len(values)) - 8)
    bars = []
    for index, (label, value) in enumerate(values.items()):
        x = margin + index * (bar_width + 8)
        bar_height = (height - 2 * margin) * float(value) / maximum
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#2563eb"><title>{escape(str(label))}: '
            f'{escape(str(value))}</title></rect>'
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 10}" text-anchor="middle">'
            f'{escape(str(label))}</text>'
        )
    return (
        f'<section><h2>{escape(title)}</h2><svg data-chart="{escape(title)}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        + "".join(bars)
        + "</svg></section>"
    )


def _line_chart(title: str, history: list[dict]) -> str:
    points = [
        (float(row["step"]), float(row["loss"]))
        for row in history
        if "step" in row and "loss" in row
    ]
    if not points:
        return ""
    width = 640
    height = 220
    margin = 32
    max_x = max(point[0] for point in points) or 1.0
    max_y = max(point[1] for point in points) or 1.0
    path = []
    for x_value, y_value in points:
        x = margin + (width - 2 * margin) * x_value / max_x
        y = height - margin - (height - 2 * margin) * y_value / max_y
        path.append(f"{x:.1f},{y:.1f}")
    return (
        f'<section><h2>{escape(title)}</h2><svg data-chart="{escape(title)}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<polyline points="{" ".join(path)}" fill="none" stroke="#dc2626" '
        f'stroke-width="3"/></svg></section>'
    )


def write_html_report(
    path: Path,
    metrics: dict,
    *,
    training_history: list[dict] | None = None,
) -> None:
    rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    charts = _bar_chart(
        "success_by_attempt",
        {str(key): float(value) for key, value in metrics.get("success_by_attempt", {}).items()},
    )
    charts += _line_chart("training_loss", training_history or [])
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Textual Feedback DPO Experiment Report</title>
</head>
<body>
  <h1>Textual Feedback DPO Experiment Report</h1>
  <table>
    <tbody>
{rows}
    </tbody>
  </table>
  {charts}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_comparison_report(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("comparison report requires at least one method row")
    keys = sorted({key for row in rows for key in row})
    header = "".join(f"<th>{escape(str(key))}</th>" for key in keys)
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(key, '')))}</td>" for key in keys)
        + "</tr>"
        for row in rows
    )
    losses = {
        str(row["method"]): float(row["train_loss"])
        for row in rows
        if row.get("method") and row.get("train_loss") is not None
    }
    runtimes = {
        str(row["method"]): float(row["runtime"])
        for row in rows
        if row.get("method") and row.get("runtime") is not None
    }
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Textual Feedback DPO Method Comparison</title></head><body>
<h1>Textual Feedback DPO Method Comparison</h1>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
{_bar_chart("comparison_train_loss", losses)}
{_bar_chart("comparison_runtime_seconds", runtimes)}
</body></html>
"""
    path.write_text(html, encoding="utf-8")

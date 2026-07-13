from __future__ import annotations

import html
import json
from pathlib import Path


def write_html_report(path: Path, metrics: dict, artifacts: list[str]) -> None:
    rows = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(json.dumps(value, sort_keys=True))}</td></tr>" for key, value in sorted(metrics.items()))
    links = "".join(f"<li><a href='{html.escape(item, quote=True)}'>{html.escape(item)}</a></li>" for item in artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>SearchQA Research Report</title></head><body><h1>SearchQA Research Report</h1><table>" + rows + "</table><h2>Artifacts</h2><ul>" + links + "</ul></body></html>", encoding="utf-8")

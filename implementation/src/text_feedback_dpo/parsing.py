from __future__ import annotations

import re
from dataclasses import dataclass


class TrajectoryParseError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class Trajectory:
    raw: str
    plan: str
    thinks: list[str]
    reflect: str
    final: str

    @property
    def verification_present(self) -> bool:
        marker = "verification:"
        return marker in self.reflect.lower() and bool(
            self.reflect.lower().split(marker, 1)[1].strip()
        )


def _extract_tag(text: str, tag: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = text.find(open_tag)
    if start < 0:
        raise TrajectoryParseError(f"missing_{tag}", f"missing {open_tag}")
    end = text.find(close_tag, start + len(open_tag))
    if end < 0:
        raise TrajectoryParseError(f"missing_close_{tag}", f"missing {close_tag}")
    duplicate = text.find(open_tag, start + len(open_tag))
    if duplicate >= 0 and duplicate < end:
        raise TrajectoryParseError(f"nested_{tag}", f"nested {open_tag} is not allowed")
    content = text[start + len(open_tag) : end].strip()
    if not content:
        raise TrajectoryParseError(f"empty_{tag}", f"{open_tag} must not be empty")
    return content


def _extract_thinks(text: str) -> list[str]:
    matches = list(re.finditer(r"<think(?:\s+[^>]*)?>(.*?)</think>", text, flags=re.DOTALL))
    if not matches:
        raise TrajectoryParseError("missing_think", "missing <think> or <think branch=\"...\"> block")
    contents = [match.group(1).strip() for match in matches]
    if any(not content for content in contents):
        raise TrajectoryParseError("empty_think", "<think> blocks must not be empty")
    return contents


def parse_trajectory(text: str) -> Trajectory:
    if not text or not text.strip():
        raise TrajectoryParseError("empty_trajectory", "trajectory text is empty")

    if "<thinking" in text:
        raise TrajectoryParseError(
            "legacy_thinking_tag", "legacy <thinking> tag is invalid; use <think branch=\"A\">"
        )

    reflect_start = text.find("<reflect>")
    final_start = text.find("<final>")
    plan_start = text.find("<plan>")
    if reflect_start < 0:
        raise TrajectoryParseError("missing_reflect", "missing <reflect>")
    if final_start < 0:
        raise TrajectoryParseError("missing_final", "missing <final>")
    if plan_start < 0:
        raise TrajectoryParseError("missing_plan", "missing <plan>")
    if text.find("<final>", final_start + len("<final>")) >= 0:
        raise TrajectoryParseError("duplicate_final", "duplicate <final> block")

    think_matches = list(re.finditer(r"<think(?:\s+[^>]*)?>(.*?)</think>", text, flags=re.DOTALL))
    first_think = think_matches[0].start() if think_matches else -1
    if first_think < 0:
        _extract_thinks(text)
    if not (plan_start < first_think < reflect_start < final_start):
        raise TrajectoryParseError(
            "blocks_out_of_order",
            "trajectory blocks are out of order; expected plan, think, reflect, final",
        )

    trajectory = Trajectory(
        raw=text,
        plan=_extract_tag(text, "plan"),
        thinks=_extract_thinks(text),
        reflect=_extract_tag(text, "reflect"),
        final=_extract_tag(text, "final"),
    )
    if not trajectory.verification_present:
        raise TrajectoryParseError(
            "verification_missing", "<reflect> must contain a non-empty Verification section"
        )
    return trajectory

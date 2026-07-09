from __future__ import annotations

from dataclasses import dataclass


class TrajectoryParseError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class Trajectory:
    raw: str
    plan: str
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
    return text[start + len(open_tag) : end].strip()


def parse_trajectory(text: str) -> Trajectory:
    if not text or not text.strip():
        raise TrajectoryParseError("empty_trajectory", "trajectory text is empty")

    reflect_start = text.find("<reflect>")
    final_start = text.find("<final>")
    if reflect_start < 0:
        raise TrajectoryParseError("missing_reflect", "missing <reflect>")
    if final_start < 0:
        raise TrajectoryParseError("missing_final", "missing <final>")
    if final_start < reflect_start:
        raise TrajectoryParseError("final_before_reflect", "<final> appears before <reflect>")
    if text.find("<final>", final_start + len("<final>")) >= 0:
        raise TrajectoryParseError("duplicate_final", "duplicate <final> block")

    trajectory = Trajectory(
        raw=text,
        plan=_extract_tag(text, "plan"),
        reflect=_extract_tag(text, "reflect"),
        final=_extract_tag(text, "final"),
    )
    if not trajectory.verification_present:
        raise TrajectoryParseError(
            "verification_missing", "<reflect> must contain a non-empty Verification section"
        )
    return trajectory


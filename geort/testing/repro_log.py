"""Byte-level training-log comparison for reproducibility evidence."""

from pathlib import Path


def compare_logs(baseline_path: Path | str, candidate_path: Path | str) -> dict:
    """Return exact equality plus the first differing one-based text line."""
    baseline_lines = Path(baseline_path).read_text(encoding="utf-8").splitlines(keepends=True)
    candidate_lines = Path(candidate_path).read_text(encoding="utf-8").splitlines(keepends=True)
    for number, (baseline, candidate) in enumerate(
        zip(baseline_lines, candidate_lines),
        start=1,
    ):
        if baseline != candidate:
            return {
                "equal": False,
                "first_difference": {
                    "line": number,
                    "baseline": baseline,
                    "candidate": candidate,
                },
            }
    if len(baseline_lines) != len(candidate_lines):
        number = min(len(baseline_lines), len(candidate_lines)) + 1
        return {
            "equal": False,
            "first_difference": {
                "line": number,
                "baseline": baseline_lines[number - 1] if number <= len(baseline_lines) else None,
                "candidate": candidate_lines[number - 1] if number <= len(candidate_lines) else None,
            },
        }
    return {"equal": True, "first_difference": None}

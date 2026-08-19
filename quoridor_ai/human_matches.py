"""Validation and deterministic reporting for human Quoridor matches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quoridor_ai.core.engine import RULES_VERSION, State, apply_unchecked, legal_actions

FIELDS = (
    "checkpoint", "generation", "iteration", "rules_version", "encoding", "sims",
    "search", "human_player", "starting_player", "winner", "draw", "plies",
    "actions", "notes",
)


def validate_match(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one match record and return normalized replay details."""
    if not isinstance(record, dict):
        raise TypeError("record must be an object")
    unknown = set(record) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
    if record.get("rules_version", RULES_VERSION) != RULES_VERSION:
        raise ValueError("unsupported rules_version")
    actions = record.get("actions")
    if not isinstance(actions, list) or any(not isinstance(a, int) or isinstance(a, bool) for a in actions):
        raise ValueError("actions must be a list of integers")
    human = record.get("human_player")
    starting = record.get("starting_player", 0)
    if human not in (0, 1) or starting not in (0, 1):
        raise ValueError("human_player and starting_player must be 0 or 1")
    state = State(player=starting)
    for index, action in enumerate(actions):
        if action not in legal_actions(state):
            raise ValueError(f"illegal action at ply {index}: {action}")
        state = apply_unchecked(state, action)
    winner = state.winner
    declared_winner = record.get("winner")
    draw = record.get("draw", False)
    if declared_winner not in (None, 0, 1) or not isinstance(draw, bool):
        raise ValueError("winner must be 0, 1, or null and draw must be boolean")
    if record.get("plies", len(actions)) != len(actions):
        raise ValueError("plies does not match actions")
    if declared_winner != winner:
        raise ValueError("winner does not match replay")
    if winner is not None and draw:
        raise ValueError("terminal win cannot be a draw")
    return {"state": state, "winner": winner, "draw": draw, "plies": len(actions)}


def read_jsonl(path: str | Path, *, skip_truncated_final: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if skip_truncated_final and number == len(lines):
                continue
            raise ValueError(f"invalid JSON on line {number}") from None
        validate_match(record)
        records.append(record)
    return records


def build_report(records: list[dict[str, Any]]) -> dict[str, int]:
    """Build deterministic aggregate data from validated records."""
    results = [validate_match(record) for record in records]
    wins = sum(result["winner"] is not None for result in results)
    draws = sum(result["draw"] for result in results)
    return {"matches": len(results), "decisive": wins, "draws": draws,
            "plies": sum(result["plies"] for result in results)}


def report(records: list[dict[str, Any]], format: str = "json") -> str:
    """Render a stable JSON or Markdown report."""
    payload = build_report(records)
    if format == "json":
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if format == "markdown":
        return ("# Human matches\n\n"
                f"- Matches: {payload['matches']}\n"
                f"- Decisive: {payload['decisive']}\n"
                f"- Draws: {payload['draws']}\n"
                f"- Plies: {payload['plies']}\n")
    raise ValueError("format must be json or markdown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)
    print(report(read_jsonl(args.jsonl, skip_truncated_final=True), args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

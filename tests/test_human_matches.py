import json

import pytest

from quoridor_ai.core.engine import State, legal_actions
from quoridor_ai.human_matches import read_jsonl, report, validate_match


def _record(actions=None, **extra):
    result = {"human_player": 0, "starting_player": 0, "actions": actions or []}
    result.update(extra)
    return result


def test_validate_replays_official_legal_actions():
    action = legal_actions(State())[0]
    result = validate_match(_record([action], draw=True, plies=1))
    assert result["plies"] == 1 and result["winner"] is None


def test_validate_rejects_illegal_action():
    with pytest.raises(ValueError, match="illegal action"):
        validate_match(_record([999]))


def test_report_skips_truncated_final_jsonl_line(tmp_path):
    path = tmp_path / "matches.jsonl"
    path.write_text(json.dumps(_record([], draw=True)) + "\n{", encoding="utf-8")
    records = read_jsonl(path, skip_truncated_final=True)
    assert report(records) == '{"decisive":0,"draws":1,"matches":1,"plies":0}'
    assert report(records, "markdown") == (
        "# Human matches\n\n- Matches: 1\n- Decisive: 0\n- Draws: 1\n- Plies: 0\n")

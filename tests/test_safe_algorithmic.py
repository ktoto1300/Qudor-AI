import json

import numpy as np
import pytest

from quoridor_ai.az_selfplay import validate_gumbel_value_mixture
from quoridor_ai.core.encoding import MIRROR
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.opening_bank import load_opening_bank, mirror_line, select_opening
from quoridor_ai.td import alternating_td_lambda


def test_alternating_td_lambda_negates_the_next_players_value():
    targets = alternating_td_lambda([0.0, 0.5, -0.25], [0.0, 0.0], [False, True], lam=1.0)
    assert np.allclose(targets, [-1.0, 0.0])


def test_alternating_td_lambda_rejects_mismatched_sequences():
    with pytest.raises(ValueError, match="one more"):
        alternating_td_lambda([0.0], [0.0], [False])


def test_opening_bank_loads_validates_mirrors_and_selects_seeded(tmp_path):
    line = [legal_actions(State())[0]]
    path = tmp_path / "openings.json"
    path.write_text(json.dumps({"version": 1, "rules_version": 2, "lines": [line]}))
    bank = load_opening_bank(path)
    assert bank == (tuple(line),)
    assert mirror_line(line) == (int(MIRROR[line[0]]),)
    assert select_opening(bank, 7) == select_opening(bank, 7)
    assert apply_unchecked(State(), bank[0][0]).ply == 1


def test_opening_bank_rejects_illegal_action(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "rules_version": 2, "lines": [[0]]}))
    with pytest.raises(ValueError, match="illegal"):
        load_opening_bank(path)


def test_gumbel_value_mixture_modes_are_explicitly_validated():
    assert validate_gumbel_value_mixture("adapted") == "adapted"
    assert validate_gumbel_value_mixture("canonical") == "canonical"
    with pytest.raises(ValueError, match="gumbel_value_mixture"):
        validate_gumbel_value_mixture("unknown")

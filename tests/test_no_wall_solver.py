import pytest

from quoridor_ai.core.engine import State, legal_actions
from quoridor_ai.no_wall_solver import Outcome, solve


def test_rejects_positions_with_walls():
    with pytest.raises(ValueError, match="walls0=walls1=0"):
        solve(State(), 2)


def test_zero_wall_actions_are_legal():
    state = State(walls0=0, walls1=0)
    result = solve(state, 1)
    assert result.outcome is Outcome.DRAW
    assert set(result.optimal_actions) == set(legal_actions(state))


def test_horizon_is_draw():
    result = solve(State(walls0=0, walls1=0), 0)
    assert result.outcome is Outcome.DRAW
    assert result.optimal_actions == ()


def test_simple_terminal_win():
    result = solve(State(p0=0, p1=4, walls0=0, walls1=0, player=1), 2)
    assert result.outcome is Outcome.LOSS


def test_immediate_win_and_budget_exhaustion():
    state = State(p0=9, p1=40, walls0=0, walls1=0, player=0)
    won = solve(state, 1)
    assert won.outcome is Outcome.WIN and won.optimal_actions == (0,)
    assert won.pv == (0,) and won.as_dict()["node_budget"] == 100_000
    assert solve(state, 2, node_budget=1).outcome is Outcome.UNKNOWN

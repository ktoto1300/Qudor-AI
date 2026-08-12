"""The baseline bots are the project's only absolute yardstick, so they must not drift.

Every other strength number here is relative - candidate against champion, same lineage.
These bots supply the fixed point that makes a win rate mean the same thing at generation 15
and at generation 40. That only holds if their behaviour is pinned: a bot that silently got
stronger or weaker between runs would invalidate every comparison made against it, and would
do so invisibly, because the number it produces still looks perfectly reasonable.
"""
import numpy as np
import pytest
import torch

from quoridor_ai.baseline import BOTS, _score, greedy, play, rusher
from quoridor_ai.core.encoding import PLANES_BY_VERSION
from quoridor_ai.core.engine import (State, apply_unchecked, dist_to_goal, legal_actions,
                                     pawn_moves)
from quoridor_ai.model import PolicyValueNet

BOT_IDS = sorted(BOTS)


def _net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(12, 2, PLANES_BY_VERSION[3], True).eval()


def _rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("name", BOT_IDS)
def test_bot_only_ever_returns_a_legal_action(name):
    """Bots feed apply_unchecked, which validates nothing - an illegal move corrupts state."""
    bot, s, rng = BOTS[name], State(), _rng(4)
    for _ in range(120):
        if s.winner is not None or not legal_actions(s):
            break
        a = bot(s, rng)
        assert a in legal_actions(s), f"{name} returned illegal action {a} at ply {s.ply}"
        s = apply_unchecked(s, a)


def test_rusher_never_places_a_wall():
    """The floor of the scale is defined by *not* using walls; if it does, it is not the floor."""
    s, rng = State(), _rng(1)
    for _ in range(120):
        if s.winner is not None:
            break
        a = rusher(s, rng)
        assert a < 81, f"rusher placed a wall ({a})"
        s = apply_unchecked(s, a)
    assert (s.walls0, s.walls1) == (10, 10)


def test_rusher_always_shortens_its_own_path():
    """A shortest-path walker never stalls or steps backwards.

    A pawn jump can reduce BFS distance by two in one move, so exactly one is not a
    Quoridor invariant.
    """
    s, rng = State(), _rng(2)
    for _ in range(120):
        if s.winner is not None:
            break
        p = s.player
        before = dist_to_goal(s, p)
        s = apply_unchecked(s, rusher(s, rng))
        assert dist_to_goal(s, p) < before, f"distance went {before} -> {dist_to_goal(s, p)}"


def test_rusher_finishes_the_game_unobstructed():
    """Two shortest-path racers finish promptly on an empty board.

    Which colour wins is not fixed: random tie breaking and a legal pawn jump can let
    the second mover pass the first.
    """
    s, rng = State(), _rng(3)
    for _ in range(40):
        if s.winner is not None:
            break
        s = apply_unchecked(s, rusher(s, rng))
    assert s.winner in {0, 1}
    assert s.ply <= 16


def test_greedy_places_a_wall_when_it_pays():
    """A blocker that never blocks is just a slower rusher."""
    s, rng = State(), _rng(5)
    walls = 0
    while s.winner is None and s.ply < 220:
        a = greedy(s, rng)
        walls += a >= 81
        s = apply_unchecked(s, a)
    assert walls > 0


def test_greedy_takes_the_immediate_win():
    """One step from the goal with a wall available: a one-ply bot must not find that hard."""
    s = State(p0=9, p1=40)              # player 0 sits on row 1, one step from row 0
    assert greedy(s, _rng(6)) in {a for a in pawn_moves(s, 0) if a < 9}


def test_score_prefers_blocking_the_opponent():
    """The race term is the whole heuristic; if its sign is wrong the bot helps its opponent."""
    s = State()
    near = max(legal_actions(s), key=lambda a: _score(apply_unchecked(s, a), 0))
    after = apply_unchecked(s, near)
    assert _score(after, 0) >= _score(s, 0)


def test_a_wall_is_worth_less_than_a_step():
    """Otherwise the bot hoards walls to the end of the game and never uses one."""
    s = State()
    spent = State(p0=s.p0, p1=s.p1, walls0=9, walls1=10)
    assert _score(s, 0) - _score(spent, 0) < 1.0


@pytest.mark.parametrize("name", BOT_IDS)
def test_match_scores_every_game_exactly_once(name):
    r = play(_net(0), BOTS[name], torch.device("cpu"), games=6, sims=8, max_plies=30, seed=2)
    assert r["games"] == 6
    assert r["wins"] + r["draws"] + r["losses"] == 6
    assert r["win_rate"] == pytest.approx((r["wins"] + 0.5 * r["draws"]) / 6)


def test_match_splits_colours_evenly():
    """Player 0 moves first, a real edge in Quoridor; one-sided colours measure that edge."""
    r = play(_net(0), greedy, torch.device("cpu"), games=8, sims=8, max_plies=30, seed=1)
    assert r["win_rate"] == pytest.approx((r["win_rate_as_p0"] + r["win_rate_as_p1"]) / 2)


def test_match_is_reproducible():
    """The baseline is tracked across generations, so two runs of one seed must agree."""
    kw = dict(games=4, sims=8, max_plies=30, seed=9)
    a = play(_net(0), greedy, torch.device("cpu"), **kw)
    b = play(_net(0), greedy, torch.device("cpu"), **kw)
    assert (a["wins"], a["draws"], a["losses"]) == (b["wins"], b["draws"], b["losses"])


def test_ply_cap_is_scored_as_a_draw():
    r = play(_net(0), greedy, torch.device("cpu"), games=4, sims=8, max_plies=2, seed=5)
    assert (r["draws"], r["win_rate"]) == (4, 0.5)


def test_an_untrained_net_does_not_beat_greedy():
    """Calibration sanity: random weights must score badly, or the yardstick is broken."""
    r = play(_net(7), greedy, torch.device("cpu"), games=10, sims=8, max_plies=120, seed=3)
    assert r["win_rate"] <= 0.5, f"random weights scored {r['win_rate']} against greedy"

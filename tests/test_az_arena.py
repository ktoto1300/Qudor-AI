"""Gating decides which network survives, so the arena's bookkeeping has to be exact.

A colour-assignment or result-mapping bug here does not crash - it silently promotes the
wrong network, which looks like "training plateaued" many iterations later.
"""
import numpy as np
import pytest
import torch

from quoridor_ai.az_arena import _Duel, _play_batch, compare
from quoridor_ai.core.encoding import PLANES_BY_VERSION
from quoridor_ai.model import PolicyValueNet

MODES = [pytest.param(False, id="puct"), pytest.param(True, id="gumbel")]


def _net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(12, 2, PLANES_BY_VERSION[3], True).eval()


@pytest.mark.parametrize("gumbel", MODES)
def test_every_duel_is_scored_exactly_once(gumbel):
    a, b = _net(0), _net(1)
    r = compare(a, b, torch.device("cpu"), games=6, sims=12, max_plies=40, seed=3,
                gumbel=gumbel)
    assert r["games"] == 6
    assert r["wins"] + r["draws"] + r["losses"] == 6
    assert r["win_rate"] == pytest.approx((r["wins"] + 0.5 * r["draws"]) / 6)
    assert r["gumbel"] is gumbel


def test_colours_are_split_evenly():
    """If one net got the first move more often, the win rate would measure the opening."""
    duels = [_Duel(i % 2 == 1, i) for i in range(8)]
    assert sum(d.swap for d in duels) == 4


@pytest.mark.parametrize("gumbel", MODES)
def test_a_net_against_itself_scores_even(gumbel):
    """Identical players, colours swapped: any systematic bias here is a scoring bug."""
    net = _net(0)
    scores = _play_batch(net, net, torch.device("cpu"), 12, 12, 1.6, 0.6, 30, 7, 3, 3,
                         gumbel, 16)
    assert len(scores) == 12
    assert set(scores) <= {0.0, 0.5, 1.0}
    assert 0.2 <= sum(scores) / 12 <= 0.8, f"lopsided at {sum(scores) / 12}"


@pytest.mark.parametrize("gumbel", MODES)
def test_same_seed_reproduces_the_match(gumbel):
    """Gating compares runs across iterations; a non-reproducible arena makes that noise."""
    a, b = _net(0), _net(1)
    kw = dict(games=4, sims=12, max_plies=30, seed=11, gumbel=gumbel)
    first = compare(a, b, torch.device("cpu"), **kw)
    second = compare(a, b, torch.device("cpu"), **kw)
    assert first["win_rate"] == second["win_rate"]
    assert (first["wins"], first["draws"], first["losses"]) == \
           (second["wins"], second["draws"], second["losses"])


def test_ply_cap_is_scored_as_a_draw():
    """A cap of 2 plies cannot finish a game, so every duel must come back a draw."""
    net = _net(0)
    scores = _play_batch(net, net, torch.device("cpu"), 4, 8, 1.6, 0.6, 2, 5, 3, 3,
                         False, 16)
    assert scores == [0.5] * 4


def test_gumbel_gating_does_not_need_more_evaluations_than_puct():
    """The point of the Gumbel gate is a cheaper match, not a different-priced one."""
    import quoridor_ai.az_arena as arena
    net = _net(0)
    counts = {}
    real = arena._evaluate
    for gumbel in (False, True):
        total = [0]

        def counting(n_, nodes, *rest, _t=total, **kw):
            _t[0] += len(nodes)
            return real(n_, nodes, *rest, **kw)

        arena._evaluate = counting
        try:
            _play_batch(net, net, torch.device("cpu"), 4, 24, 1.6, 0.6, 24, 5, 3, 3,
                        gumbel, 16)
        finally:
            arena._evaluate = real
        counts[gumbel] = total[0]
    assert counts[True] <= counts[False] * 1.1, counts

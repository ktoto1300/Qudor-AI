"""Gating decides which network survives, so the arena's bookkeeping has to be exact.

A colour-assignment or result-mapping bug here does not crash - it silently promotes the
wrong network, which looks like "training plateaued" many iterations later.
"""
import pytest
import torch

from quoridor_ai.az_arena import _Duel, _play_batch, compare, elo_delta, summarise
from quoridor_ai.core.encoding import PLANES_BY_VERSION
from quoridor_ai.model import PolicyValueNet

MODES = [pytest.param(False, id="puct"), pytest.param(True, id="gumbel")]


def _net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(12, 2, PLANES_BY_VERSION[3], True).eval()


def _scores(*args, **kwargs):
    return [d.result for d in _play_batch(*args, **kwargs)]


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.parametrize("gumbel", MODES)
def test_a_net_against_itself_scores_even(gumbel):
    """Identical players, colours swapped: any systematic bias here is a scoring bug."""
    net = _net(0)
    scores = _scores(net, net, torch.device("cpu"), 12, 12, 1.6, 0.6, 30, 7, 3, 3,
                     gumbel, 16)
    assert len(scores) == 12
    assert set(scores) <= {0.0, 0.5, 1.0}
    assert 0.2 <= sum(scores) / 12 <= 0.8, f"lopsided at {sum(scores) / 12}"


@pytest.mark.integration
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


@pytest.mark.integration
def test_ply_cap_is_scored_as_a_draw():
    """A cap of 2 plies cannot finish a game, so every duel must come back a draw."""
    net = _net(0)
    scores = _scores(net, net, torch.device("cpu"), 4, 8, 1.6, 0.6, 2, 5, 3, 3,
                     False, 16)
    assert scores == [0.5] * 4


@pytest.mark.integration
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



# --- shared match summary -------------------------------------------------------
# Every harness reports through summarise(), so a change here moves the promotion gate,
# the baseline evals, minimax and the foreign arena at once.

class _Finished:
    """A duel that has already been played, for testing the summary alone."""

    def __init__(self, result, swap, ply=40):
        self.result = result
        self.swap = swap
        self.s = type('S', (), {'ply': ply})()


def test_summary_counts_draws_as_half_a_win():
    r = summarise([_Finished(1.0, False), _Finished(0.5, True),
                   _Finished(0.0, False), _Finished(0.5, True)])
    assert (r['wins'], r['draws'], r['losses']) == (1, 2, 1)
    assert r['win_rate'] == pytest.approx(0.5)


def test_summary_splits_the_win_rate_by_opening_colour():
    """Player 0 has a real first-move advantage; averaging it away hides a scoring bug."""
    r = summarise([_Finished(1.0, False), _Finished(1.0, False),
                   _Finished(0.0, True), _Finished(0.0, True)])
    assert r['win_rate_as_p0'] == pytest.approx(1.0)
    assert r['win_rate_as_p1'] == pytest.approx(0.0)
    assert r['win_rate'] == pytest.approx(0.5)


def test_summary_of_an_empty_match_is_zeros_not_a_division_by_zero():
    """games=0 is a legitimate degenerate call and carries no evidence of strength."""
    r = summarise([])
    assert r['games'] == 0
    assert r['win_rate'] == 0.0 and r['elo_delta'] == 0.0
    assert r['win_rate_as_p0'] == 0.0 and r['win_rate_as_p1'] == 0.0
    assert r['avg_plies'] == 0.0


def test_summary_refuses_an_unfinished_duel():
    """A None result would silently become a TypeError deep inside the arithmetic."""
    with pytest.raises(ValueError, match='finished'):
        summarise([_Finished(1.0, False), _Finished(None, True)])


def test_summary_passes_through_run_settings():
    r = summarise([_Finished(1.0, False)], sims=48, temperature=0.6, seed=7)
    assert (r['sims'], r['temperature'], r['seed']) == (48, 0.6, 7)


def test_compare_rejects_a_negative_game_count():
    with pytest.raises(ValueError, match='games'):
        compare(_net(0), _net(1), torch.device('cpu'), games=-1)


@pytest.mark.parametrize('win_rate,expected', [(0.5, 0.0), (1.0, 1600.0), (0.0, -1600.0)])
def test_elo_is_clamped_at_both_extremes(win_rate, expected):
    """A clean sweep has no finite Elo; +-1600 reads as 'past what this many games measure'."""
    assert elo_delta(win_rate) == pytest.approx(expected)


def test_elo_is_monotone_in_the_win_rate():
    rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    deltas = [elo_delta(r) for r in rates]
    assert deltas == sorted(deltas)

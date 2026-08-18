import numpy as np
import pytest
import torch

from quoridor_ai.az_selfplay import (Node, _completed_q, _considered, _evaluate,
                                     _improved_policy, inv_value_transform,
                                     search, selfplay, value_transform)
from quoridor_ai.core.encoding import PLANES_BY_VERSION
from quoridor_ai.core.engine import State
from quoridor_ai.model import PolicyValueNet


@pytest.mark.integration
def test_selfplay_reports_actual_terminal_ply_not_last_recorded_sample():
    """Full-search samples are intentionally sparse; game length must not inherit that sparsity."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    data, stats = selfplay(net, torch.device("cpu"), games=2, encoding=3,
                           sims=3, fast_sims=1, full_frac=0.5,
                           max_plies=12, seed=123)
    assert stats["games"] == 2
    assert stats["draws"] == 2
    assert stats["avg_plies"] == pytest.approx(12.0)
    assert data
    assert all(float(pi.sum()) == pytest.approx(1.0) for _, pi, _, _ in data)


def test_search_reports_a_won_position_as_a_loss_for_the_mover():
    """Finished boards used to return root value 0.0, which reads as a draw in the UI."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    won = State(p0=0, p1=40, player=1)          # player 0 already reached its goal row
    acts, probs, value = search(net, won, torch.device("cpu"), encoding=3, sims=4)
    assert acts == []
    assert probs.shape == (0,)
    assert value == -1.0, "the mover faces an already-lost board, not a draw"


@pytest.mark.integration
def test_batched_gumbel_is_deterministic_and_compatible():
    """The batched Gumbel variant is a speed option, not an equivalent clone of the
    per-visit loop (its leaf selections inside a round see the previous round's backups
    rather than the same round's). What it must guarantee is a stable, well-formed
    result, and identical output under the same seed."""

    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    dev = torch.device("cpu")
    opening = State()
    for sims, cap in ((24, 16), (64, 16)):
        a, p, v = search(net, opening, dev, encoding=3, sims=sims, gumbel=True,
                         gumbel_cap=cap, seed=7, batched=True)
        a2, p2, v2 = search(net, opening, dev, encoding=3, sims=sims, gumbel=True,
                            gumbel_cap=cap, seed=7, batched=True)
        a3, p3, v3 = search(net, opening, dev, encoding=3, sims=sims, gumbel=True,
                            gumbel_cap=cap, seed=7, batched=False)
        assert a == a2 and v == v2
        assert np.array_equal(p, p2)
        assert set(a) == set(range(81, 209)) | {67, 75, 77}
        assert np.isfinite(p).all() and 0.0 < p.sum() <= 1.0 + 1e-6
        assert a3 == a and p.shape == p3.shape and np.isfinite(v3)


def _root(net, sims_seed=0):
    node = Node(State())
    _evaluate(net, [node], torch.device("cpu"), 3, True)
    return node


def test_considered_is_a_power_of_two_the_budget_can_halve():
    """Sequential Halving needs m*log2(m) visits, so m must shrink with the budget."""
    assert _considered(30, 200, 16) == 16       # budget is ample, capped at 16
    assert _considered(30, 8, 16) == 4          # 4*2=8 fits, 8*3=24 does not
    assert _considered(3, 200, 16) == 2         # cannot consider more actions than exist
    assert _considered(30, 1, 16) == 1
    for k, b, cap in ((30, 8, 16), (30, 200, 16), (5, 13, 8), (64, 40, 32)):
        m = _considered(k, b, cap)
        assert m & (m - 1) == 0, "must be a power of two"
        assert m <= k and m <= cap


def test_completed_q_fills_unvisited_edges_with_the_mixed_value():
    """Never-visited edges get v_mix, not 0 - that is what makes the target dense."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    node = _root(net)
    node.value = 0.5
    q = _completed_q(node)
    assert np.allclose(q, 0.5), "with no visits every edge falls back to the node value"

    node.n[0] = 4
    node.w[0] = 4.0                              # that edge is worth +1
    node.total = 4
    q = _completed_q(node)
    assert q[0] == pytest.approx(1.0)
    unseen = q[1:]
    assert np.allclose(unseen, unseen[0]), "unvisited edges all share one v_mix"
    assert 0.5 < unseen[0] < 1.0, "v_mix moves from the node value toward observed Q"


def test_improved_policy_covers_every_legal_action():
    """The whole point of the completed policy: unsearched actions still get probability."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    node = _root(net)
    node.n[0] = 6
    node.w[0] = 6.0
    node.total = 6
    pi = _improved_policy(node)
    assert pi.shape == (len(node.acts),)
    assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)
    assert (pi > 0).all(), "no legal action may be assigned zero mass"
    assert int(np.argmax(pi)) == 0, "the edge search liked must lead"


@pytest.mark.integration
def test_gumbel_learns_a_dense_target_from_a_tiny_budget():
    """Gumbel's headline claim: usable targets at budgets where visit counts are noise."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    dev = torch.device("cpu")
    kw = dict(games=2, encoding=3, fast_sims=2, full_frac=1.0, max_plies=14, seed=7)
    gum, gs = selfplay(net, dev, sims=4, gumbel=True, **kw)
    plain, _ = selfplay(net, dev, sims=4, **kw)

    assert gs["games"] == 2 and gum
    for _, pi, z, q in gum:
        assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)
        assert (pi >= 0).all()
        assert -1.0 <= z <= 1.0 and -1.0 <= q <= 1.0
    support = lambda d: float(np.mean([(pi > 0).sum() for _, pi, _, _ in d]))
    assert support(gum) > 4 * support(plain), (
        f"gumbel {support(gum):.1f} vs visit counts {support(plain):.1f} actions per target")


@pytest.mark.integration
def test_gumbel_target_stays_inside_the_legal_action_set():
    """Mass outside the legal canonical actions would teach the net illegal moves."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    legal = set(_root(net).cacts)                # opening position, matching the first sample
    data, _ = selfplay(net, torch.device("cpu"), games=1, encoding=3, sims=6, fast_sims=2,
                       full_frac=1.0, max_plies=6, seed=11, gumbel=True)
    _, pi, _, _ = data[0]
    assert set(np.nonzero(pi)[0].tolist()) <= legal


@pytest.mark.integration
@pytest.mark.parametrize('gumbel', [False, True])
def test_rolling_game_pool_preserves_each_games_search(gumbel):
    """Pool size may change inference batching, never an individual tree's search."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    kwargs = dict(games=4, encoding=3, sims=4, fast_sims=2, full_frac=0.5,
                  max_plies=10, seed=19, gumbel=gumbel, gumbel_cap=4,
                  resign_skip=1.0)
    all_live, all_stats = selfplay(net, torch.device('cpu'), concurrent_games=4, **kwargs)
    rolling, rolling_stats = selfplay(net, torch.device('cpu'), concurrent_games=2, **kwargs)

    assert all_stats == rolling_stats
    assert len(all_live) == len(rolling)
    for expected, actual in zip(all_live, rolling, strict=True):
        assert np.array_equal(expected[0], actual[0])
        assert np.allclose(expected[1], actual[1])
        assert expected[2:] == pytest.approx(actual[2:], abs=1e-8)


def test_selfplay_rejects_invalid_pool_sizes():
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    with pytest.raises(ValueError, match='concurrent_games'):
        selfplay(net, torch.device('cpu'), games=2, concurrent_games=0)


@pytest.mark.integration
def test_game_shards_preserve_the_full_runs_rng_streams():
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    kwargs = dict(encoding=3, sims=4, fast_sims=2, full_frac=0.5, max_plies=8,
                  seed=23, gumbel=True, gumbel_cap=4, resign_skip=1.0)
    whole, whole_stats = selfplay(net, torch.device('cpu'), games=4, **kwargs)
    left, left_stats = selfplay(net, torch.device('cpu'), games=2, game_offset=0,
                                total_games=4, **kwargs)
    right, right_stats = selfplay(net, torch.device('cpu'), games=2, game_offset=2,
                                  total_games=4, **kwargs)
    assert whole_stats['games'] == left_stats['games'] + right_stats['games']
    combined = left + right
    assert len(whole) == len(combined)
    for expected, actual in zip(whole, combined, strict=True):
        assert np.array_equal(expected[0], actual[0])
        assert np.allclose(expected[1], actual[1])
        assert expected[2:] == pytest.approx(actual[2:], abs=1e-8)


def test_value_transform_and_inv_value_transform():
    """value_transform must be strictly monotonic, invert cleanly, and preserve [-1, 1] bounds."""
    vals = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    t_vals = value_transform(vals, alpha=1.0)
    assert np.all(np.diff(t_vals) > 0), "value_transform must be strictly monotonic"
    assert value_transform(1.0) == pytest.approx(1.0)
    assert value_transform(-1.0) == pytest.approx(-1.0)
    assert value_transform(0.0) == pytest.approx(0.0)
    assert (-1.0 <= t_vals).all() and (t_vals <= 1.0).all()

    inv_vals = inv_value_transform(t_vals, alpha=1.0)
    assert np.allclose(vals, inv_vals, atol=1e-6)

    # PyTorch Tensor support
    t_tensor = value_transform(torch.tensor([-1.0, 0.0, 0.5, 1.0]), alpha=1.0)
    assert torch.allclose(t_tensor, torch.tensor([-1.0, 0.0, value_transform(0.5), 1.0]))
    inv_tensor = inv_value_transform(t_tensor, alpha=1.0)
    assert torch.allclose(inv_tensor, torch.tensor([-1.0, 0.0, 0.5, 1.0]), atol=1e-6)


@pytest.mark.integration
def test_search_and_selfplay_default_settings_match_explicit_false():
    """Default parameters must produce bit-identical output to explicit legacy defaults."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    s = State()
    dev = torch.device('cpu')

    a1, p1, v1 = search(net, s, dev, encoding=3, sims=8, gumbel=True, seed=42)
    a2, p2, v2 = search(net, s, dev, encoding=3, sims=8, gumbel=True, seed=42,
                        tanh_value_transform=False, root_visit_compensation=False)
    assert a1 == a2 and v1 == v2
    assert np.array_equal(p1, p2)

    d1, s1 = selfplay(net, dev, games=2, encoding=3, sims=4, fast_sims=2,
                      full_frac=0.5, max_plies=8, seed=42, gumbel=True)
    d2, s2 = selfplay(net, dev, games=2, encoding=3, sims=4, fast_sims=2,
                      full_frac=0.5, max_plies=8, seed=42, gumbel=True,
                      tanh_value_transform=False, root_visit_compensation=False)
    assert s1 == s2
    assert len(d1) == len(d2)
    for sample1, sample2 in zip(d1, d2, strict=True):
        assert np.array_equal(sample1[0], sample2[0])
        assert np.array_equal(sample1[1], sample2[1])
        assert sample1[2:] == sample2[2:]


@pytest.mark.integration
def test_tanh_value_transform_search_and_selfplay():
    """Enabling tanh_value_transform preserves valid probabilities and bounded values in [-1, 1]."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    s = State()
    dev = torch.device('cpu')

    acts, probs, val = search(net, s, dev, encoding=3, sims=12, gumbel=True, seed=99,
                              tanh_value_transform=True)
    assert -1.0 <= val <= 1.0
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)
    assert (probs >= 0.0).all()
    assert len(acts) == len(probs)

    data, stats = selfplay(net, dev, games=2, encoding=3, sims=4, fast_sims=2,
                           full_frac=1.0, max_plies=8, seed=99, gumbel=True,
                           tanh_value_transform=True)
    assert stats['games'] == 2 and data
    for _, pi, z, q in data:
        assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= z <= 1.0 and -1.0 <= q <= 1.0


@pytest.mark.integration
def test_root_visit_compensation_search_and_selfplay():
    """Root visit compensation in Gumbel and PUCT modes behaves stably and deterministically."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3]).eval()
    s = State()
    dev = torch.device('cpu')

    # Gumbel search
    a1, p1, v1 = search(net, s, dev, encoding=3, sims=12, gumbel=True, seed=105,
                        root_visit_compensation=True)
    a2, p2, v2 = search(net, s, dev, encoding=3, sims=12, gumbel=True, seed=105,
                        root_visit_compensation=True)
    assert a1 == a2 and v1 == v2
    assert np.array_equal(p1, p2)
    assert float(p1.sum()) == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= v1 <= 1.0

    # PUCT search
    pa1, pp1, pv1 = search(net, s, dev, encoding=3, sims=12, gumbel=False, seed=105,
                           root_visit_compensation=True)
    assert pa1 and float(pp1.sum()) == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= pv1 <= 1.0

    # Selfplay
    data, stats = selfplay(net, dev, games=2, encoding=3, sims=4, fast_sims=2,
                           full_frac=0.5, max_plies=8, seed=105, gumbel=True,
                           root_visit_compensation=True)
    assert stats['games'] == 2 and data
    for _, pi, z, q in data:
        assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= z <= 1.0 and -1.0 <= q <= 1.0


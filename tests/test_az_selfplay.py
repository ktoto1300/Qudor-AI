import numpy as np
import pytest
import torch

from quoridor_ai.az_selfplay import (Node, _completed_q, _considered, _evaluate,
                                     _improved_policy, search, selfplay)
from quoridor_ai.core.encoding import PLANES_BY_VERSION
from quoridor_ai.core.engine import State
from quoridor_ai.model import PolicyValueNet


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


def test_gumbel_target_stays_inside_the_legal_action_set():
    """Mass outside the legal canonical actions would teach the net illegal moves."""
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    legal = set(_root(net).cacts)                # opening position, matching the first sample
    data, _ = selfplay(net, torch.device("cpu"), games=1, encoding=3, sims=6, fast_sims=2,
                       full_frac=1.0, max_plies=6, seed=11, gumbel=True)
    _, pi, _, _ = data[0]
    assert set(np.nonzero(pi)[0].tolist()) <= legal

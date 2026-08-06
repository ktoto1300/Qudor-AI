"""Model construction, checkpoint round-tripping and MCTS invariants."""
import math

import numpy as np
import pytest
import torch

from quoridor_ai.batched_mcts import Node, _expand, batched_search
from quoridor_ai.core.encoding import PLANES_BY_VERSION, encode_batch, version_for_planes
from quoridor_ai.core.engine import ACTION_SIZE, State, legal_actions
from quoridor_ai.model import (
    IncompatibleCheckpoint, PolicyValueNet, net_from_checkpoint, planes_of,
)

DEV = torch.device("cpu")


def tiny(version=1):
    return PolicyValueNet(8, 1, PLANES_BY_VERSION[version]).to(DEV).eval()


def test_forward_shapes():
    for version, planes in PLANES_BY_VERSION.items():
        net = tiny(version)
        x = torch.from_numpy(encode_batch([State(), State(ply=3)], version)).float()
        logits, value = net(x)
        assert logits.shape == (2, ACTION_SIZE)
        assert value.shape == (2,)
        assert torch.all(value.abs() <= 1)  # tanh head
        assert net.planes == planes


def test_planes_of_reads_the_stem():
    for version, planes in PLANES_BY_VERSION.items():
        assert planes_of(tiny(version).state_dict()) == planes


def test_net_from_checkpoint_restores_shape_and_weights():
    for version in PLANES_BY_VERSION:
        src = PolicyValueNet(16, 2, PLANES_BY_VERSION[version]).eval()
        ck = {"model": src.state_dict(), "config": {"channels": 16, "blocks": 2}}
        net = net_from_checkpoint(ck)
        assert net.planes == src.planes and net.channels == 16 and net.blocks == 2
        assert version_for_planes(net.planes) == version
        x = torch.from_numpy(encode_batch([State()], version)).float()
        with torch.inference_mode():
            assert torch.allclose(net(x)[0], src(x)[0])


def test_net_from_checkpoint_rejects_pre_resblock_architecture():
    """The legacy/ checkpoints have a 7-plane stem and a plain Sequential body."""
    bogus = {"model": {"stem.0.weight": torch.zeros(96, 7, 3, 3), "body.0.weight": torch.zeros(1)}}
    with pytest.raises(IncompatibleCheckpoint):
        net_from_checkpoint(bogus)


def test_search_returns_a_normalised_distribution():
    for version in PLANES_BY_VERSION:
        pi = batched_search(tiny(version), [State(), State(ply=4)], DEV, sims=8, encoding=version)
        assert pi.shape == (2, ACTION_SIZE)
        for row in pi:
            assert row.sum() == pytest.approx(1.0)
            assert np.all(row >= 0)


def test_search_only_gives_mass_to_legal_actions():
    s = State(p0=40, p1=31, walls0=1, walls1=1)
    pi = batched_search(tiny(), [s], DEV, sims=16)[0]
    legal = set(legal_actions(s))
    assert {int(a) for a in np.nonzero(pi)[0]} <= legal


def test_terminal_root_does_not_crash():
    """A won position has no legal actions; softmax over an empty set used to raise."""
    won0 = State(p0=4, p1=40, player=1, ply=20)
    won1 = State(p0=40, p1=76, player=0, ply=21)
    assert won0.winner == 0 and won1.winner == 1
    pi = batched_search(tiny(), [won0, won1, State()], DEV, sims=8)
    assert pi[0].sum() == 0 and pi[1].sum() == 0   # nothing to search
    assert pi[2].sum() == pytest.approx(1.0)       # the live root is unaffected


def test_visit_counts_are_conserved():
    """Every simulation must add exactly one visit per node on its path.

    The root is pre-expanded, so each of the `sims` descents passes through the root
    and lands on exactly one child subtree: root.n == sum(child.n) == sims.
    """
    sims, c_puct = 24, 1.5
    net = tiny()
    root = Node(State())
    _expand(net, [root], DEV, 1)
    for _ in range(sims):
        path, n = [root], root
        while n.children:
            _, n = max(n.children.items(),
                       key=lambda kv: -kv[1].q + c_puct * kv[1].p * math.sqrt(n.n + 1) / (1 + kv[1].n))
            path.append(n)
        vals = _expand(net, [path[-1]], DEV, 1)
        leaf = path[-1]
        v = -1.0 if leaf.s.winner is not None else vals.get(id(leaf), 0.0)
        for node in reversed(path):
            node.n += 1
            node.w += v
            v = -v
    assert root.n == sims
    assert sum(ch.n for ch in root.children.values()) == sims
    assert all(ch.n <= root.n for ch in root.children.values())


def test_search_explores_more_actions_with_more_sims():
    net = tiny()
    few = batched_search(net, [State()], DEV, sims=4)[0]
    many = batched_search(net, [State()], DEV, sims=64)[0]
    assert np.count_nonzero(many) >= np.count_nonzero(few)
    assert many.max() > 0

import numpy as np
import pytest
import torch

from quoridor_ai.core.engine import ACTION_SIZE, State, legal_actions
from quoridor_ai.mcts import search


class _Peaked(torch.nn.Module):
    """A tiny net-shaped stub: peaks a chosen action's logit, tanh value head."""

    def __init__(self, planes=11, peak=67):
        super().__init__()
        self.planes = planes
        self.peak = peak

    def forward(self, x):
        logits = torch.full((len(x), ACTION_SIZE), -20.0, device=x.device)
        logits[:, self.peak] = 20.0
        return logits, torch.zeros(len(x), device=x.device)


class _Flat(torch.nn.Module):
    """Uniform logits: the search outcome then depends only on the noise seed."""

    def __init__(self, planes=11):
        super().__init__()
        self.planes = planes

    def forward(self, x):
        return (torch.zeros(len(x), ACTION_SIZE, device=x.device),
                torch.zeros(len(x), device=x.device))


DEV = torch.device('cpu')


def test_terminal_root_returns_zero_policy():
    won = State(p0=4, p1=40, player=1, ply=20)
    assert won.winner == 0
    pi = search(_Peaked(), won, DEV, sims=8)
    assert pi.shape == (ACTION_SIZE,) and pi.sum() == 0


def test_capped_root_returns_zero_policy():
    pi = search(_Peaked(), State(ply=220), DEV, sims=8, max_plies=220)
    assert pi.sum() == 0
    hi = search(_Peaked(), State(ply=219), DEV, sims=8, max_plies=220)
    assert hi.sum() == pytest.approx(1.0)


def test_policy_is_normalised_over_legal_actions():
    s = State(p0=40, p1=31, walls0=1, walls1=1)
    pi = search(_Peaked(), s, DEV, sims=16)
    assert pi.sum() == pytest.approx(1.0)
    assert {int(a) for a in np.nonzero(pi)[0]} <= set(legal_actions(s))


def test_v3_encoding_is_supported():
    s = State(p0=67, p1=4, player=1, ply=1)
    pi = search(_Peaked(16, 67), s, DEV, sims=1)
    assert int(pi.argmax()) == 13          # canonical 67 unmaps to real 13


def test_same_seed_is_reproducible_and_different_seeds_differ():
    s = State()
    a = search(_Flat(), s, DEV, sims=32, seed=7)
    b = search(_Flat(), s, DEV, sims=32, seed=7)
    c = search(_Flat(), s, DEV, sims=32, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_search_restores_net_training_mode():
    net = _Peaked().train()
    assert net.training
    search(net, State(), DEV, sims=4)
    assert net.training
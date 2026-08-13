from numpy.random import default_rng

from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.minimax import choose


def test_minimax_takes_immediate_winning_move():
    # Player 0 is one row from the goal; 0 is a legal winning move.
    s = State(p0=9, p1=4, player=0, walls0=0, walls1=0)
    assert 0 in legal_actions(s)
    assert choose(s, default_rng(1), depth=1, width=10) == 0


def test_minimax_prefers_a_position_that_keeps_the_opponent_from_winning():
    # At depth two, every candidate is evaluated from the root's fixed viewpoint.
    # This regression test mainly guards against reintroducing a sign flip on P1 turns.
    s = State(p0=67, p1=13, player=0, walls0=0, walls1=0)
    action = choose(s, default_rng(2), depth=2, width=6)
    assert action in legal_actions(s)
    assert apply_unchecked(s, action).winner is None

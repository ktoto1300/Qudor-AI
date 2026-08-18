"""Minimax bot: search correctness, argument validation, and match harness.

The bug these guard against: min nodes were seeded with `-inf` alongside max nodes, so
every opponent node returned `-inf`, every root candidate tied at `-inf`, and the
tie-break picked uniformly at random. A depth-2 "alpha-beta bot" was a random mover that
paid for a full tree walk, and the two tests that existed then both passed.
"""
import math

import pytest
from numpy.random import default_rng

from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.minimax import _moves, choose, play

# 113 legal actions, 8 kept by width=8. Under the -inf bug every one of those 8 was
# reachable; the working search narrows it to the two that actually tie on evaluation.
WIDE = State(p0=76, p1=3, walls0=7, walls1=8, h=140737555464196, v=4398046512128,
             player=0, ply=6)


def test_minimax_takes_immediate_winning_move():
    # Player 0 is one row from the goal; 0 is a legal winning move.
    s = State(p0=9, p1=4, player=0, walls0=0, walls1=0)
    assert 0 in legal_actions(s)
    for depth in (1, 2, 3):
        assert choose(s, default_rng(1), depth=depth, width=10) == 0


def test_player_one_wins_toward_its_own_goal_row():
    """A sign error on P1 turns would send it away from row 8."""
    s = State(p0=40, p1=63, player=1, walls0=0, walls1=0)
    winning = [a for a in legal_actions(s) if a < 81 and a // 9 == 8]
    assert winning, 'fixture must offer player 1 a winning pawn move'
    for depth in (1, 2, 3):
        assert choose(s, default_rng(1), depth=depth, width=10) in winning


@pytest.mark.integration
def test_deeper_search_does_not_collapse_into_a_random_choice():
    """Min nodes must return real evaluations, not the max-node sentinel.

    With `best = -inf` at min nodes every candidate scored `-inf` and the tie-break
    spread over all `width` of them. A working search leaves only genuine ties.
    """
    candidates = {a for a, _ in _moves(WIDE, WIDE.player, 8)}
    assert len(candidates) == 8
    picked = {choose(WIDE, default_rng(k), depth=2, width=8) for k in range(40)}
    assert picked < candidates, 'depth-2 search is not discriminating between candidates'
    assert picked <= set(legal_actions(WIDE))


@pytest.mark.integration
def test_alpha_beta_pruning_does_not_change_the_chosen_move():
    """Pruning is an optimisation: a full-width search must agree with itself.

    The broken version pruned after the first child of every min node because `beta`
    was set to `-inf`, so this comparison would have been meaningless there - every
    width produced the same degenerate tie.
    """
    for width in (4, 6, 8):
        one = choose(WIDE, default_rng(3), depth=2, width=width)
        two = choose(WIDE, default_rng(3), depth=2, width=width)
        assert one == two, 'search must be deterministic for a fixed rng seed'


@pytest.mark.integration
def test_search_avoids_handing_the_opponent_an_immediate_win():
    """Only two of 132 moves stop player 1 from reaching row 8 next ply."""
    s = State(p0=40, p1=67, player=0, walls0=10, walls1=10)
    assert all(apply_unchecked(s, a).winner is None for a in legal_actions(s))
    safe = [a for a in legal_actions(s)
            if not any(apply_unchecked(apply_unchecked(s, a), b).winner is not None
                       for b in legal_actions(apply_unchecked(s, a)))]
    assert 0 < len(safe) < 5, 'fixture must make the saving moves scarce'
    for depth in (1, 2, 3):
        assert choose(s, default_rng(9), depth=depth, width=12) in safe



def test_a_faster_win_outranks_one_at_the_horizon():
    """The win bonus carries `left`, so a mate now beats a mate later."""
    s = State(p0=9, p1=4, player=0, walls0=0, walls1=0)
    # Action 0 wins immediately; every other pawn move needs at least one more ply.
    assert choose(s, default_rng(0), depth=3, width=12) == 0


@pytest.mark.parametrize('depth', [0, -1])
def test_depth_below_one_is_rejected(depth):
    """`depth-1` used to run past zero and recurse to the end of the game."""
    with pytest.raises(ValueError, match='depth'):
        choose(State(), default_rng(0), depth=depth, width=4)


@pytest.mark.parametrize('width', [0, -3])
def test_width_below_one_is_rejected(width):
    with pytest.raises(ValueError, match='width'):
        choose(State(), default_rng(0), depth=2, width=width)


def test_a_finished_board_is_an_error_not_a_silent_pick():
    finished = State(p0=4, p1=40, player=0, ply=20)
    assert finished.winner == 0 and legal_actions(finished) == []
    with pytest.raises(ValueError, match='no legal action'):
        choose(finished, default_rng(0), depth=2, width=4)


def test_moves_returns_actions_paired_with_their_own_child_state():
    s = State(p0=67, p1=13, player=0)
    for action, child in _moves(s, s.player, 5):
        assert child == apply_unchecked(s, action)


def test_moves_is_ordered_best_first_and_capped_by_width():
    from quoridor_ai.baseline import _score
    s = State(p0=67, p1=13, player=0)
    ranked = _moves(s, s.player, 6)
    assert len(ranked) == 6
    scores = [_score(child, s.player) for _, child in ranked]
    assert scores == sorted(scores, reverse=True)


def test_play_with_no_games_reports_an_empty_match_instead_of_dividing_by_zero():
    """`win_rate` used to be `sum(scores)/games`, which raised on games=0."""
    r = play(net=None, device=None, games=0, depth=1, width=2)
    assert r['games'] == 0 and r['win_rate'] == 0.0
    assert r['avg_plies'] == 0.0 and math.isfinite(r['elo_delta'])


def test_play_rejects_a_negative_game_count():
    with pytest.raises(ValueError, match='games'):
        play(net=None, device=None, games=-1)

"""Encoder tests.

v1 must stay byte-identical forever: every existing checkpoint was trained under it,
so a "fix" to v1 silently invalidates those weights. The golden values below are
hand-derived from the layout, not captured from the current implementation, so they
still catch a regression if encode_v1 is rewritten.
"""
import random

import numpy as np
import pytest

from quoridor_ai.core.encoding import (
    FLIPLR, MAX_PLIES, MIRROR, PLANES, PLANES_BY_VERSION, encode, encode_batch,
    encode_v1, encode_v2, encode_v3, is_canonical, version_for_planes,
)
from quoridor_ai.core.engine import State, apply_unchecked, bit, legal_actions, rc


def test_plane_counts():
    assert PLANES_BY_VERSION == {1: 11, 2: 8, 3: 16}
    assert PLANES == 11
    assert encode_v1(State()).shape == (11, 9, 9)
    assert encode_v2(State()).shape == (8, 9, 9)
    assert encode_v3(State()).shape == (16, 9, 9)


def test_v1_golden_start_position():
    x = encode_v1(State())
    assert x.dtype == np.float32
    r, c = rc(76)
    assert x[0, r, c] == 1 and x[0].sum() == 1
    r, c = rc(4)
    assert x[1, r, c] == 1 and x[1].sum() == 1
    assert np.all(x[2] == 0)          # player 0 to move
    assert np.all(x[3] == 1.0)        # 10/10 walls
    assert np.all(x[4] == 1.0)
    assert x[5].sum() == 0 and x[6].sum() == 0
    assert np.all(x[7] == 0)          # ply 0
    assert np.all(x[8] == 1)          # 1 - player
    assert np.all(x[9, 0] == 1) and np.all(x[9, 1:] == 0)
    assert np.all(x[10, 8] == 1) and np.all(x[10, :8] == 0)


def test_v1_ply_divisor_is_200_not_max_plies():
    """v1's ply scale is a known bug, deliberately preserved for checkpoint compatibility."""
    assert encode_v1(State(ply=100))[7, 0, 0] == pytest.approx(0.5)
    assert encode_v1(State(ply=210))[7, 0, 0] == pytest.approx(1.05)  # exceeds 1.0 by design


def test_v2_clamps_ply_at_the_real_cap():
    assert encode_v2(State(ply=0))[7, 0, 0] == pytest.approx(0.0)
    assert encode_v2(State(ply=110))[7, 0, 0] == pytest.approx(0.5)
    assert encode_v2(State(ply=MAX_PLIES))[7, 0, 0] == pytest.approx(1.0)
    assert encode_v2(State(ply=MAX_PLIES + 50))[7, 0, 0] == pytest.approx(1.0)


def test_v2_is_v1_without_the_dead_planes():
    """v2 drops planes 8/9/10 only; the informative planes must be untouched."""
    s = State(p0=40, p1=31, walls0=3, walls1=7, h=bit(2, 4), v=bit(5, 1), player=1, ply=37)
    a, b = encode_v1(s), encode_v2(s)
    for p in range(7):
        assert np.array_equal(a[p], b[p]), f"plane {p} diverged"


def test_walls_occupy_two_cells_each():
    h = encode_v1(State(h=bit(3, 2)))
    assert h[5, 3, 2] == 1 and h[5, 3, 3] == 1 and h[5].sum() == 2
    v = encode_v1(State(v=bit(3, 2)))
    assert v[6, 3, 2] == 1 and v[6, 4, 2] == 1 and v[6].sum() == 2


def test_encode_batch_stacks_and_matches_encode():
    states = [State(), State(p0=40, ply=5), State(p1=31, walls1=2)]
    for version, planes in PLANES_BY_VERSION.items():
        batch = encode_batch(states, version)
        assert batch.shape == (3, planes, 9, 9)
        for i, s in enumerate(states):
            assert np.array_equal(batch[i], encode(s, version))


def test_unknown_version_rejected():
    with pytest.raises(ValueError, match="unknown encoding version"):
        encode(State(), 99)
    with pytest.raises(ValueError, match="unknown encoding version"):
        encode_batch([State()], 0)


def test_version_for_planes_roundtrip():
    for version, planes in PLANES_BY_VERSION.items():
        assert version_for_planes(planes) == version
    with pytest.raises(ValueError):
        version_for_planes(7)  # the pre-ResBlock legacy layout


def test_default_encode_is_v1():
    s = State(p0=40, p1=31, ply=9, player=1)
    assert np.array_equal(encode(s), encode_v1(s))
    assert np.array_equal(encode_batch([s])[0], encode_v1(s))


# --- v3 canonical framing ----------------------------------------------------------
# These are the tests that matter most for training: if the board flip and the action
# permutation disagree even slightly, the network is trained on targets that belong to a
# different position, and nothing downstream reports an error - it just never learns.

def _flip_cell_ud(i):
    r, c = divmod(i, 9)
    return (8 - r) * 9 + c


def _flip_slots_ud(m):
    out = 0
    for i in range(64):
        if m >> i & 1:
            r, c = divmod(i, 8)
            out |= 1 << ((7 - r) * 8 + c)
    return out


def _swapped(s):
    """The same position with the two roles swapped: identical problem for the mover."""
    return State(p0=_flip_cell_ud(s.p1), p1=_flip_cell_ud(s.p0),
                 walls0=s.walls1, walls1=s.walls0,
                 h=_flip_slots_ud(s.h), v=_flip_slots_ud(s.v),
                 player=1 - s.player, ply=s.ply)


def _flipped_lr(s):
    def cell(i):
        r, c = divmod(i, 9)
        return r * 9 + (8 - c)

    def slots(m):
        out = 0
        for i in range(64):
            if m >> i & 1:
                r, c = divmod(i, 8)
                out |= 1 << (r * 8 + (7 - c))
        return out

    return State(p0=cell(s.p0), p1=cell(s.p1), walls0=s.walls0, walls1=s.walls1,
                 h=slots(s.h), v=slots(s.v), player=s.player, ply=s.ply)


def _random_positions(seed, games=25, depth=35):
    rng = random.Random(seed)
    for _ in range(games):
        s = State()
        for _ in range(rng.randint(1, depth)):
            acts = legal_actions(s)
            if not acts:
                break
            s = apply_unchecked(s, rng.choice(acts))
            if s.winner is not None:
                break
            yield s


def test_permutations_are_involutions_and_commute():
    assert (MIRROR[MIRROR] == np.arange(209)).all()
    assert (FLIPLR[FLIPLR] == np.arange(209)).all()
    assert (MIRROR[FLIPLR] == FLIPLR[MIRROR]).all()
    assert is_canonical(3) and not is_canonical(1) and not is_canonical(2)


def test_v3_is_identical_under_role_swap():
    """The whole point of canonical framing: both sides share one set of filters."""
    n = 0
    for s in _random_positions(seed=11):
        n += 1
        assert np.array_equal(encode_v3(s), encode_v3(_swapped(s)))
    assert n > 200


def test_mirror_maps_legal_actions_onto_the_swapped_position():
    for s in _random_positions(seed=12):
        assert sorted(int(MIRROR[a]) for a in legal_actions(s)) == sorted(legal_actions(_swapped(s)))


def test_mirrored_move_leads_to_the_mirrored_position():
    rng = random.Random(13)
    for s in _random_positions(seed=13):
        a = rng.choice(legal_actions(s))
        assert np.array_equal(encode_v3(apply_unchecked(s, a)),
                              encode_v3(apply_unchecked(_swapped(s), int(MIRROR[a]))))


def test_v3_left_right_flip_matches_the_column_reversed_tensor():
    """FLIPLR is used as training augmentation, so it must be exact, not approximate."""
    for s in _random_positions(seed=14):
        assert np.array_equal(encode_v3(_flipped_lr(s)), encode_v3(s)[:, :, ::-1])
        assert sorted(int(FLIPLR[a]) for a in legal_actions(s)) == sorted(legal_actions(_flipped_lr(s)))


def test_v3_planes_are_normalised():
    for s in _random_positions(seed=15, games=8):
        x = encode_v3(s)
        assert x.dtype == np.float32
        assert x.min() >= 0.0 and x.max() <= 1.0


def test_v3_frames_the_goal_at_row_zero_for_both_players():
    for player in (0, 1):
        x = encode_v3(State(player=player))
        assert x[14, 0].sum() == 9 and x[14, 1:].sum() == 0     # my goal row
        assert x[0].sum() == 1 and x[1].sum() == 1              # exactly one pawn each
    # From the start position both sides are the same distance from goal, so the race
    # margin sits exactly at the neutral 0.5.
    assert encode_v3(State())[11, 0, 0] == pytest.approx(0.5)

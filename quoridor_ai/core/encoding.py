"""Board -> tensor encoding, versioned.

v1 (11 planes) is the original layout and is kept byte-for-byte identical so every
existing checkpoint keeps working. It has three known flaws:
  * plane 8 is 1-player, fully redundant with plane 2;
  * planes 9 and 10 are constants (row 0 / row 8), so they carry no information;
  * plane 7 is ply/200, which exceeds 1.0 whenever a game runs past ply 200 while
    the training cap is max_plies=220.

v2 (8 planes) drops the three dead planes and normalises ply by the real cap.

v3 (16 planes) is the layout for serious training. On top of v2 it adds the features
a Quoridor net otherwise has to derive from raw walls over many layers:
  * per-side BFS distance-to-goal fields and the pawn's own distance,
  * the distance difference (who is winning the race),
  * an explicit "wall slot is occupied" plane,
  * side-to-move relative framing so the net always sees "me" vs "opponent".
Everything is normalised to roughly [0,1] and the whole board is flipped when player 1
is to move, so one set of filters serves both sides.
"""
import numpy as np
from .engine import State, rc, dist_field, _UNREACH

MAX_PLIES = 220
PLANES_BY_VERSION = {1: 11, 2: 8, 3: 16}
DEFAULT_VERSION = 1
PLANES = PLANES_BY_VERSION[DEFAULT_VERSION]  # kept for callers that import PLANES directly
BEST_VERSION = 3  # what new training runs should use


def _walls(x, s, hi, vi):
    for i in range(64):
        r, c = divmod(i, 8)
        if s.h >> i & 1: x[hi, r, c] = x[hi, r, c + 1] = 1
        if s.v >> i & 1: x[vi, r, c] = x[vi, r + 1, c] = 1


def encode_v1(s: State):
    x = np.zeros((11, 9, 9), np.float32)
    r, c = rc(s.p0); x[0, r, c] = 1
    r, c = rc(s.p1); x[1, r, c] = 1
    x[2].fill(s.player); x[3].fill(s.walls0 / 10); x[4].fill(s.walls1 / 10)
    _walls(x, s, 5, 6)
    x[7].fill(s.ply / 200); x[8].fill(1 - s.player); x[9, 0] = 1; x[10, 8] = 1
    return x


def encode_v2(s: State):
    x = np.zeros((8, 9, 9), np.float32)
    r, c = rc(s.p0); x[0, r, c] = 1
    r, c = rc(s.p1); x[1, r, c] = 1
    x[2].fill(s.player); x[3].fill(s.walls0 / 10); x[4].fill(s.walls1 / 10)
    _walls(x, s, 5, 6)
    x[7].fill(min(1.0, s.ply / MAX_PLIES))
    return x


_DNORM = 24.0  # longest sane Quoridor path; distances are clipped to this before scaling


def _dist_plane(field):
    """81 BFS distances -> a 9x9 plane in [0,1], with unreachable pinned at 1.0."""
    d = np.asarray(field, np.float32).reshape(9, 9)
    d[d >= _UNREACH] = _DNORM
    return np.minimum(d, _DNORM) / _DNORM


def _slot_blocks(mask, out):
    """Paint each set 8x8 slot bit as a 2x2 block on a 9x9 plane.

    A wall in slot (r,c) sits *between* rows r and r+1, so painting only row r makes the
    plane asymmetric under a vertical flip: slot rows mirror as r -> 7-r while cell rows
    mirror as r -> 8-r. No single-cell embedding of the 8x8 slot grid into 9x9 can be
    flip-equivariant; the two-row footprint {r, r+1} is the smallest one that is, since
    {7-r, 8-r} is exactly the mirror of {r, r+1}. v3 relies on this to share filters
    between the two sides.
    """
    for i in range(64):
        if mask >> i & 1:
            r, c = divmod(i, 8)
            out[r, c] = out[r, c + 1] = out[r + 1, c] = out[r + 1, c + 1] = 1


def encode_v3(s: State):
    """16 planes, framed from the side to move.

    The board is mirrored vertically for player 1 so "my goal" is always row 0. That
    halves what the net must learn: without it every tactic has to be memorised twice,
    once per direction of travel. Actions must be mirrored to match - see MIRROR.
    """
    me_is_1 = s.player == 1
    mine, theirs = (s.p1, s.p0) if me_is_1 else (s.p0, s.p1)
    my_walls, their_walls = (s.walls1, s.walls0) if me_is_1 else (s.walls0, s.walls1)

    # --- spatial planes, still in real board coordinates ---
    sp = np.zeros((7, 9, 9), np.float32)
    r, c = rc(mine); sp[0, r, c] = 1
    r, c = rc(theirs); sp[1, r, c] = 1
    _slot_blocks(s.h, sp[2])
    _slot_blocks(s.v, sp[3])
    _slot_blocks(s.h | s.v, sp[4])
    sp[5] = _dist_plane(dist_field(s, 8 if me_is_1 else 0))   # my distance-to-goal field
    sp[6] = _dist_plane(dist_field(s, 0 if me_is_1 else 8))   # theirs
    my_d = float(sp[5].reshape(-1)[mine])
    their_d = float(sp[6].reshape(-1)[theirs])
    if me_is_1:
        sp = sp[:, ::-1, :]        # into canonical frame: my goal becomes row 0

    x = np.zeros((16, 9, 9), np.float32)
    x[:7] = sp
    x[7].fill(my_walls / 10); x[8].fill(their_walls / 10)
    x[9].fill(my_d); x[10].fill(their_d)
    x[11].fill(np.clip(0.5 + (their_d - my_d) * 2.0, 0.0, 1.0))   # who wins the race
    x[12].fill(min(1.0, s.ply / MAX_PLIES))
    x[13].fill(1.0)                # bias plane: lets convs feel the board edge
    x[14, 0] = 1                   # my goal row, canonical
    x[15].fill(1.0 if my_walls else 0.0)
    return x


def _build_mirror():
    """Action permutation matching v3's vertical board flip: cell (r,c) -> (8-r,c).

    A horizontal wall in slot row r separates rows r and r+1; after the flip those
    become rows 8-r and 7-r, i.e. slot row 7-r. Vertical walls span the same two rows,
    so both wall families map (r,c) -> (7-r,c). The permutation is its own inverse,
    which is what lets the same table serve encoding and decoding.
    """
    m = np.empty(209, np.int64)
    for a in range(81):
        r, c = divmod(a, 9); m[a] = (8 - r) * 9 + c
    for base in (81, 145):
        for i in range(64):
            r, c = divmod(i, 8); m[base + i] = base + (7 - r) * 8 + c
    return m


MIRROR = _build_mirror()
assert (MIRROR[MIRROR] == np.arange(209)).all(), "mirror must be an involution"


def _build_fliplr():
    """Action permutation for the left-right board flip: cell (r,c) -> (r,8-c).

    This is a true symmetry of Quoridor - the rules never distinguish left from right - and
    it is orthogonal to MIRROR, so it doubles the training data for free. A wall in slot
    (r,c) spans columns c and c+1, which flip to 8-c and 7-c, i.e. slot column 7-c; the same
    mapping serves both orientations, and it is again its own inverse.
    """
    m = np.empty(209, np.int64)
    for a in range(81):
        r, c = divmod(a, 9); m[a] = r * 9 + (8 - c)
    for base in (81, 145):
        for i in range(64):
            r, c = divmod(i, 8); m[base + i] = base + r * 8 + (7 - c)
    return m


FLIPLR = _build_fliplr()
assert (FLIPLR[FLIPLR] == np.arange(209)).all(), "left-right flip must be an involution"


def canonical_actions(actions, player: int):
    """Map real action indices into the v3 canonical frame (identity for player 0)."""
    return [int(MIRROR[a]) for a in actions] if player == 1 else list(actions)


def decanonical_action(action: int, player: int) -> int:
    """Map a canonical-frame action back to a real one."""
    return int(MIRROR[action]) if player == 1 else int(action)


def is_canonical(version: int) -> bool:
    """True when the encoder mirrors the board for player 1, so actions need mirroring too."""
    return version >= 3


_ENCODERS = {1: encode_v1, 2: encode_v2, 3: encode_v3}


def encode(s: State, version: int = DEFAULT_VERSION):
    try:
        return _ENCODERS[version](s)
    except KeyError:
        raise ValueError(f"unknown encoding version {version!r}; known: {sorted(_ENCODERS)}") from None


def encode_batch(states, version: int = DEFAULT_VERSION):
    enc = _ENCODERS.get(version)
    if enc is None:
        raise ValueError(f"unknown encoding version {version!r}; known: {sorted(_ENCODERS)}")
    return np.stack([enc(s) for s in states])


def version_for_planes(planes: int) -> int:
    """Map a network's input-channel count back to the encoder that produced it."""
    for v, n in PLANES_BY_VERSION.items():
        if n == planes:
            return v
    raise ValueError(f"no encoding version produces {planes} planes")


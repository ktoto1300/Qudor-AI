r"""Bridge for pavlosdais/Quoridor ("SP Quoridor"), a C alpha-beta engine.

The engine is a compiled C program that speaks the QTP (Quoridor Text Protocol)
over stdin/stdout. This bridge launches it as a subprocess and translates between
our JSON-lines protocol and QTP. All of the engine's QTP chatter is consumed here;
only JSON ever reaches our stdout.

======================================================================
QTP commands used (see engine's src/commands.c and README.md)
======================================================================
  playmove <color> <vertex>            move <color>'s pawn to <vertex>
  playwall <color> <vertex> <h|v>      place a <color> wall at <vertex>
  genmove  <color>                     engine picks and PLAYS a move for <color>
  clear_board                          reset pawns to start, clear walls
  quit                                 exit
<color> is "white" or "black" (the parser also accepts w/b). Every command
answers with a single line "= ..." on success or "? ..." on failure, followed
by a blank line. genmove answers "= <VERTEX>" (pawn) or "= <VERTEX> <h|v>"
(wall) with an UPPERCASE column letter (the parser lowercases its input, but the
output is printed with 'A'+col). We read until the first '='/'?' line.

======================================================================
Full-position setup vs incremental play  ->  WE ONLY GET INCREMENTAL PLAY
======================================================================
The engine has NO command to load an arbitrary position: pawns can only move via
`playmove` (legal 1-2 cell steps), walls only via `playwall`, and there is no way
to set an arbitrary walls-remaining count. So the engine only supports incremental
play from the fixed start position.

Our arena, however, sends a FULL board state every turn (never a move history) and
it runs many games concurrently, interleaving their `state` messages through this
one bridge process. A single incremental session therefore cannot work.

Design: one engine subprocess PER GAME. Each session keeps a "shadow" copy of the
exact position its engine currently holds internally (after our last genmove, the
engine has already applied its own move, so the shadow == engine's internal state).
When a `state` message M arrives we:
  1. Find the session whose shadow T differs from M by exactly ONE move of the
     side to move in T (that is the opponent/net move played since our last turn),
     or whose shadow already equals M (our engine's very first move, before the
     opponent has replied). We verify the candidate by replaying it on a copy of T
     and comparing full fingerprints, so a false match is impossible.
  2. Apply that one opponent move to the engine (playmove/playwall), leaving it at M.
  3. `genmove` for our colour; the engine plays and prints its move; we decode it.
If no session matches (a brand-new game, an evicted session, or a desync) we spawn
a fresh engine and RECONSTRUCT M from the start: walk both pawns to their squares
on the still-empty board (BFS, two colour orderings tried), then place every wall
(any order is legal once pawns sit on their real squares, because a subset of a
legal wall set keeps every path open). Walls are attributed 10-w0 to white and the
rest to black so walls-remaining ends at (w0, w1); which physical wall belongs to
which colour is irrelevant to the engine (the wall matrix stores no owner, only the
per-player counts and the wall positions matter). A brand-new game's first state is
at ply 0 or 1, so its reconstruction is trivially 0 or 1 pawn step.
Sessions are capped (LRU eviction) so a long match cannot leak processes.

======================================================================
Colour <-> our player
======================================================================
Our p0 starts at the bottom (cell 76, row 8) and wins at row 0; engine WHITE starts
at i=0 (bottom) and wins at i=boardsize-1 (top). So  p0 == white.
Our p1 starts at the top (cell 4, row 0) and wins at row 8; engine BLACK starts at
i=boardsize-1 (top) and wins at i=0.               So  p1 == black.
The engine does not enforce turn order (colour is an explicit argument), so the fact
that our p0 moves first while the README says black moves first does not matter.

======================================================================
Coordinate / wall mapping (exact)
======================================================================
Engine rows: i=0 is the BOTTOM row, i increases upward, i=8 is the TOP.
Our rows:    row 0 is the TOP, row 8 is the BOTTOM.
So the two representations are a vertical reflection:  our_row = 8 - i,  i = 8 - our_row.
Columns are identical:  our_col = j (both have column 0 = 'a' = LEFT).
A QTP vertex is "<letter><number>" with letter = 'a'+j (column) and number = i+1 (row).

Pawn cell  (our cell = row*9+col):
    our -> vertex:  i = 8-row, letter = 'a'+col, number = i+1
    vertex -> our:  j = letter-'a', i = number-1, cell = (8-i)*9 + j
    e.g. cell 76 (p0 start, row8 col4) -> "e1";  cell 4 (p1 start, row0 col4) -> "e9".

Walls: the engine places a wall at a VERTEX + orientation. A horizontal 'b' wall at
engine [i][j] blocks the edge between engine rows i-1 and i over columns j,j+1; a
vertical 'r' wall at [i][j] blocks the edge between columns j,j+1 over engine rows
i-1,i. Our h slot (r,c) is on the line between rows r,r+1 over cols c,c+1; our v slot
(r,c) is between cols c,c+1 over rows r,r+1. Both orientations map with the SAME
transform as pawns:
    our slot (r,c) -> vertex:  i = 8-r, letter = 'a'+c, number = i+1, orient = h|v
    vertex+orient -> our slot: r = 8-(number-1), c = letter-'a'
    action = 81 + r*8+c  (horizontal)   or   145 + r*8+c  (vertical)
Engine valid wall range i in 1..8, j in 0..7 maps exactly onto our r,c in 0..7.

======================================================================
Build
======================================================================
Binary: bin/ipquoridor (Linux) or bin/ipquoridor.exe (Windows), per the makefile
(EXEC_NAME=ipquoridor). If it is missing we run `make final` (then plain `make` as a
fallback) in the engine's own directory. THIS MACHINE HAS NO C COMPILER, so the build
fails here and every move then falls back to a legal move (logged to stderr); on the
Linux tournament server `make` succeeds and the real engine plays. The engine uses a
fixed wall-clock search budget (6 s per move for a 9x9 board, from generate_moves.h),
well under the arena's 300 s move timeout.

======================================================================
Verified here (no compiler): python -m ruff check --select F,E9,B, ast.parse, and the
`--selftest` asserts below (coordinate round-trips + engine-output decode + move diff).
MUST be verified on the Linux server: that `make` builds bin/ipquoridor; that the
engine actually launches and answers QTP; that a full game plays through with correct
win/lose results; and that reconstruction (the rare fallback path) is exercised.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quoridor_ai.paths import bots_dir, repo_root      # noqa: E402

ROOT = bots_dir() / 'pavlosdais_Quoridor'

_REPO_PY = repo_root()
if str(_REPO_PY) not in sys.path:
    sys.path.insert(0, str(_REPO_PY))
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions  # noqa: E402

# How many concurrent per-game engine subprocesses to keep alive at once. The arena
# advances every live game one ply per round, so up to `--games` sessions can be live;
# this cap only evicts (and later reconstructs) once a match is unusually wide.
MAX_SESSIONS = 256

RNG = random.Random(0)
SESSIONS: list[EngineSession] = []


# --------------------------------------------------------------------------- #
# Coordinate helpers (pure; exercised by --selftest)                          #
# --------------------------------------------------------------------------- #
def cell_to_vertex(cell: int) -> str:
    """Our pawn cell (row*9+col) -> QTP vertex string, e.g. 76 -> 'e1'."""
    r, c = divmod(cell, 9)
    return f"{chr(97 + c)}{(8 - r) + 1}"


def vertex_to_cell(vertex: str) -> int:
    """QTP vertex -> our pawn cell. Column letter is case-insensitive."""
    c = ord(vertex[0].lower()) - 97
    i = int(vertex[1:]) - 1
    return (8 - i) * 9 + c


def slot_to_vertex(slot: int) -> str:
    """Our 8x8 wall slot (r*8+c) -> QTP vertex string (orientation carried separately)."""
    r, c = divmod(slot, 8)
    return f"{chr(97 + c)}{(8 - r) + 1}"


def vertex_to_slot(vertex: str) -> int:
    """QTP vertex -> our 8x8 wall slot (r*8+c)."""
    c = ord(vertex[0].lower()) - 97
    i = int(vertex[1:]) - 1
    return (8 - i) * 8 + c


def action_to_qtp(action: int, player: int):
    """Our action + mover -> (qtp command string). player 0 = white, 1 = black."""
    colour = 'white' if player == 0 else 'black'
    if action < 81:
        return f'playmove {colour} {cell_to_vertex(action)}'
    if action < 145:
        return f'playwall {colour} {slot_to_vertex(action - 81)} h'
    return f'playwall {colour} {slot_to_vertex(action - 145)} v'


def decode_genmove(line: str) -> int:
    """A genmove response line ('= <VERTEX>' or '= <VERTEX> <h|v>') -> our action int."""
    toks = line[1:].split()
    if not toks:
        raise EngineError(f'empty genmove response: {line!r}')
    if len(toks) == 1:
        return vertex_to_cell(toks[0])
    if toks[1] == 'h':
        return 81 + vertex_to_slot(toks[0])
    if toks[1] == 'v':
        return 145 + vertex_to_slot(toks[0])
    raise EngineError(f'unparseable genmove response: {line!r}')


def _slots(mask: int):
    out = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


def _fp(s: State):
    return (s.p0, s.p1, s.walls0, s.walls1, s.h, s.v, s.player, s.ply)


def diff_action(T: State, M: State):
    """The single action of side-to-move in T that would turn T into M, or None.

    Used to recover the opponent's last move from two consecutive states. The
    candidate is always re-verified by the caller (replay + fingerprint), so a
    heuristic false positive here is caught rather than acted on.
    """
    if M.player != 1 - T.player or M.ply != T.ply + 1:
        return None
    p = T.player
    if T.h == M.h and T.v == M.v and T.walls0 == M.walls0 and T.walls1 == M.walls1:
        if p == 0 and T.p1 == M.p1 and T.p0 != M.p0:
            return M.p0
        if p == 1 and T.p0 == M.p0 and T.p1 != M.p1:
            return M.p1
        return None
    if T.p0 != M.p0 or T.p1 != M.p1:
        return None
    if p == 0 and (M.walls0 != T.walls0 - 1 or M.walls1 != T.walls1):
        return None
    if p == 1 and (M.walls1 != T.walls1 - 1 or M.walls0 != T.walls0):
        return None
    dh, dv = M.h ^ T.h, M.v ^ T.v
    if dh and not dv and (T.h & M.h) == T.h:
        bits = _slots(dh)
        if len(bits) == 1:
            return 81 + bits[0]
    if dv and not dh and (T.v & M.v) == T.v:
        bits = _slots(dv)
        if len(bits) == 1:
            return 145 + bits[0]
    return None


def empty_board_path(src: int, dst: int, blocked: int):
    """Shortest cell path src->dst on a wall-free 9x9 board, avoiding `blocked`.

    Returns the list of cells to step through (excluding src, including dst), or None
    if dst is unreachable (only when dst == blocked). Used to walk pawns during a
    from-scratch reconstruction, where there are no walls yet to obstruct them.
    """
    if src == dst:
        return []
    prev = {src: -1}
    q = deque((src,))
    while q:
        x = q.popleft()
        if x == dst:
            break
        r, c = divmod(x, 9)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                y = rr * 9 + cc
                if y != blocked and y not in prev:
                    prev[y] = x
                    q.append(y)
    if dst not in prev:
        return None
    path = []
    x = dst
    while x != src:
        path.append(x)
        x = prev[x]
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# Engine subprocess                                                           #
# --------------------------------------------------------------------------- #
class EngineError(RuntimeError):
    """The engine subprocess misbehaved (crashed, or rejected a command)."""


def engine_binary():
    """Path to the compiled engine, or None if it has not been built yet."""
    for name in ('ipquoridor', 'ipquoridor.exe'):
        p = ROOT / 'bin' / name
        if p.is_file():
            return p
    return None


def ensure_engine_built() -> bool:
    """Build the engine if its binary is missing. Works on the Linux server; on a
    machine with no C compiler the build fails and False is returned (moves then
    fall back to a legal move)."""
    if engine_binary() is not None:
        return True
    if not ROOT.is_dir():
        print(f'[pavlosdais_bridge] engine dir not found: {ROOT}', file=sys.stderr, flush=True)
        return False
    for target in (['make', 'final'], ['make']):
        try:
            r = subprocess.run(target, cwd=str(ROOT), capture_output=True,
                               text=True, timeout=600)
            if r.returncode != 0:
                print(f'[pavlosdais_bridge] {" ".join(target)} rc={r.returncode}: '
                      f'{r.stderr.strip()[:400]}', file=sys.stderr, flush=True)
        except (OSError, subprocess.SubprocessError) as e:
            print(f'[pavlosdais_bridge] {" ".join(target)} failed: {e}',
                  file=sys.stderr, flush=True)
        if engine_binary() is not None:
            return True
    print('[pavlosdais_bridge] no engine binary and build failed '
          '(no C compiler on this host?).', file=sys.stderr, flush=True)
    return False


class EngineSession:
    """One engine subprocess dedicated to a single game, plus a shadow of its position."""

    def __init__(self):
        binary = engine_binary()
        if binary is None:
            raise EngineError('engine binary not available (build it with `make`)')
        try:
            self.proc = subprocess.Popen(
                [str(binary)], cwd=str(ROOT),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace', bufsize=1)
        except OSError as e:
            raise EngineError(f'cannot launch engine: {e}') from e
        # A brand-new engine sits at the start position with 10 walls each.
        self.shadow = State()

    def command(self, cmd: str) -> str:
        """Send one QTP command, return its '='/'?' response line. Raises on '?' or death."""
        if self.proc.poll() is not None:
            raise EngineError(f'engine exited with code {self.proc.returncode}')
        try:
            self.proc.stdin.write(cmd + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise EngineError(f'engine stdin closed: {e}') from e
        line = self._read_response()
        if line.startswith('?'):
            raise EngineError(f'engine rejected {cmd!r}: {line!r}')
        return line

    def _read_response(self) -> str:
        while True:
            line = self.proc.stdout.readline()
            if line == '':
                raise EngineError('engine closed its stdout (crashed?)')
            line = line.strip()
            if not line:
                continue
            if line[0] in '=?':
                return line
            # Stray output (we never call showboard/list_commands, so this is defensive).

    def apply(self, action: int, player: int):
        """Replay one of our actions on the engine (leaves the engine at the new state)."""
        self.command(action_to_qtp(action, player))

    def genmove(self, player: int) -> int:
        """Ask the engine to play for `player`; return the move as our action int."""
        return decode_genmove(self.command(f'genmove {"white" if player == 0 else "black"}'))

    def close(self):
        proc = getattr(self, 'proc', None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write('quit\n')
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        finally:
            self.proc = None


def _drop_session(sess: EngineSession):
    if sess in SESSIONS:
        SESSIONS.remove(sess)
    sess.close()


def reconstruct(sess: EngineSession, M: State):
    """Drive a fresh engine from the start position to match M, or raise EngineError."""
    starts = {'white': 76, 'black': 4}       # our-cell pawn starts (p0=white, p1=black)
    targets = {'white': M.p0, 'black': M.p1}

    placed = False
    for first, second in (('white', 'black'), ('black', 'white')):
        try:
            sess.command('clear_board')      # pawns back to start; walls-remaining still 10/10
            pos = dict(starts)
            for colour in (first, second):
                other = 'black' if colour == 'white' else 'white'
                path = empty_board_path(pos[colour], targets[colour], pos[other])
                if path is None:
                    raise EngineError('pawn blocked during reconstruction')
                for cell in path:
                    sess.command(f'playmove {colour} {cell_to_vertex(cell)}')
                pos[colour] = targets[colour]
            placed = True
            break
        except EngineError:
            continue
    if not placed:
        raise EngineError('could not place pawns during reconstruction')

    # Walls: 10-w0 attributed to white, the rest to black, so counts end at (w0, w1).
    to_white = 10 - M.walls0
    done_white = 0
    walls = [('h', s) for s in _slots(M.h)] + [('v', s) for s in _slots(M.v)]
    for orient, slot in walls:
        colour = 'white' if done_white < to_white else 'black'
        if colour == 'white':
            done_white += 1
        sess.command(f'playwall {colour} {slot_to_vertex(slot)} {orient}')
    sess.shadow = M.copy()


def find_session(M: State):
    """Locate the session already tracking this game. Returns (session, opp_move|None)
    where opp_move is the single move to replay to bring the engine up to M, or None
    if the engine already sits at M. Returns (None, None) if no session matches."""
    for sess in reversed(SESSIONS):
        T = sess.shadow
        if _fp(T) == _fp(M):
            return sess, None
        a = diff_action(T, M)
        if a is not None and _fp(apply_unchecked(T, a)) == _fp(M):
            return sess, a
    return None, None


def engine_move(M: State) -> int:
    """Produce the engine's move for state M as our action int, or raise EngineError."""
    sess, opp = find_session(M)
    if sess is None:
        if len(SESSIONS) >= MAX_SESSIONS:
            _drop_session(SESSIONS[0])
        sess = EngineSession()
        SESSIONS.append(sess)
        try:
            reconstruct(sess, M)
        except EngineError:
            _drop_session(sess)
            raise
    else:
        SESSIONS.remove(sess)
        SESSIONS.append(sess)                # most-recently-used at the end
        if opp is not None:
            try:
                sess.apply(opp, 1 - M.player)  # the opponent (net) side made this move
                sess.shadow = M.copy()
            except EngineError:
                _drop_session(sess)
                raise

    raw = sess.genmove(M.player)
    if raw in legal_actions(M):
        sess.shadow = apply_unchecked(M, raw)  # keep the engine's internal state tracked
        return raw
    # The engine's move is illegal in our rules: its internal state now diverges from
    # what we will report, so this session can no longer be trusted. Drop it (the next
    # state for this game reconstructs) and signal a fallback.
    _drop_session(sess)
    raise EngineError(f'engine returned illegal action {raw}')


def handle_state(msg) -> dict:
    M = State(msg['p0'], msg['p1'], msg['w0'], msg['w1'],
              sum(1 << s for s in msg['h']), sum(1 << s for s in msg['v']),
              msg['player'], msg['ply'])
    legal = legal_actions(M)
    try:
        return {'a': int(engine_move(M))}
    except EngineError as e:
        print(f'[pavlosdais_bridge] {e}', file=sys.stderr, flush=True)
    except Exception as e:                                     # noqa: BLE001 - never crash a match
        print(f'[pavlosdais_bridge] {type(e).__name__}: {e}', file=sys.stderr, flush=True)
    if legal:
        return {'a': int(legal[RNG.randrange(len(legal))])}
    return {'forfeit': 'no legal actions'}


def _close_all():
    while SESSIONS:
        SESSIONS.pop().close()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get('type')
        if kind == 'hello':
            RNG.seed(msg.get('seed', 0))
            ensure_engine_built()
            reply = {'ok': True, 'name': 'pavlosdais'}
        elif kind == 'state':
            reply = handle_state(msg)
        elif kind == 'bye':
            _close_all()
            sys.exit(0)
        else:
            reply = {'forfeit': f'unknown message type {kind!r}'}
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()
    _close_all()


# --------------------------------------------------------------------------- #
# Offline self-test (no engine / no compiler required)                        #
# --------------------------------------------------------------------------- #
def _selftest():
    # Pawn cell round-trips, and our action -> QTP -> our action for every cell.
    for cell in range(81):
        assert vertex_to_cell(cell_to_vertex(cell)) == cell, cell
        assert decode_genmove('= ' + cell_to_vertex(cell)) == cell, cell
    # Known landmark vertices (start squares).
    assert cell_to_vertex(76) == 'e1', cell_to_vertex(76)   # p0 / white start
    assert cell_to_vertex(4) == 'e9', cell_to_vertex(4)     # p1 / black start
    assert vertex_to_cell('E1') == 76 and vertex_to_cell('e9') == 4  # case-insensitive

    # Wall slot round-trips, and full action round-trips through QTP text.
    for slot in range(64):
        assert vertex_to_slot(slot_to_vertex(slot)) == slot, slot
        ha, va = 81 + slot, 145 + slot
        hcmd, vcmd = action_to_qtp(ha, 0), action_to_qtp(va, 1)
        assert hcmd == f'playwall white {slot_to_vertex(slot)} h', hcmd
        assert vcmd == f'playwall black {slot_to_vertex(slot)} v', vcmd
        assert decode_genmove('= ' + slot_to_vertex(slot) + ' h') == ha, slot
        assert decode_genmove('= ' + slot_to_vertex(slot) + ' v') == va, slot
    # Landmark wall: our h slot (0,0) is the top-most internal line -> engine i=8, 'a9'.
    assert slot_to_vertex(0) == 'a9', slot_to_vertex(0)
    assert vertex_to_slot('a9') == 0
    # Uppercase genmove output (as the engine actually prints it) decodes correctly.
    assert decode_genmove('= E4') == vertex_to_cell('e4')
    assert decode_genmove('= A9 h') == 81 + vertex_to_slot('a9')

    # Pawn-move command formatting for both colours.
    assert action_to_qtp(49, 0) == f'playmove white {cell_to_vertex(49)}'
    assert action_to_qtp(49, 1) == f'playmove black {cell_to_vertex(49)}'

    # Move-diff recovers a pawn move: p0 (start) steps 76 -> 67.
    s0 = State()
    s1 = apply_unchecked(s0, 67)
    a = diff_action(s0, s1)
    assert a == 67 and _fp(apply_unchecked(s0, a)) == _fp(s1), a
    # Move-diff recovers a wall: from s1 (p1 to move) place horizontal wall slot 20.
    s2 = apply_unchecked(s1, 81 + 20)
    a = diff_action(s1, s2)
    assert a == 81 + 20 and _fp(apply_unchecked(s1, a)) == _fp(s2), a
    # Move-diff recovers a vertical wall by p0.
    s3 = apply_unchecked(s2, 145 + 33)
    a = diff_action(s2, s3)
    assert a == 145 + 33 and _fp(apply_unchecked(s2, a)) == _fp(s3), a
    # Non-consecutive / unrelated states do not spuriously match.
    assert diff_action(s0, s2) is None
    assert diff_action(s0, s0) is None

    # empty_board_path: adjacency, endpoints, and detour around a blocked cell.
    p = empty_board_path(76, 4, 0)
    assert p and p[-1] == 4 and 4 not in (76,)
    prev = 76
    for cell in p:
        pr, pc = divmod(prev, 9)
        cr, cc = divmod(cell, 9)
        assert abs(pr - cr) + abs(pc - cc) == 1, (prev, cell)
        prev = cell
    assert empty_board_path(76, 4, 4) is None      # dst == blocked -> unreachable
    assert empty_board_path(40, 40, 0) == []       # already there
    # Walk avoids the blocked cell entirely.
    assert 13 not in empty_board_path(4, 22, 13)

    print('selftest: OK')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        _selftest()
    else:
        main()

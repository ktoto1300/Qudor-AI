r"""Bridge for giorgosnikolaou/Quoridor-Engine (C, alpha-beta + iterative deepening).

The engine is a compiled C program that speaks the Quoridor Text Protocol (QTP) over
stdin/stdout. This bridge launches it as a subprocess and translates between OUR
JSON-lines protocol (full state each turn) and QTP.

============================================================================
WHY WE REBUILD THE FULL POSITION EVERY TURN (no position-load command)
============================================================================
The engine's QTP command set (see src/quoridor.c, `commands[]`) is:
    name known_command list_commands quit boardsize clear_board walls
    playmove playwall genmove undo winner showboard
There is NO command to load an arbitrary position. Pawns move one legal step at a time
(`playmove`), walls are placed one at a time (`playwall`, which decrements the placing
player's wall count and validates path existence). `boardsize`/`clear_board` only reset
to the standard start.

Our arena sends the FULL state every turn, not a move list -- AND it multiplexes many
concurrent games through a single bridge process (see foreign_arena.play(): every round
it asks this one bot to move for every live game). So consecutive `state` messages are
usually UNRELATED positions from different games. A persistent per-game engine session
is therefore impossible with one subprocess.

Instead, every turn this bridge rebuilds the requested position from a clean board:
    boardsize 9; walls N; (walk each pawn from its start to its cell);
    (place every wall, attributed to White/Black so the remaining counts match); genmove.
Legality holds at every step because the target position is one our arena validated:
pawns walk on the still-empty board (a 9x9 grid minus one occupied cell stays connected),
then walls are added to a pawn layout that is already final, and any subset of a legal
wall set keeps both goal paths open. The engine's own move after `genmove` is discarded --
the next turn starts from a clean board again -- so no `undo` bookkeeping is needed.

Only `genmove` is slow (~4.5s, hard-coded); the setup commands return instantly.

============================================================================
QTP COMMANDS USED
============================================================================
    boardsize 9          -> "=\n\n"      (MUST come first: engine->board is NULL until
                                          boardsize allocates it, and every play command
                                          dereferences it)
    walls N              -> "=\n\n"       (starting walls per player; size-9 default is
                                          (int)(7/4*9 - 23/4) = 10)
    playmove <C> <V>     -> "=\n\n"       (one pawn step for colour C to vertex V)
    playwall <C> <V> <O> -> "=\n\n"       (wall for colour C at vertex V orientation O)
    genmove  <C>         -> "= <V>\n\n" (pawn) | "= <V> <O>\n\n" (wall)
    quit                 -> "=\n\n"       (on {"type":"bye"})
Each response is one line starting with '=' (ok) or '?' (error), then a blank line; we
read line-by-line, skipping blanks and stray output, until the marker line.

============================================================================
COLOUR MAPPING
============================================================================
Engine `board->player` = WHITE ('W'): starts internal (i=dim-1, j=dim/2), goal row i==0.
Engine `board->enemy`  = BLACK ('B'): starts (i=0, j=dim/2), goal i==dim-1. Internal row
i=0 is TOP, i=dim-1 is BOTTOM -- identical to OUR convention. Hence:
    OUR player 0 (start cell 76 = row8/bottom, goal row0) == engine WHITE 'W'
    OUR player 1 (start cell 4  = row0/top,    goal row8) == engine BLACK 'B'
No mirroring: internal (i,j) == (our_row, our_col).

============================================================================
COORDINATE / WALL MAPPING  (DIM = 9)
============================================================================
QTP vertex is "<Letter><Number>", e.g. "E1". From src/quoridor.c read_command:
    column j = Letter - 'A'        (A=col0 ... I=col8)
    internal row i = dim - Number  (Number 9 -> i=0 top ... Number 1 -> i=8 bottom)
genmove prints the same: Letter = j+'A', Number = dim - i.

PAWN (our cell = row*9 + col):
    our -> QTP:  Letter=chr('A'+col), Number=9-row
    QTP -> our:  col=Letter-'A', row=9-Number, cell=row*9+col
    Check: cell 76 (row8,col4) -> "E1" (white start); cell 4 (row0,col4) -> "E9".

WALLS. Engine hor_walls[x][y] blocks the line between rows x,x+1 across cols y,y+1;
ver_walls[x][y] blocks between cols y,y+1 across rows x,x+1. playwall/genmove use
internal (x = dim - Number, y = Letter - 'A') plus orientation 'H'/'V'. This matches OUR
slot semantics exactly:
    OUR h-slot (r,c): line between rows r,r+1 across cols c,c+1  == engine (x=r,y=c) 'H'
    OUR v-slot (r,c): line between cols c,c+1 across rows r,r+1  == engine (x=r,y=c) 'V'
So:
    our h action a (81..144): slot=a-81, r,c=divmod(slot,8)
        -> vertex Letter=chr('A'+c), Number=9-r, orientation 'H'
    our v action a (145..208): slot=a-145 -> ..., orientation 'V'
    QTP -> our: r=9-Number, c=Letter-'A'; 'H' -> 81+r*8+c ; 'V' -> 145+r*8+c
    Check: h-slot (0,0) -> "A9 H";  v-slot (0,0) -> "A9 V".

============================================================================
TIME / DEPTH CONTROL
============================================================================
The engine HARD-CODES its budget in engine_genmove (time_left=4.5s, max_depth=5) and
`main()` takes no args, so there is NO runtime knob to shorten its thinking without
editing its source (forbidden). It self-limits to ~4.5s/move regardless.

`GIORGOS_MOVETIME` (float seconds, default 30.0) is the bridge's *patience*: how long to
wait for a `genmove` reply before treating the engine as hung and forfeiting. It cannot
reduce the engine's think time; it only turns a genuine deadlock/crash into a clean
forfeit. Keep it above ~5s. Other commands use a short fixed timeout (CMD_TIMEOUT).

============================================================================
BUILD
============================================================================
Makefile at repo root: `EXEC = main`, rule `gcc $(OBJS) -o main`. Binary name is `main`
(no extension on Linux; `main.exe` if built on Windows). If missing, this bridge runs
`make` in the repo dir (works on the Linux tournament server with gcc). Build command:
`make` (cwd = repo dir).

============================================================================
ILLEGAL-MOVE FALLBACK
============================================================================
Each state is reconstructed as an OUR-encoding `State`; the engine's chosen move is
checked against `legal_actions`. If it is illegal under our rules we return a legal
fallback instead (a pawn step that most reduces our distance to goal, else any legal
action). No engine sync is needed because the next turn rebuilds from a clean board.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

# Running as a spawned script, not through the package: make the repo importable first,
# mirroring the setup in the sibling bridges (see dimi_bridge.py).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quoridor_ai.paths import bots_dir, repo_root      # noqa: E402

_REPO_PY = repo_root()
if str(_REPO_PY) not in sys.path:
    sys.path.insert(0, str(_REPO_PY))
from quoridor_ai.core.engine import (                  # noqa: E402
    State, dist_field, legal_actions,
)

ROOT = bots_dir() / 'giorgosnikolaou_Quoridor-Engine'
BINARY_NAMES = ('main', 'main.exe')          # Makefile EXEC = main
DIM = 9
COLOUR = {0: 'W', 1: 'B'}                    # our player 0 -> WHITE, player 1 -> BLACK
P0_START = 76                                # row8,col4 (white)
P1_START = 4                                 # row0,col4 (black)

MOVETIME = float(os.environ.get('GIORGOS_MOVETIME', '30.0'))   # genmove wait, seconds
CMD_TIMEOUT = 15.0                                             # other commands, seconds


# --------------------------------------------------------------------------- #
# Pure coordinate/wall mapping helpers (no engine needed; unit-tested inline). #
# --------------------------------------------------------------------------- #
def pawn_our_to_qtp(cell: int) -> str:
    """OUR pawn cell (row*9+col) -> QTP vertex string, e.g. 76 -> 'E1'."""
    row, col = divmod(cell, 9)
    return f'{chr(ord("A") + col)}{DIM - row}'


def pawn_qtp_to_our(vertex: str) -> int:
    """QTP vertex 'E1' -> OUR pawn cell (row*9+col)."""
    col = ord(vertex[0].upper()) - ord('A')
    number = int(vertex[1:])
    row = DIM - number
    return row * 9 + col


def wall_our_to_qtp(action: int) -> tuple[str, str]:
    """OUR wall action (81..208) -> (QTP vertex, orientation 'H'/'V')."""
    if action < 145:
        slot = action - 81
        ori = 'H'
    else:
        slot = action - 145
        ori = 'V'
    r, c = divmod(slot, 8)
    return f'{chr(ord("A") + c)}{DIM - r}', ori


def wall_qtp_to_our(vertex: str, ori: str) -> int:
    """(QTP vertex, orientation) -> OUR wall action (81..208)."""
    c = ord(vertex[0].upper()) - ord('A')
    number = int(vertex[1:])
    r = DIM - number
    base = 145 if ori.upper().startswith('V') else 81
    return base + r * 8 + c


def _grid_path(start: int, target: int, avoid: int):
    """Shortest path start..target on the empty 9x9 grid, avoiding cell `avoid`.

    Returns the list of cells [start, ..., target] (inclusive) or None. Walls are absent
    at pawn-placement time, so plain 4-neighbour adjacency is enough; a 9x9 grid minus
    one occupied cell stays connected, so a path all but always exists.
    """
    if start == target:
        return [start]
    prev = {start: -1}
    q = deque([start])
    while q:
        x = q.popleft()
        r, c = divmod(x, 9)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                y = rr * 9 + cc
                if y == avoid or y in prev:
                    continue
                prev[y] = x
                if y == target:
                    path = [y]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(y)
    return None


def _slot_bits(mask: int):
    """Set bit indices of a wall bitboard, low to high."""
    out = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


def _fallback(s: State):
    """A legal action for `s`: a pawn step that most reduces distance to goal, else any."""
    legal = legal_actions(s)
    if not legal:
        return None
    pawns = [a for a in legal if a < 81]
    if pawns:
        d = dist_field(s, 0 if s.player == 0 else 8)
        return min(pawns, key=lambda a: d[a])
    return legal[0]


class EngineError(RuntimeError):
    """The engine subprocess could not be started, built, or talked to."""


class Desync(RuntimeError):
    """A position could not be reconstructed in the engine (should be rare)."""


# --------------------------------------------------------------------------- #
# Engine subprocess wrapper.                                                    #
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self):
        self.proc = None
        self._q: queue.Queue = queue.Queue()
        self._reader = None

    # -- process lifecycle -------------------------------------------------- #
    def _binary(self) -> Path:
        for name in BINARY_NAMES:
            p = ROOT / name
            if p.is_file():
                return p
        # Missing: build it. `make` works on the Linux server (gcc present).
        print(f'[giorgos] binary missing; running make in {ROOT}', file=sys.stderr, flush=True)
        try:
            res = subprocess.run(['make'], cwd=str(ROOT), capture_output=True, text=True)
        except OSError as exc:
            raise EngineError(f'cannot run make in {ROOT}: {exc}') from exc
        if res.stdout:
            print(res.stdout, file=sys.stderr, flush=True)
        if res.stderr:
            print(res.stderr, file=sys.stderr, flush=True)
        if res.returncode != 0:
            raise EngineError(f'make failed (exit {res.returncode}) in {ROOT}')
        for name in BINARY_NAMES:
            p = ROOT / name
            if p.is_file():
                return p
        raise EngineError(f'make produced no binary {BINARY_NAMES} in {ROOT}')

    def start(self):
        binary = self._binary()
        try:
            self.proc = subprocess.Popen(
                [str(binary)], cwd=str(ROOT),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                text=True, encoding='utf-8', errors='replace', bufsize=1)
        except OSError as exc:
            raise EngineError(f'cannot start engine {binary}: {exc}') from exc
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self._q.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._q.put(None)                    # EOF sentinel

    # -- low-level IO ------------------------------------------------------- #
    def _send(self, text: str):
        if self.proc is None or self.proc.poll() is not None:
            raise EngineError('engine is not running')
        try:
            self.proc.stdin.write(text + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError(f'engine closed its stdin: {exc}') from exc

    def _read_response(self, timeout: float):
        """Read until a QTP marker line ('=' ok / '?' error). Returns (ok, payload)."""
        while True:
            try:
                line = self._q.get(timeout=timeout)
            except queue.Empty as exc:
                raise EngineError(f'engine did not answer within {timeout:.0f}s') from exc
            if line is None:
                raise EngineError('engine closed its stdout')
            line = line.strip()
            if not line:
                continue
            if line[0] == '=':
                return True, line[1:].strip()
            if line[0] == '?':
                return False, line[1:].strip()
            # Any other stray output (diagnostics) is ignored.

    def _command(self, text: str, timeout: float = CMD_TIMEOUT):
        self._send(text)
        return self._read_response(timeout)

    def _require(self, text: str):
        ok, payload = self._command(text)
        if not ok:
            raise Desync(f'engine rejected {text!r}: {payload!r}')

    # -- position reconstruction -------------------------------------------- #
    def _rebuild(self, cur: State):
        """Recreate `cur` on a clean board via boardsize/walls/playmove/playwall."""
        h_slots = _slot_bits(cur.h)
        v_slots = _slot_bits(cur.v)
        on_board = len(h_slots) + len(v_slots)
        # start walls = placed + remaining, and placed = 2*start - (w0+w1) = on_board.
        start_walls = (on_board + cur.walls0 + cur.walls1) // 2

        self._require(f'boardsize {DIM}')
        self._require(f'walls {start_walls}')

        # Pawns first, on the still-empty board: White (p0) from its start, then Black.
        self._walk_pawn(0, P0_START, cur.p0, avoid=P1_START)
        self._walk_pawn(1, P1_START, cur.p1, avoid=cur.p0)

        # Walls next, attributed so the remaining counts end at (w0, w1). Any order is
        # legal: every subset of a legal wall set keeps both goal paths open.
        placed_white = start_walls - cur.walls0
        walls = ([('H', s) for s in h_slots] + [('V', s) for s in v_slots])
        for i, (ori, slot) in enumerate(walls):
            colour = 'W' if i < placed_white else 'B'
            action = (81 + slot) if ori == 'H' else (145 + slot)
            vertex, o = wall_our_to_qtp(action)
            self._require(f'playwall {colour} {vertex} {o}')

    def _walk_pawn(self, player: int, start: int, target: int, avoid: int):
        if start == target:
            return
        path = _grid_path(start, target, avoid)
        if path is None:
            raise Desync(f'no pawn path {start}->{target} avoiding {avoid}')
        colour = COLOUR[player]
        for cell in path[1:]:
            self._require(f'playmove {colour} {pawn_our_to_qtp(cell)}')

    # -- move generation ---------------------------------------------------- #
    def _parse_genmove(self, payload: str):
        toks = payload.split()
        if not toks:
            raise EngineError(f'empty genmove reply: {payload!r}')
        if len(toks) == 1:                         # pawn move
            return pawn_qtp_to_our(toks[0])
        return wall_qtp_to_our(toks[0], toks[1])   # wall placement

    def move(self, cur: State):
        """Return (action, note) in OUR encoding, or raise EngineError/Desync."""
        self._rebuild(cur)
        ok, payload = self._command(f'genmove {COLOUR[cur.player]}', MOVETIME)
        if not ok:
            raise EngineError(f'genmove failed: {payload!r}')
        a_engine = self._parse_genmove(payload)
        if a_engine in set(legal_actions(cur)):
            return a_engine, None
        # Illegal under our rules: substitute a legal action. No engine sync needed --
        # the next turn rebuilds from a clean board.
        fallback = _fallback(cur)
        if fallback is None:
            raise EngineError('no legal fallback action available')
        return fallback, f'engine move {a_engine} illegal under our rules'

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write('quit\n')
                    self.proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.proc = None


def main():
    session = Session()
    started = False
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            kind = msg.get('type')

            if kind == 'hello':
                # Lazy engine start: build+spawn on the first real move so the handshake
                # never blocks and build errors surface as a per-move forfeit.
                sys.stdout.write(json.dumps({'ok': True, 'name': 'giorgos'}) + '\n')
                sys.stdout.flush()
                continue

            if kind == 'bye':
                break

            if kind == 'state':
                cur = State(msg['p0'], msg['p1'], msg['w0'], msg['w1'],
                            sum(1 << s for s in msg['h']),
                            sum(1 << s for s in msg['v']),
                            msg['player'], msg['ply'])
                try:
                    if not started:
                        session.start()
                        started = True
                    a, note = session.move(cur)
                    reply = {'a': int(a)}
                    if note:
                        reply['illegal'] = note
                except Exception as exc:            # never crash the bridge
                    reply = {'forfeit': f'{type(exc).__name__}: {exc}'}
                sys.stdout.write(json.dumps(reply) + '\n')
                sys.stdout.flush()
                continue

            # Unknown message type: ignore.
    finally:
        session.close()
    sys.exit(0)


if __name__ == '__main__':
    main()

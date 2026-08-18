"""Bridge for bartolomeo3000/SigmaQuoridor (AlphaZero: ResNet policy/value + PUCT MCTS).

Speaks the arena JSON-lines protocol on stdin/stdout and drives SigmaQuoridor's
own `game.State` + `mcts.MCTSAgent` with the trained 9x9 net.

Weights
-------
    runs/models_9x9_heads/best.pt  (~11 MB, boardsize=9, filters=128, 10 residual
    blocks). Loaded once via `dual_network.make_nn_evaluator(path, device=cpu)` and
    reused. `MCTSAgent(training=False, temperature=0.0)` -> deterministic argmax play,
    exactly as app.py wires the "alphazero" agent for 9x9.

Search budget
-------------
    Env var SIGMA_SIMS (default 200) sets MCTS simulations per move. ~200 sims returns
    in a couple of seconds on CPU.

Coordinate / wall mapping (OURS <-> THEIRS) -- discovered by reading game.py
---------------------------------------------------------------------------
OURS   : cell = row*9 + col; row 0 = TOP, row 8 = BOTTOM, col 0 = LEFT.
         p0 starts bottom (idx 76, row 8 col 4) and races to row 0 (top).
         p1 starts top    (idx  4, row 0 col 4) and races to row 8 (bottom).
THEIRS : positions are (x, y) with origin (0,0) at BOTTOM-LEFT; y grows UPWARD.
         player1 starts (4,0) bottom, goal row y=N-1 (top).
         player2 starts (4,8) top,    goal row y=0   (bottom).

So the two vertical axes are inverted, giving a clean role map:
    OUR p0  == THEIR player1   (both start bottom, climb up)
    OUR p1  == THEIR player2   (both start top,    go down)
    THEIR x   = OUR col
    THEIR y   = 8 - OUR row            (row = 8 - y,  col = x)
    OUR cell  = (8 - y)*9 + x          THEIR (x,y) = (col, 8 - row)

Walls. OUR wall slot index = r*8 + c (r,c in 0..7). THEIR walls are anchored at
(x,y) on an (N-1)x(N-1) grid, orientation 'h'/'v'. A horizontal wall spans the
boundary between THEIR rows y and y+1 over columns x,x+1; a vertical wall spans
the boundary between THEIR columns x and x+1 over rows y,y+1. Matching those
boundaries to OUR slot (which sits between OUR rows r and r+1 / cols c and c+1)
gives the SAME transform for both orientations:
    OUR slot (r,c)  ->  THEIR anchor (x = c, y = 7 - r)
    THEIR anchor (x,y) -> OUR slot (r = 7 - y, c = x); slot index = (7-y)*8 + x

Action int `a` reply encoding (OURS):
    a < 81            : pawn move to destination cell a (= row*9+col)
    81  <= a < 145    : place horizontal wall, slot = a-81
    145 <= a < 209    : place vertical wall,   slot = a-145

Protocol hygiene: SigmaQuoridor modules can print to stdout, which would corrupt
the JSON stream, so every engine call (import, load, search) runs with stdout
temporarily redirected to stderr. Only JSON replies ever reach real stdout, which
is flushed after each line.
"""
import contextlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quoridor_ai.paths import bots_dir, repo_root     # noqa: E402
from quoridor_ai.core.engine import State as OurState, legal_actions  # noqa: E402

ENGINE = bots_dir() / 'bartolomeo3000_SigmaQuoridor'
# Repo root first (so `quoridor_ai` stays importable), then the engine dir at the
# very front so its top-level `game`/`mcts`/`dual_network` win any name lookup.
sys.path.insert(0, str(repo_root()))
sys.path.insert(0, str(ENGINE))

N = 9
WALLS_INITIAL = 10
WEIGHTS = ENGINE / 'runs' / 'models_9x9_heads' / 'best.pt'
SIMS = int(os.environ.get('SIGMA_SIMS', '200'))


@contextlib.contextmanager
def _stdout_to_stderr():
    """Silence engine prints so they cannot corrupt the JSON stream."""
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = saved


with _stdout_to_stderr():
    import torch
    from game import PawnAction, State as SigmaState
    from mcts import MCTSAgent
    from dual_network import make_nn_evaluator


def _build_sigma_state(msg):
    """Translate OUR (p0,p1,w0,w1,h,v,player,ply) into a SigmaQuoridor State."""
    p0r, p0c = divmod(msg['p0'], 9)
    p1r, p1c = divmod(msg['p1'], 9)
    player1pos = (p0c, 8 - p0r)   # OUR p0 == THEIR player1
    player2pos = (p1c, 8 - p1r)   # OUR p1 == THEIR player2

    hwalls = bytearray(N * N)
    vwalls = bytearray(N * N)
    hanch = set()
    vanch = set()
    for s in msg['h']:
        r, c = divmod(s, 8)
        x, y = c, 7 - r
        hanch.add((x, y))
        hwalls[y * N + x] = 1
        hwalls[y * N + x + 1] = 1
    for s in msg['v']:
        r, c = divmod(s, 8)
        x, y = c, 7 - r
        vanch.add((x, y))
        vwalls[y * N + x] = 1
        vwalls[(y + 1) * N + x] = 1

    # depth parity picks whose turn it is: even -> player1 (our p0), odd -> player2.
    depth = msg['player']
    return SigmaState(
        boardsize=N,
        depth=depth,
        player1pos=player1pos,
        player2pos=player2pos,
        hwalls=hwalls,
        vwalls=vwalls,
        walls_p1=msg['w0'],
        walls_p2=msg['w1'],
        hwall_anchors=hanch,
        vwall_anchors=vanch,
        walls_initial=WALLS_INITIAL,
    )


def _sigma_action_to_ours(state, action):
    """Translate a SigmaQuoridor Action back into OUR action int."""
    if isinstance(action, PawnAction):
        dx, dy = state._pawn_dest(action)   # destination cell in THEIR (x,y)
        return (8 - dy) * 9 + dx
    # WallAction: THEIR anchor (x,y) -> OUR slot (7-y)*8 + x
    slot = (7 - action.y) * 8 + action.x
    return (81 if action.orientation == 'h' else 145) + slot


def main():
    with _stdout_to_stderr():
        evaluator = make_nn_evaluator(str(WEIGHTS), device=torch.device('cpu'))
        agent = MCTSAgent(evaluator=evaluator, num_simulations=SIMS,
                          training=False, temperature=0.0)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mtype = msg.get('type')

        if mtype == 'hello':
            sys.stdout.write(json.dumps({'ok': True, 'name': 'sigma'}) + '\n')
            sys.stdout.flush()
            continue
        if mtype == 'bye':
            break
        if mtype != 'state':
            continue

        # OUR-engine ground truth for the legality safety net.
        our_s = OurState(msg['p0'], msg['p1'], msg['w0'], msg['w1'],
                         sum(1 << s for s in msg['h']), sum(1 << s for s in msg['v']),
                         msg['player'], msg['ply'])
        legal = set(legal_actions(our_s))
        if not legal:
            sys.stdout.write(json.dumps({'forfeit': 'no legal actions'}) + '\n')
            sys.stdout.flush()
            continue

        try:
            with _stdout_to_stderr():
                sigma_s = _build_sigma_state(msg)
                action = agent.select_action(sigma_s)
            a = _sigma_action_to_ours(sigma_s, action)
        except Exception as exc:                       # noqa: BLE001
            print(f'sigma_bridge: engine error: {exc!r}', file=sys.stderr)
            a = None

        if a not in legal:
            if a is not None:
                print(f'sigma_bridge: mapped action {a} illegal; falling back',
                      file=sys.stderr)
            # Prefer a pawn move, else any legal wall.
            a = next((x for x in sorted(legal) if x < 81), None)
            if a is None:
                a = min(legal)

        sys.stdout.write(json.dumps({'a': a}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()

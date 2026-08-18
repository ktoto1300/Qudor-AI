"""Bridge for v-ade-r/QuoridorAI-AlphaZero (trained ResNet + MCTS, 136 actions).

Reads our state from stdin (JSON line), answers with our action int on stdout (JSON line).
Board mapping (their 17x17 grid, pawns on even coords, walls on odd):
  our p0 (bottom, start 76)  == their WHITE plane0, x=2*r, y=2*c
  our p1 (top,    start 4)   == their BLACK plane1, x=2*r, y=2*c
  our h-wall (r,c) -> their (x=2r+1, y=2c);         action 8+r*8+c
  our v-wall (r,c) -> their (x=2r+2, y=2c+1);       action 72+r*8+c
Their MCTS always searches the canonical board (white to move); the returned action is
in canonical space; pawn directions are converted to our pawn-destination ints.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quoridor_ai.paths import bots_dir                # noqa: E402

BOTS = bots_dir()
ROOT = BOTS / 'v-ade-r_QuoridorAI-AlphaZero'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pytorch_utils'))      # keep AFTER ROOT: pytorch_utils has
                                                     # its own utils/ subpackage shadowing
sys.path.insert(0, str(ROOT))                        # the repo's utils.py (dotdict)

import numpy as np
import torch

torch.cuda.is_available = lambda: False          # their NNet picks .cuda() from this
_real_load = torch.load                          # checkpoint predates weights_only
torch.load = lambda *a, **k: _real_load(*a, **k, map_location='cpu', weights_only=False)

from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper, args as nnet_args
from MCTS import MCTS

CKPT_DIR = str(ROOT / 'temp')
CKPT_NAME = 'best.pth.tar'
SIMS = 100
CPUCT = 1.0


class DotArgs(dict):
    """dotdict missing the getattr convenience; give MCTS its args here."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc


def main():
    np.set_printoptions(threshold=10)
    game = QuoridorGame(9)
    nnet_args.cuda = False
    nnet = NNetWrapper(game, DotArgs(numIters=1))
    nnet.load_checkpoint(CKPT_DIR, CKPT_NAME)
    mcts = MCTS(game, nnet, DotArgs(numMCTSSims=SIMS, cpuct=CPUCT,
                                    dirichletAlpha=0.03, epsilon=0.25))
    # getValidMoves (BFS wall legality, ~0.25 s) is recomputed for the same board on
    # every transposition inside MCTS; the MCTS already memoises Ps/Es/Ns but NOT valids.
    _valid_memo = {}
    _orig_valid = game.getValidMoves
    def _cached_valid(board, color):
        key = board.tobytes()
        v = _valid_memo.get(key)
        if v is None:
            v = _orig_valid(board, color)
            _valid_memo[key] = v
        return v
    game.getValidMoves = _cached_valid

    def canon(pieces, mover):
        # mover: 0 -> white is 'their +1' (we are on bottom) -> canonical = pieces as-is
        if mover == 0:
            return pieces
        out = np.empty_like(pieces)
        out[0] = np.flipud(np.fliplr(pieces[1]))
        out[1] = np.flipud(np.fliplr(pieces[0]))
        out[2] = np.flipud(np.fliplr(pieces[3]))
        out[3] = np.flipud(np.fliplr(pieces[2]))
        return out

    def to_our_action(canonical_a, mover):
        # canonical_a is expressed for 'white to move' on the canonical board.
        # For mover=1 the canonical board is the 180-degree rotation of ours, so pawn
        # destinations and wall slots must be rotated back.
        if canonical_a < 8:
            # canonical direction -> original direction (u<->d, l<->r, ul<->dr, dl<->ur)
            local = canonical_a if mover == 0 else (canonical_a + 1 if canonical_a % 2 == 0 else canonical_a - 1)
            res = game.board.move_action_destination(local, 1)
            if res is None:
                return None
            _, (x, y) = res                                # canonical 17x17 coords
            if mover == 0:
                return (x // 2) * 9 + (y // 2)
            return (8 - x // 2) * 9 + (8 - y // 2)
        if canonical_a < 72:
            c = canonical_a - 8
            r, cc = c // 8, c % 8
            if mover == 0:
                return 81 + r * 8 + cc
            return 81 + (7 - r) * 8 + (7 - cc)
        c = canonical_a - 72
        r, cc = c // 8, c % 8
        if mover == 0:
            return 145 + r * 8 + cc
        return 145 + (7 - r) * 8 + (7 - cc)

    for line in sys.stdin:
        msg = json.loads(line)
        if msg['type'] == 'hello':
            sys.stdout.write(json.dumps({'ok': True, 'name': 'vader'}) + '\n')
            sys.stdout.flush()
            continue
        if msg['type'] == 'bye':
            break
        # build pieces from our state (plane2 = h walls, plane3 = v walls; their legality
        # checks are orientation-agnostic path tests, so the split only shifts the NN input
        # slightly - their own convention splits by colour)
        pieces = np.zeros((4, 17, 17), dtype=int)
        r0, c0 = divmod(msg['p0'], 9)
        r1, c1 = divmod(msg['p1'], 9)
        pieces[0][2 * r0][2 * c0] = 1
        pieces[1][2 * r1][2 * c1] = 1
        for slot in msg['h']:
            r, c = divmod(slot, 8)
            pieces[2][2 * r + 1][2 * c] = 1
            pieces[2][2 * r + 1][2 * c + 2] = 1
        for slot in msg['v']:
            r, c = divmod(slot, 8)
            pieces[3][2 * r + 2][2 * c + 1] = 1
            pieces[3][2 * r][2 * c + 1] = 1
        mover = msg['player']
        cb = canon(pieces, mover)
        probs = mcts.getActionProb(cb, temp=0)
        best = int(np.argmax(probs))
        out = to_our_action(best, mover)
        if out is None:
            sys.stdout.write(json.dumps({'a': -1, 'illegal': True}) + '\n')
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({'a': out}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()

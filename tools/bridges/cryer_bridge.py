"""Bridge for cryer/AlphaZero_Quoridor (pure-MCTS, no trained weights).

Their game: player 1 = top (starts cell 4, goal >71), player 2 = bottom (starts cell 76,
goal <9) -> player 1 is our p1, player 2 is our p0; cell indices match ours (r*9+c).
Actions: 0..11 pawn directions with deltas (0:+9,1:-9,2:+1,3:-1,4:+18,5:-18,6:+2,7:-2,
8:+10,9:+8,10:-10,11:-8) from _handle_pawn_action; 12..75 = horizontal walls
(12 + r*8+c), 76..139 = vertical walls (76 + r*8+c) - wall slots identical to ours.

Their wall legality is ~0.2 s per actions() (double BFS per candidate), so random
rollouts with up to 1000 steps would take minutes per move. The bridge patches the
rollout evaluation: wall moves are disabled during rollouts only (tree expansion keeps
full legality), and the rollout step cap is lowered.
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(os.environ.get('BOTS_DIR', r'C:\Users\kamil\Desktop\bots')) / 'cryer_AlphaZero_Quoridor'
sys.path.insert(0, str(REPO))

from quoridor import Quoridor
import pure_mcts

PLAYOUTS = int(os.environ.get('CRYER_PLAYOUTS', '60'))
ROLLOUT_CAP = int(os.environ.get('CRYER_ROLLOUT_CAP', '60'))

DELTA = {0: 9, 1: -9, 2: 1, 3: -1, 4: 18, 5: -18, 6: 2, 7: -2, 8: 10, 9: 8, 10: -10, 11: -8}


def main():
    _orig_eval = pure_mcts.MCTS._evaluate_rollout

    def fast_eval(self, game, limit=ROLLOUT_CAP):
        orig = Quoridor._valid_wall_actions
        Quoridor._valid_wall_actions = lambda self_: []
        try:
            return _orig_eval(self, game, limit)
        finally:
            Quoridor._valid_wall_actions = orig

    pure_mcts.MCTS._evaluate_rollout = fast_eval
    player = pure_mcts.MCTSPlayer(c_puct=5, n_playout=PLAYOUTS)
    sink = io.StringIO()
    for line in sys.stdin:
        msg = json.loads(line)
        if msg['type'] == 'hello':
            sys.stdout.write(json.dumps({'ok': True, 'name': 'cryer', 'playouts': PLAYOUTS}) + '\n')
            sys.stdout.flush()
            continue
        if msg['type'] == 'bye':
            break
        g = Quoridor(safe=False)
        g._positions = {1: msg['p1'], 2: msg['p0']}
        g.current_player = 1 if msg['player'] == 1 else 2
        g.last_player = 2 if g.current_player == 1 else 1
        w = __import__('numpy').zeros(64)
        for slot in msg['h']:
            w[slot] = 1
        for slot in msg['v']:
            w[slot] = -1
        g._intersections = w
        g._player1_walls_remaining = msg['w1']
        g._player2_walls_remaining = msg['w0']
        with redirect_stdout(sink):
            a = player.choose_action(g)
        out = None
        if a < 12:
            out = g._positions[g.current_player] + DELTA[a]
        elif a < 76:
            out = 81 + (a - 12)
        else:
            out = 145 + (a - 76)
        sys.stdout.write(json.dumps({'a': out}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
r"""Bridge for berlioz10/quoridor-monte-carlo (pure MCTS, no weights).

Their world: human_player = top pawn (start (0,4), end_line 8), AI_player = bottom
pawn (start (8,4), end_line 0). The MCTS always plays the AI (bottom) side, so when
we must move for player 1 (top) we present the 180-degree mirror of our board and
mirror the returned move back (cells (r,c)->(8-r,8-c), wall slots (r,c)->(7-r,7-c),
wall orientation preserved; human end_line 8 / AI end_line 0 line up automatically).

Their moves: pawn strings from utils.consts (up/down/left/right/upup/.../downright),
walls as (row, col, HORIZONTAL|VERTICAL) tuples with the SAME slot geometry as ours
(h covers cols c..c+1, v covers rows r..r+1).
"""
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quoridor_ai.paths import bots_dir                # noqa: E402

BOTS = bots_dir()
ROOT = BOTS / 'berlioz10_quoridor-monte-carlo'
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from game.game import Game                                    # noqa: E402
from montecarlo.node import Node                              # noqa: E402
from utils.consts import BOARD_PAWN_DIM, HORIZONTAL, VERTICAL  # noqa: E402

ITER = int(os.environ.get('BERLIOZ_ITER', '250'))
ROLLOUT_CAP = int(os.environ.get('BERLIOZ_ROLLOUT_CAP', '20'))
ACTION_SIZE = 209

# their full rollout plays the game to the end (~100ms); cap it at ROLLOUT_CAP plies
# and score the truncated game by who is closer to the goal.
def _fast_sim(self, probability=0.70):
    game = self.deepcopy()
    plies = 0
    while not game.game_finished() and plies < ROLLOUT_CAP:
        random_move = random.uniform(0, 1)
        player1, player2 = game.human_player, game.AI_player
        dest = BOARD_PAWN_DIM - 1
        if not game.human_turn:
            player1, player2 = player2, player1
            dest = 0
        move = ''
        if player1.no_walls == 0:
            probability = 0.99
        if random_move < probability:
            move = game.board.shortest_path_move(player1, player2, dest)
            if move == '':
                random_move = 1.0
            else:
                game.make_move(move)
        if random_move >= probability:
            moves = game.board.get_all_actions_for_a_player(player1, player2)
            if not moves:
                break
            game.make_move(random.choice(moves))
        plies += 1
    if game.human_won():
        return -1
    if game.game_finished():
        return 1
    s1 = game.board.shortest_path_score(game.AI_player, game.human_player, game.AI_player.end_line)
    s2 = game.board.shortest_path_score(game.human_player, game.AI_player, game.human_player.end_line)
    return 1 if s1 <= s2 else -1


Game.simulate_gameV2 = _fast_sim

# row+1 is "down"; cell ids are our r*9+c
DELTA = {'up': -9, 'down': 9, 'left': -1, 'right': 1,
         'upup': -18, 'downdown': 18, 'leftleft': -2, 'rightright': 2,
         'upleft': -10, 'upright': -8, 'downleft': 8, 'downright': 10}


def run_mcts(root, no):
    random.seed(random.random())
    for _ in range(no):
        n = root
        while not n.game_finished() and len(n.children) != 0:
            n = n.select_random_child_with_best_UCB_score()
        if n.game_finished():
            win = not n.human_won()
            n.update_visits(win)
            continue
        if n.simulatedOnce:
            n.create_children()
            n = n.select_random_child_with_best_UCB_score()
        win = n.simulate_game() > 0          # simulate_gameV2: +1 iff AI (bottom) won
        n.update_visits(win)


def best_move(root):
    if root.game_finished():
        return None
    if not root.children:
        root.create_children()
    best = None
    for c in root.children:
        if c.total_games == 0:
            continue
        if best is None or c.win_games / c.total_games > best[0]:
            best = (c.win_games / c.total_games, c)
    if best is None:
        return root.children[0].move
    return best[1].move


def to_our_action(game, move, mir):
    if isinstance(move, tuple):                      # (row, col, HORIZONTAL|VERTICAL)
        r, c, o = move
        if mir:
            r, c = 7 - r, 7 - c
        return (81 + r * 8 + c) if o == HORIZONTAL else (145 + r * 8 + c)
    dest = game.AI_player.x * 9 + game.AI_player.y + DELTA[move]
    if mir:
        r, c = dest // 9, dest % 9
        dest = (8 - r) * 9 + (8 - c)
    return dest


def build(msg):
    mir = msg['player'] == 1
    game = Game(human_turn=False)
    game.human_turn = False
    hp, ap = game.human_player, game.AI_player           # hp human=True (top), ap AI (bottom)
    def pw(pos):
        return (8 - (pos // 9), 8 - (pos % 9)) if mir else (pos // 9, pos % 9)
    ap.x, ap.y = pw(msg['p0' if not mir else 'p1'])
    hp.x, hp.y = pw(msg['p1' if not mir else 'p0'])
    for s in msg['h']:
        r, c = s // 8, s % 8
        if mir:
            r, c = 7 - r, 7 - c
        game.board.use_wall(r, c, HORIZONTAL)
    for s in msg['v']:
        r, c = s // 8, s % 8
        if mir:
            r, c = 7 - r, 7 - c
        game.board.use_wall(r, c, VERTICAL)
    ap.no_walls = msg['w0'] if not mir else msg['w1']
    hp.no_walls = msg['w1'] if not mir else msg['w0']
    return game, mir


def main():
    for line in sys.stdin:
        msg = json.loads(line)
        if msg['type'] == 'hello':
            random.seed(msg.get('seed', 0))
            sys.stdout.write(json.dumps({'ok': True, 'name': 'berlioz', 'iter': ITER}) + '\n')
            sys.stdout.flush()
            continue
        if msg['type'] == 'bye':
            sys.exit(0)
        try:
            game, mir = build(msg)
            root = Node(None, game, None)
            run_mcts(root, ITER)
            move = best_move(root)
            if move is None:
                raise RuntimeError('game over at bridge state')
            a = to_our_action(game, move, mir)
            if not 0 <= a < ACTION_SIZE:
                a = fallback = None
                moves = [x for x in game.get_all_actions()
                         if not (isinstance(x, tuple) and (x[1] < 0 or x[1] > 7 or x[0] < 0 or x[0] > 7))]
                for m in moves:
                    cand = to_our_action(game, m, mir)
                    if 0 <= cand < ACTION_SIZE:
                        fallback = cand
                        break
                if fallback is not None:
                    a = fallback
                else:
                    raise RuntimeError('no convertible move')
            sys.stdout.write(json.dumps({'a': a}) + '\n')
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            sys.stdout.write(json.dumps({'forfeit': 'berlioz error: ' + str(e)}) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()

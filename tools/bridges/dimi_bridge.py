r"""Bridge for dimitrijekaranfilovic/quoridor (console MCTS, no weights).

Their GameState: player_one = BOTTOM pawn (win at row 0), player_two = TOP pawn
(win at row 16), 17x17 cell grid (pawns on even coords, wall slots at odd coords).
Their MCTS rollouts reward the TOP side (game_result(False)), so we always present
the board with the side to move as their player_two (top): mirror 180deg when we
must move for player 0, use directly when we must move for player 1.
Wall 6-tuples (r1,c1,r2,c2,r3,c3): h walls are (i, j, i, j+2, i, j+1) [i odd row
slot, j odd/even col]; v walls (i, j, i+2, j, i+1, j). Slot mapping matches ours
(h slot (r,c) blocks columns c..c+1, v slot blocks rows r..r+1).
"""
import json
import os
import sys

BOTS = os.environ.get('BOTS_DIR', r'C:\Users\kamil\Desktop\bots')
ROOT = os.path.join(BOTS, 'dimitrijekaranfilovic_quoridor')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.environ.get('QUDOR_REPO', r'C:\Users\kamil\Desktop\Qudor'))
from quoridor_ai.core.engine import State as OurState, legal_actions as our_legal  # noqa: E402

import numpy as np                                                     # noqa: E402
from console.algorithms.monte_carlo_tree_search import SearchNode      # noqa: E402
from console.states.game_state import BoardPieceStatus, GameState      # noqa: E402

OCC = BoardPieceStatus


def build_state(msg):
    mir = msg['player'] == 0
    st = GameState(is_simulation=True, initialize=True)
    st.player_one = False                       # player TWO (top) is to move
    def pc(pos):                                # our cell -> 17-grid cell coords
        r, c = pos // 9, pos % 9
        if mir:
            r, c = 8 - r, 8 - c
        return (2 * r, 2 * c)
    st.player_two_pos[0], st.player_two_pos[1] = pc(msg['p0' if mir else 'p1'])
    st.player_one_pos[0], st.player_one_pos[1] = pc(msg['p1' if mir else 'p0'])
    # walls: our slot (r,c) -> their 3 cells (h: row 2r+1, cols 2c+1..2c+3; v: col 2c+1, rows 2r+1..2r+3)
    for slot in msg['h']:
        r, c = slot // 8, slot % 8
        if mir:
            r, c = 7 - r, 7 - c
        for j in (2 * c + 1, 2 * c + 2, 2 * c + 3):
            st.board[(2 * r + 1) * 17 + j] = OCC.OCCUPIED_WALL
    for slot in msg['v']:
        r, c = slot // 8, slot % 8
        if mir:
            r, c = 7 - r, 7 - c
        for i in (2 * r + 1, 2 * r + 2, 2 * r + 3):
            st.board[i * 17 + (2 * c + 1)] = OCC.OCCUPIED_WALL
    # pawn cells
    st.board[st.player_one_pos[0] * 17 + st.player_one_pos[1]] = OCC.OCCUPIED_BY_PLAYER_1
    st.board[st.player_two_pos[0] * 17 + st.player_two_pos[1]] = OCC.OCCUPIED_BY_PLAYER_2
    st.player_one_walls_num = msg['w1' if mir else 'w0']
    st.player_two_wall_num = msg['w0' if mir else 'w1']
    return st, mir


def wall_to_our(positions, mir):
    r1, c1, r2, c2, r3, c3 = positions
    if r1 == r2 == r3:                          # horizontal
        r = (r1 - 1) // 2
        c = c1 // 2 if c1 % 2 == 0 else (c1 - 1) // 2
        kind = 81
    else:                                       # vertical
        r = (r1 - 1) // 2 if r1 % 2 == 1 else r1 // 2 - 1
        c = (c1 - 1) // 2
        kind = 145
    if mir:
        r, c = 7 - r, 7 - c
    return kind + r * 8 + c


def main():
    for line in sys.stdin:
        msg = json.loads(line)
        if msg['type'] == 'hello':
            np.random.seed(msg.get('seed', 0))
            sys.stdout.write(json.dumps({'ok': True, 'name': 'dimi'}) + '\n')
            sys.stdout.flush()
            continue
        if msg['type'] == 'bye':
            sys.exit(0)

        our_state = OurState(msg['p0'], msg['p1'], msg['w0'], msg['w1'],
                             sum(1 << sl for sl in msg['h']),
                             sum(1 << sl for sl in msg['v']),
                             msg['player'], msg['ply'])
        legal = sorted(our_legal(our_state))
        legal_set = set(legal)

        try:
            st, mir = build_state(msg)
            start = SearchNode(state=st, player_one_maximizer=False)
            selected = start.best_action()
            action = selected.parent_action
            if len(action) == 2:
                r, c = action
                if mir:
                    r, c = 16 - r, 16 - c
                a = (r // 2) * 9 + (c // 2)
            else:
                a = wall_to_our(action, mir)
            a = int(a)
            if a not in legal_set:
                a = None
        except Exception:
            a = None
        if a is None:
            a = int(legal[np.random.randint(len(legal))]) if legal else -1
        sys.stdout.write(json.dumps({'a': a}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()

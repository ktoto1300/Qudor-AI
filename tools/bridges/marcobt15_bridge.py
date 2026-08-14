"""Bridge for marcobt15/Quoridor_Reinforcement_Learning (MaskablePPO, 136 actions).

Reimplements their observation vector and action mask locally (no pettingzoo/pygame),
mirroring quoridor.py exactly:
  obs = [my(r,c), opp(r,c), walls.flatten(), my_walls_left, opp_walls_left, my_jump, opp_jump]
  walls.flatten(): (8,8,2) row-major -> flat = r*16 + c*2 + orientation (0=horizontal)
  agent player_1 = our p1 (top, (0,4)); agent player_2 = our p0 (bottom, (8,4)).
  player_2's obs is the 180-degree mirror (rows/cols/walls flipped) and its action ids
  live in mirror space (their step() swaps ids back to absolute).
  Pawn actions 0..7 are directions (0/4 up, 1/5 down, 2/6 left, 3/7 right); their env
  executes every one as a ONE-cell step (the jump actions never moved 2 cells).
  Wall ids in absolute space: h = 8+r*8+c, v = 72+r*8+c (row from top).
"""
import json
import os
import sys
from pathlib import Path

BOTS = Path(os.environ.get('BOTS_DIR', r'C:\Users\kamil\Desktop\bots'))
REPO = BOTS / 'marcobt15_Quoridor_Reinforcement_Learning'
sys.path.insert(0, os.environ.get('QUDOR_REPO', r'C:\Users\kamil\Desktop\Qudor'))

import numpy as np
from sb3_contrib import MaskablePPO

from quoridor_ai.core.engine import State, legal_actions

MODEL = os.environ.get('MARCOBT15_MODEL', 'quoridor_aec_v6_runs_and_walls.zip')

DIRS = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
JUMP = {(-2, 0): 4, (2, 0): 5, (0, -2): 6, (0, 2): 7}


def main():
    model = MaskablePPO.load(str(REPO / MODEL), device='cpu',
                             custom_objects={'clip_range': 0.2, 'lr_schedule': lambda _: 1e-4})
    rng = np.random.default_rng()

    for line in sys.stdin:
        msg = json.loads(line)
        if msg['type'] == 'hello':
            sys.stdout.write(json.dumps({'ok': True, 'name': 'marcobt15', 'model': MODEL}) + '\n')
            sys.stdout.flush()
            continue
        if msg['type'] == 'bye':
            break

        s = State(msg['p0'], msg['p1'], msg['w0'], msg['w1'],
                  sum(1 << sl for sl in msg['h']), sum(1 << sl for sl in msg['v']),
                  msg['player'], msg['ply'])
        mirrored = msg['player'] == 0                      # our p0 == their player_2

        if mirrored:
            me = (8 - s.p0 // 9, 8 - s.p0 % 9)
            opp = (8 - s.p1 // 9, 8 - s.p1 % 9)
        else:
            me = (s.p1 // 9, s.p1 % 9)
            opp = (s.p0 // 9, s.p0 % 9)
        walls = np.zeros((8, 8, 2), dtype=np.int16)
        m = s.h
        while m:
            b = m & -m
            sl = b.bit_length() - 1
            walls[sl // 8][sl % 8][0] = 1
            m ^= b
        m = s.v
        while m:
            b = m & -m
            sl = b.bit_length() - 1
            walls[sl // 8][sl % 8][1] = 1
            m ^= b
        walls = np.flip(walls, axis=(0, 1)) if mirrored else walls
        obs = np.concatenate([
            np.array(me, np.int16), np.array(opp, np.int16),
            walls.flatten().astype(np.int16),
            np.array([s.walls0 if mirrored else s.walls1,   # remaining walls of the agent
                      s.walls1 if mirrored else s.walls0], np.int16),
            np.array([1, 1], np.int16),                    # jump availability
        ]).astype(np.int16)

        legal = legal_actions(s)
        mask = np.zeros(136, dtype=np.int8)
        my_r, my_c = (s.p0 // 9, s.p0 % 9) if mirrored else (s.p1 // 9, s.p1 % 9)
        for a in legal:
            if a < 81:
                dr, dc = a // 9 - my_r, a % 9 - my_c
                if abs(dr) + abs(dc) not in (1, 2):
                    continue
                d0 = DIRS.get((dr, dc), JUMP.get((dr, dc), -1))
                if d0 < 0:
                    continue
                if mirrored:
                    d0 = d0 + 1 if d0 % 2 == 0 else d0 - 1
                mask[d0] = 1
                mask[4 + d0 % 4] = 1
            else:
                slot = a - 81 if a < 145 else a - 145
                wall_id = (8 + slot) if a < 145 else (72 + slot)
                if mirrored:
                    wall_id = (8 + 63 - slot) if a < 145 else (72 + 63 - slot)
                mask[wall_id] = 1

        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        act = int(act)
        if not mask[act]:
            ok = np.flatnonzero(mask)
            act = int(rng.choice(ok)) if len(ok) else -1
        if act < 8:
            d = act % 4
            nr, nc = (my_r - 1, my_c) if d == 0 else (my_r + 1, my_c) if d == 1 \
                else (my_r, my_c - 1) if d == 2 else (my_r, my_c + 1)
            out = nr * 9 + nc
        elif act < 72:
            r, c = divmod(act - 8, 8)
            out = 81 + (7 - r) * 8 + (7 - c) if mirrored else 81 + r * 8 + c
        else:
            r, c = divmod(act - 72, 8)
            out = 145 + (7 - r) * 8 + (7 - c) if mirrored else 145 + r * 8 + c

        legal_set = set(legal)
        if out not in legal_set:
            cand = []
            if act < 8:                                   # try the 2-cell version of the step
                d = act % 4
                dr, dc = (-2, 0) if d == 0 else (2, 0) if d == 1 else (0, -2) if d == 2 else (0, 2)
                cand.append((my_r + dr) * 9 + (my_c + dc))
                cand.append((my_r - dr) * 9 + (my_c - dc))
                cand.append((my_r + dc) * 9 + (my_c + dr))   # diagonals
                cand.append((my_r - dc) * 9 + (my_c - dr))
            cand += [a for a in legal if a < 81]
            cand += [a for a in legal if a >= 81]
            out = next((x for x in cand if x in legal_set), None)
            if out is None:
                out = int(rng.choice(list(legal_set))) if legal_set else -1
        sys.stdout.write(json.dumps({'a': out}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
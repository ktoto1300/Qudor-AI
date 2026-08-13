from quoridor_ai.core.engine import *
def test_initial():
 s=State();assert s.p0==76 and s.p1==4 and len(pawn_moves(s))==3 and len(legal_actions(s))>100
def test_jump():
 s=State(p0=40,p1=31);assert 22 in pawn_moves(s)
def test_diagonal():
 s=State(p0=40,p1=31,h=bit(2,4));assert {30,32}.issubset(pawn_moves(s))
def test_random_legality():
 import random
 s=State()
 for _ in range(300):
  if s.winner is not None:s=State()
  a=random.choice(legal_actions(s));s=apply_unchecked(s,a);assert has_path(s,0) and has_path(s,1)

def test_actions_never_leave_board():
 s=State()
 import random
 for _ in range(500):
  if s.winner is not None: s=State()
  actions=legal_actions(s)
  assert all(0 <= a < ACTION_SIZE for a in actions)
  assert all(0 <= a < 81 for a in actions if a < 81)
  s=apply_unchecked(s,random.choice(actions))
  assert 0 <= s.p0 < 81 and 0 <= s.p1 < 81

def test_each_player_can_place_at_most_ten_walls():
 s=State()
 placed=[0,0]
 while s.winner is None and s.ply < 200:
  actions=legal_actions(s); walls=[a for a in actions if a >= 81]
  if walls:
   player=s.player;s=apply_unchecked(s,walls[0]);placed[player]+=1
  else:
   s=apply_unchecked(s,pawn_moves(s)[0])
  assert 0 <= s.walls0 <= 10 and 0 <= s.walls1 <= 10
 assert placed[0] <= 10 and placed[1] <= 10

def test_wall_bit_indices_are_eight_by_eight():
 for r in range(8):
  for c in range(8):
   assert (bit(r,c).bit_length()-1) == r*8+c

def test_end_to_end_wall_chains_are_legal():
    # a horizontal wall may be extended end-to-end by another horizontal wall
    s = apply_unchecked(State(), 81 + 4*8 + 2)          # h in slot (4,2)
    plays = legal_actions(s)
    assert 81 + 4*8 + 1 in plays and 81 + 4*8 + 3 in plays
    # a vertical wall may be extended end-to-end by another vertical wall
    s = apply_unchecked(State(), 145 + 2*8 + 4)         # v in slot (2,4)
    plays = legal_actions(s)
    assert 145 + 1*8 + 4 in plays and 145 + 3*8 + 4 in plays

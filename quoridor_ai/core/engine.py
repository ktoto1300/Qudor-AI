from __future__ import annotations
from dataclasses import dataclass
N=9; ACTION_SIZE=209
@dataclass(slots=True)
class State:
    p0:int=76; p1:int=4; walls0:int=10; walls1:int=10; h:int=0; v:int=0; player:int=0; ply:int=0
    def copy(self): return State(self.p0,self.p1,self.walls0,self.walls1,self.h,self.v,self.player,self.ply)
    @property
    def winner(self):
        if self.p0//9==0:return 0
        if self.p1//9==8:return 1
        return None

def rc(x):return divmod(x,9)
def bit(r,c):return 1<<(r*8+c)

# --- precomputed edge tables -------------------------------------------------
# Both wall boards are packed into a single int w = h | v<<64, so testing an edge
# is one AND against a precomputed mask instead of divmod + bit arithmetic.
def _edge_mask(a,b):
 ar,ac=divmod(a,9);br,bc=divmod(b,9);m=0
 if ar==br:                       # sideways step: blocked by vertical walls
  c=min(ac,bc)
  if ar>0:m|=1<<(64+(ar-1)*8+c)
  if ar<8:m|=1<<(64+ar*8+c)
 else:                            # up/down step: blocked by horizontal walls
  r=min(ar,br)
  if ac>0:m|=1<<(r*8+ac-1)
  if ac<8:m|=1<<(r*8+ac)
 return m

# NB[x] = ((y,mask),...) in the original neighbour order: up, down, left, right.
NB=[]
for _x in range(81):
 _r,_c=divmod(_x,9);_e=[]
 for _dr,_dc in ((-1,0),(1,0),(0,-1),(0,1)):
  _rr,_cc=_r+_dr,_c+_dc
  if 0<=_rr<9 and 0<=_cc<9:_y=_rr*9+_cc;_e.append((_y,_edge_mask(_x,_y)))
 NB.append(tuple(_e))
NB=tuple(NB)
EM=[0]*6561                       # flat a*81+b -> mask, used to walk a path back
for _x in range(81):
 for _y,_m in NB[_x]:EM[_x*81+_y]=_m
del _x,_y,_m,_r,_c,_e,_dr,_dc,_rr,_cc

def blocked(a,b,s):return ((s.h|s.v<<64)&EM[a*81+b])!=0

def neighbors(x,s):
 w=s.h|s.v<<64
 for y,m in NB[x]:
  if not w&m:yield y

def pawn_moves(s,player=None):
    p=s.player if player is None else player; me=s.p0 if p==0 else s.p1; opp=s.p1 if p==0 else s.p0; out=set()
    mr,mc=rc(me);or_,oc=rc(opp)
    for q in neighbors(me,s):
        if q!=opp:out.add(q);continue
        dr,dc=or_-mr,oc-mc;rr,cc=or_+dr,oc+dc
        if 0<=rr<9 and 0<=cc<9 and not blocked(opp,rr*9+cc,s):out.add(rr*9+cc)
        else:
            for tr,tc in ((dc,dr),(-dc,-dr)):
                rr,cc=or_+tr,oc+tc
                if 0<=rr<9 and 0<=cc<9 and not blocked(opp,rr*9+cc,s):out.add(rr*9+cc)
    return sorted(out)

def _reach(w,start,goal):
 """True if `start` reaches row `goal` given packed walls `w`."""
 if start//9==goal:return True
 seen=1<<start;q=[start];i=0
 while i<len(q):
  for y,m in NB[q[i]]:
   if not(w&m or seen>>y&1):
    if y//9==goal:return True
    seen|=1<<y;q.append(y)
  i+=1
 return False

def _path_cut(w,start,goal):
 """Shortest path start -> row `goal`; returns the OR of every wall bit that
 would cross that path (so `cut & candidate == 0` proves the path survives),
 or None when no path exists."""
 if start//9==goal:return 0
 seen=1<<start;q=[start];par=[0]*81;i=0
 while i<len(q):
  x=q[i];i+=1
  for y,m in NB[x]:
   if not(w&m or seen>>y&1):
    par[y]=x
    if y//9==goal:
     cut=0
     while y!=start:p=par[y];cut|=EM[p*81+y];y=p
     return cut
    seen|=1<<y;q.append(y)
 return None

def has_path(s,player):return _reach(s.h|s.v<<64,s.p0 if player==0 else s.p1,0 if player==0 else 8)

_UNREACH=127                      # distance sentinel; fits in uint8 and survives /9 scaling

def dist_field(s,goal):
 """BFS distance from every cell to goal row, ignoring pawns (walls only).

 Returned as a list of 81 ints with _UNREACH where the goal is unreachable. This is
 the single most informative hand-crafted feature for Quoridor: the whole game is a
 race between two shortest-path lengths, and a conv net has to spend many layers
 rediscovering it from raw wall planes.
 """
 w=s.h|s.v<<64;d=[_UNREACH]*81;q=[]
 for c in range(9):
  x=goal*9+c;d[x]=0;q.append(x)
 i=0
 while i<len(q):
  x=q[i];i+=1;nd=d[x]+1
  for y,m in NB[x]:
   if d[y]>nd and not w&m:d[y]=nd;q.append(y)
 return d

def dist_to_goal(s,player):
 """Shortest path length for `player`, or _UNREACH when walled off entirely."""
 return dist_field(s,0 if player==0 else 8)[s.p0 if player==0 else s.p1]

def can_wall(s,o,r,c):
 if not(0<=r<8 and 0<=c<8) or (s.walls0 if s.player==0 else s.walls1)<=0:return False
 b=bit(r,c)
 if (s.h|s.v)&b:return False
 if o=='h':
  if (c>0 and s.h&bit(r,c-1)) or (c<7 and s.h&bit(r,c+1)):return False
  w=(s.h|b)|s.v<<64
 else:
  if (r>0 and s.v&bit(r-1,c)) or (r<7 and s.v&bit(r+1,c)):return False
  w=s.h|(s.v|b)<<64
 return _reach(w,s.p0,0) and _reach(w,s.p1,8)

_PCAP=16                          # alternative paths kept per player, per call

def _safe(w,bw,pool,start,goal):
 """Does the wall `bw` leave `start` connected to row `goal`? A candidate that
 misses any known-good path is provably safe, so BFS only runs when the wall
 crosses every path in the pool. Each forced BFS returns a path that avoids the
 wall which triggered it, so the pool diversifies on its own."""
 for cut in pool:
  if not bw&cut:return True
 cut=_path_cut(w|bw,start,goal)
 if cut is None:return False
 # A path valid with the extra wall is valid without it, so it is safe to cache.
 if len(pool)<_PCAP:pool.append(cut)
 else:pool.pop(1);pool.append(cut)   # keep slot 0 (the true shortest path), FIFO the rest
 return True

def legal_actions(s):
 if s.winner is not None:return []
 a=pawn_moves(s)
 if (s.walls0 if s.player==0 else s.walls1)<=0:return a
 h=s.h;v=s.v;hv=h|v;w=h|v<<64;p0=s.p0;p1=s.p1
 c0=_path_cut(w,p0,0);c1=_path_cut(w,p1,8)
 if c0 is None or c1 is None:return a   # already cut off: no wall can be legal
 P0=[c0];P1=[c1]
 for r in range(8):
  for c in range(8):
   b=1<<(r*8+c)
   if hv&b:continue                     # slot taken by either orientation
   if not((c>0 and h&b>>1)or(c<7 and h&b<<1)):
    if _safe(w,b,P0,p0,0) and _safe(w,b,P1,p1,8):a.append(81+r*8+c)
   if not((r>0 and v&b>>8)or(r<7 and v&b<<8)):
    bv=b<<64
    if _safe(w,bv,P0,p0,0) and _safe(w,bv,P1,p1,8):a.append(145+r*8+c)
 return a

def apply_unchecked(s,a):
    n=s.copy();p=s.player
    if a<81:
        if p==0:n.p0=a
        else:n.p1=a
    elif a<145:
        x=a-81;n.h|=bit(x//8,x%8);n.walls0-=p==0;n.walls1-=p==1
    else:
        x=a-145;n.v|=bit(x//8,x%8);n.walls0-=p==0;n.walls1-=p==1
    n.player=1-p;n.ply+=1;return n

def apply(s,a):
    if a not in legal_actions(s):raise ValueError(a)
    return apply_unchecked(s,a)

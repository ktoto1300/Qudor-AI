import argparse,torch,random,json,math
from pathlib import Path
import numpy as np
from .model import PolicyValueNet,net_from_checkpoint
from .core.engine import State,legal_actions,apply_unchecked
from .core.encoding import encode_batch,version_for_planes
from .safe_loader import load_checkpoint

def load(path,device):
 return net_from_checkpoint(load_checkpoint(path,map_location=device),device)
def choose(m,s,d,temp=0.0,rng=None):
 """Pick a legal action. temp=0 is argmax; temp>0 samples from softmax(logits/temp).

 Sampling matters for the arena: with pure argmax both nets are deterministic, so all
 games are byte-identical and the measured win rate carries no information.
 """
 a=legal_actions(s);x=torch.from_numpy(encode_batch([s],version_for_planes(m.planes))).to(d)
 with torch.inference_mode():z,_=m(x)
 logits=z[0,a]
 if temp<=0:return a[int(logits.argmax())]
 p=torch.softmax(logits.float()/temp,0).cpu().numpy()
 return a[int((rng or np.random).choice(len(a),p=p/p.sum()))]
def play(a,b,d,swap=False,temp=0.35,rng=None,max_plies=220):
 s=State()
 while s.winner is None and s.ply<max_plies:
  m=(a if s.player==0 else b) if not swap else (b if s.player==0 else a);s=apply_unchecked(s,choose(m,s,d,temp,rng))
 if s.winner is None:return .5
 winner_a=(s.winner==0) != swap;return 1. if winner_a else 0.
def run(candidate,best,games,out,temp=0.35,seed=0):
 d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a=load(candidate,d);b=load(best,d)
 rng=np.random.default_rng(seed)
 scores=[play(a,b,d,i%2==1,temp,rng) for i in range(games)];wr=sum(scores)/games;elo=400*math.log10(max(1e-4,wr)/max(1e-4,1-wr));r={'games':games,'wins':sum(x==1 for x in scores),'draws':sum(x==.5 for x in scores),'win_rate':wr,'elo_delta':elo,'temperature':temp,'seed':seed};Path(out).write_text(json.dumps(r,indent=2));print(r);return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--best',required=True);p.add_argument('--games',type=int,default=40);p.add_argument('--output',default='arena.json');p.add_argument('--temp',type=float,default=0.35);p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.candidate,a.best,a.games,a.output,a.temp,a.seed)
if __name__=='__main__':main()

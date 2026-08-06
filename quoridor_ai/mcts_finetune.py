import argparse,json,random,torch
from pathlib import Path
import numpy as np
from .model import net_from_checkpoint
from .core.engine import State,apply_unchecked
from .core.encoding import encode,version_for_planes
from .batched_mcts import batched_search
from .safe_loader import load_checkpoint

def run(config,checkpoint,output,games=16,sims=64,steps=100):
 c=json.load(open(config,encoding='utf-8'));d=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 # Shape the net from the checkpoint, not from the config: a config edited after training
 # would otherwise build a net whose state_dict does not match.
 net=net_from_checkpoint(load_checkpoint(checkpoint,map_location=d),d);enc=version_for_planes(net.planes);net.train()
 states=[State() for _ in range(games)];trajectories=[[] for _ in states];done=[False]*games
 while not all(done):
  ids=[i for i,x in enumerate(states) if not done[i]];pis=batched_search(net,[states[i] for i in ids],d,sims=sims,encoding=enc)
  for j,i in enumerate(ids):
   pi=pis[j]
   if pi.sum()<=0:done[i]=True;continue  # no visits => terminal or no legal move
   a=int(np.random.choice(len(pi),p=pi));trajectories[i].append((encode(states[i],enc),pi,states[i].player));states[i]=apply_unchecked(states[i],a);done[i]=states[i].winner is not None or states[i].ply>=c['max_plies']
  print(f'mcts active={len(ids)} total_positions={sum(map(len,trajectories))}',flush=True)
 data=[]
 for s,tr in zip(states,trajectories):data += [(x,pi,0 if s.winner is None else (1 if s.winner==pl else -1)) for x,pi,pl in tr]
 if not data:raise RuntimeError('mcts finetune produced no positions')
 opt=torch.optim.AdamW(net.parameters(),lr=c['lr']*.25)
 for _ in range(steps):
  b=random.sample(data,min(c['batch'],len(data)));x=torch.from_numpy(np.stack([q[0] for q in b])).to(d);pi=torch.from_numpy(np.stack([q[1] for q in b])).to(d);z=torch.tensor([q[2] for q in b],dtype=torch.float32,device=d);logits,v=net(x);loss=-(pi*torch.log_softmax(logits,1)).sum(1).mean()+torch.nn.functional.mse_loss(v,z);opt.zero_grad();loss.backward();opt.step()
 Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save({'model':net.state_dict(),'config':c,'encoding':enc,'stage':'mcts','positions':len(data)},output);print(f'mcts finetune saved {output}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--games',type=int,default=16);p.add_argument('--sims',type=int,default=64);p.add_argument('--steps',type=int,default=100);a=p.parse_args();run(a.config,a.checkpoint,a.output,a.games,a.sims,a.steps)
if __name__=='__main__':main()

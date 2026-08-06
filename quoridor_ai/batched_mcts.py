import math,numpy as np,torch
from .core.engine import legal_actions,apply_unchecked,ACTION_SIZE
from .core.encoding import encode_batch
class Node:
 def __init__(self,s,p=1.):self.s=s;self.p=p;self.n=0;self.w=0.;self.children={}
 @property
 def q(self):return self.w/self.n if self.n else 0.
def _expand(net,nodes,device,encoding=1):
 """Batch-evaluate nodes, attach priors as children, return {id(node): value}.

 Terminal nodes (and any node with no legal action) are dropped before the batch:
 softmax over an empty action list raises, so they must never reach the network.
 Their value is decided by the caller from the game result instead.
 """
 live=[n for n in nodes if n.s.winner is None and legal_actions(n.s)]
 if not live:return {}
 x=torch.from_numpy(encode_batch([n.s for n in live],encoding)).to(device)
 with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,values=net(x)
 vals=values.float().cpu().numpy();out={}
 for i,n in enumerate(live):
  acts=legal_actions(n.s);z=logits[i,acts].float().cpu().numpy();z-=z.max();p=np.exp(z);p/=p.sum()
  n.children={a:Node(apply_unchecked(n.s,a),float(v)) for a,v in zip(acts,p)};out[id(n)]=float(vals[i])
 return out
def batched_search(net,states,device,sims=64,c_puct=1.5,encoding=1):
 """Batched MCTS with PUCT selection, one tree per state.

 encoding: encoder version (1 or 2) matching the network's input planes; default 1 for
           backward compatibility with checkpoints trained before the encoder was versioned.
 """
 roots=[Node(s) for s in states];_expand(net,roots,device,encoding)
 for _ in range(sims):
  paths=[]
  for root in roots:
   # The root belongs in the path: it needs its visit count incremented too, otherwise
   # the exploration term sqrt(parent.n+1) stays pinned at 1 for every root child.
   path=[root];n=root
   while n.children:
    _,n=max(n.children.items(),key=lambda kv:-kv[1].q+c_puct*kv[1].p*math.sqrt(n.n+1)/(1+kv[1].n));path.append(n)
   paths.append(path)
  vals=_expand(net,[p[-1] for p in paths],device,encoding)
  for path in paths:
   leaf=path[-1]
   # Terminal leaf: the side to move there has already lost, so -1 from its perspective.
   v=-1. if leaf.s.winner is not None else vals.get(id(leaf),0.)
   for node in reversed(path):node.n+=1;node.w+=v;v=-v
 out=[]
 for root in roots:
  pi=np.zeros(ACTION_SIZE,np.float32)
  for a,n in root.children.items():pi[a]=n.n
  if pi.sum():pi/=pi.sum()
  out.append(pi)
 return np.stack(out)

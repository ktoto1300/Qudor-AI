import math,numpy as np,torch
from .core.engine import legal_actions,apply_unchecked,ACTION_SIZE
from .core.encoding import encode_batch
class Node:
 def __init__(self,s,prior=1):self.s=s;self.prior=prior;self.n=0;self.w=0.;self.children={}
 @property
 def q(self):return self.w/self.n if self.n else 0

def search(net,state,device,sims=64,c_puct=1.5,noise=True):
 root=Node(state);_expand(net,[root],device)
 if noise and root.children:
  keys=list(root.children);d=np.random.dirichlet([.3]*len(keys))
  for k,x in zip(keys,d,strict=True):root.children[k].prior=.75*root.children[k].prior+.25*x
 for _ in range(sims):
  n=root;path=[]
  while n.children:
   a,ch=max(n.children.items(),key=lambda kv:-kv[1].q+c_puct*kv[1].prior*math.sqrt(n.n+1)/(1+kv[1].n));path.append(n);n=ch
  if n.s.winner is not None:v=-1.
  else:v=_expand(net,[n],device)[0]
  n.n+=1;n.w+=v
  for p in reversed(path):v=-v;p.n+=1;p.w+=v
 pi=np.zeros(ACTION_SIZE,np.float32)
 for a,ch in root.children.items():pi[a]=ch.n
 if pi.sum():pi/=pi.sum()
 return pi

def _expand(net,nodes,device):
 ss=[n.s for n in nodes];x=torch.from_numpy(encode_batch(ss)).to(device)
 with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,vals=net(x)
 for i,n in enumerate(nodes):
  acts=legal_actions(n.s);z=logits[i,acts].float().cpu().numpy();z-=z.max();p=np.exp(z);p/=p.sum()
  n.children={a:Node(apply_unchecked(n.s,a),float(pr)) for a,pr in zip(acts,p,strict=True)}
 return vals.float().cpu().tolist()

import numpy as np,torch,time
from .core.engine import State,ACTION_SIZE,legal_actions,apply_unchecked
from .core.encoding import encode_batch

def batched_selfplay(net,device,games=64,max_plies=220,batch_cap=256,encoding=1):
 """Self-play with network policy rollouts, temperature-sampled.

 encoding: encoder version (1 or 2) matching the network's input planes; default 1 for
           backward compatibility with existing checkpoints.
 """
 states=[State() for _ in range(games)];tr=[[] for _ in states];done=[False]*games;t=time.time()
 net.eval()
 while not all(done):
  ids=[i for i,s in enumerate(states) if not done[i]]
  for off in range(0,len(ids),batch_cap):
   chunk=ids[off:off+batch_cap];ss=[states[i] for i in chunk];acts=[legal_actions(s) for s in ss]
   x=torch.from_numpy(encode_batch(ss,encoding)).to(device,non_blocking=True,memory_format=torch.channels_last)
   with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,_=net(x)
   for j,i in enumerate(chunk):
    a=acts[j];v=logits[j,a].float().cpu().numpy();temp=1 if states[i].ply<20 else .25;v=v/max(temp,.05);v-=v.max();p=np.exp(v);p/=p.sum();choice=int(np.random.choice(a,p=p));pi=np.zeros(ACTION_SIZE,np.float16);pi[a]=p.astype(np.float16);tr[i].append((encode_batch([states[i]],encoding)[0].astype(np.float16),pi,states[i].player));states[i]=apply_unchecked(states[i],choice)
    if states[i].winner is not None or states[i].ply>=max_plies:done[i]=True
 data=[]
 for i,s in enumerate(states):
  for x,pi,pl in tr[i]:data.append((x,pi,0 if s.winner is None else (1 if s.winner==pl else -1)))
 return data,{'games':games,'positions':len(data),'seconds':time.time()-t,'avg_length':sum(s.ply for s in states)/games}

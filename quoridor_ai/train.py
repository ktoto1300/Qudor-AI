import argparse,csv,json,random,time,os
from pathlib import Path
from collections import deque
import numpy as np,torch
from .model import PolicyValueNet,planes_of
from .core.encoding import PLANES_BY_VERSION,DEFAULT_VERSION
from .selfplay import batched_selfplay
from .safe_loader import load_checkpoint

def _encoding_of(d):
 """Encoder version a checkpoint was written with.

 Checkpoints from before the encoder was versioned carry no 'encoding' key, so the
 version is recovered from the stem conv's input channels.
 """
 if 'encoding' in d:return int(d['encoding'])
 planes=planes_of(d['model'])
 for v,n in PLANES_BY_VERSION.items():
  if n==planes:return v
 raise ValueError(f'checkpoint has {planes} input planes, no known encoder produces that')

def run(config,output,resume=True,init=None):
 c=json.load(open(config,encoding='utf-8'));seed=c.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 enc=int(c.get('encoding',DEFAULT_VERSION))
 if enc not in PLANES_BY_VERSION:raise ValueError(f'config encoding={enc} unknown; known: {sorted(PLANES_BY_VERSION)}')
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path(output);out.mkdir(parents=True,exist_ok=True)
 ck=out/'latest.pt';start=0
 # The checkpoint being continued decides the encoding: its weights and its stored replay
 # buffer are both in that layout, so the config cannot override it mid-run.
 if resume and ck.exists():
  d=load_checkpoint(ck,map_location=device);found=_encoding_of(d)
  if found!=enc:print(f'config encoding={enc} ignored; resuming checkpoint uses v{found}',flush=True);enc=found
 elif init and Path(init).exists():
  d=load_checkpoint(init,map_location=device);found=_encoding_of(d)
  if found!=enc:raise ValueError(f'--init checkpoint uses encoding v{found} but config asks for v{enc}; they are not interchangeable')
 else:d=None
 net=PolicyValueNet(c['channels'],c['blocks'],PLANES_BY_VERSION[enc]).to(device,memory_format=torch.channels_last);opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');replay=deque(maxlen=c['replay'])
 if resume and ck.exists():
  net.load_state_dict(d['model']);opt.load_state_dict(d['optimizer'])
  sc=d.get('scaler')
  if sc:scaler.load_state_dict(sc)  # GradScaler rejects an empty dict, so only load a real one.
  replay.extend(d.get('replay',[]));start=d['iteration']+1
  # Re-seed off the resume point: keeping the original seed would replay the exact
  # same self-play games that were already in the buffer before the restart.
  random.seed(seed+start);np.random.seed((seed+start)%(2**32));torch.manual_seed(seed+start)
 elif d is not None:
  net.load_state_dict(d['model']);print(f'initialized from {init}',flush=True)
 metrics=out/'metrics.csv'
 if not metrics.exists():metrics.write_text('iteration,stage,games,positions,replay,games_per_sec,positions_per_sec,avg_length,policy_loss,value_loss,total_loss,seconds,device,vram_mb\n')
 for it in range(start,c['iterations']):
  t=time.time();data,sp=batched_selfplay(net,device,c['games'],c['max_plies'],c['inference_batch'],enc);replay.extend(data)
  net.train();pl=vl=0.;steps=c['steps']
  # Snapshot once per iteration: list(deque) is O(len(replay)) and was previously
  # rebuilt on every one of the `steps` minibatches (500k copies × 80 steps).
  pool=list(replay);pool_n=len(pool)
  for _ in range(steps):
   idx=random.sample(range(pool_n),min(c['batch'],pool_n));batch=[pool[i] for i in idx];x=torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device,memory_format=torch.channels_last);pi=torch.from_numpy(np.stack([b[1] for b in batch])).float().to(device);z=torch.tensor([b[2] for b in batch],dtype=torch.float32,device=device)
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,val=net(x);lp=-(pi*torch.log_softmax(logits,1)).sum(1).mean();lv=torch.nn.functional.mse_loss(val,z);loss=lp+lv
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),5);scaler.step(opt);scaler.update();pl+=lp.item();vl+=lv.item()
  sec=time.time()-t;vram=torch.cuda.max_memory_allocated()/1048576 if device.type=='cuda' else 0;row=[it,'pretrain',sp['games'],sp['positions'],len(replay),sp['games']/sp['seconds'],sp['positions']/sp['seconds'],sp['avg_length'],pl/steps,vl/steps,(pl+vl)/steps,sec,str(device),vram]
  with metrics.open('a',newline='') as f:csv.writer(f).writerow(row)
  payload={'iteration':it,'model':net.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'replay':list(replay)[-c['checkpoint_replay']:],'config':c,'encoding':enc};tmp=out/'latest.tmp';torch.save(payload,tmp);os.replace(tmp,ck);(out/'status.json').write_text(json.dumps(dict(zip(metrics.read_text().splitlines()[0].split(','),row)),indent=2));print(f"iteration={it} games/s={row[5]:.2f} pos/s={row[6]:.1f} loss={row[10]:.4f} device={device} vram={vram:.0f}MB",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--init');p.add_argument('--no-resume',action='store_true');a=p.parse_args();run(a.config,a.output,not a.no_resume,a.init)
if __name__=='__main__':main()

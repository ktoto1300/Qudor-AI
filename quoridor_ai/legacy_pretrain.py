import argparse,torch,random,json
from pathlib import Path
import numpy as np
from .model import PolicyValueNet
from .core.encoding import PLANES_BY_VERSION
from .safe_loader import load_checkpoint

LEGACY_PLANES=7

def load_legacy(folder,limit=15000):
 """Load 7-plane legacy replays and lift them into the 11-plane v1 layout.

 The legacy dumps only carry planes 0..6. Planes 8..10 of v1 are recoverable — plane 8
 is 1-player and player is the constant plane 2, planes 9/10 are fixed goal rows — so
 they are reconstructed rather than left as zeros; feeding the net a plane pattern that
 encode_v1 never produces would put pretraining off-distribution. Plane 7 (ply/200) is
 genuinely unknown here and stays zero.
 """
 planes=PLANES_BY_VERSION[1];data=[]
 for p in sorted(Path(folder).rglob('*_legacy.pt')):
  d=load_checkpoint(p,map_location='cpu')
  for x,pi,z in d.get('replay',[]):
   src=np.asarray(x,dtype=np.float32)
   if src.shape!=(LEGACY_PLANES,9,9):continue
   y=np.zeros((planes,9,9),np.float32);y[:LEGACY_PLANES]=src
   y[8]=1.-y[2];y[9,0]=1;y[10,8]=1
   data.append((y,np.asarray(pi,dtype=np.float32),float(z)))
   if len(data)>=limit:return data
 return data

def run(folder,output,channels=96,blocks=8,epochs=5,batch=512):
 data=load_legacy(folder);assert data,'No legacy replay found';device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=PolicyValueNet(channels,blocks,PLANES_BY_VERSION[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4);net.train()
 for ep in range(epochs):
  random.shuffle(data);pl=vl=0;n=0
  for i in range(0,len(data),batch):
   b=data[i:i+batch];x=torch.from_numpy(np.stack([q[0] for q in b])).to(device);pi=torch.from_numpy(np.stack([q[1] for q in b])).to(device);z=torch.tensor([q[2] for q in b],dtype=torch.float32,device=device)
   with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):logits,v=net(x);lp=-(pi*torch.log_softmax(logits,1)).sum(1).mean();lv=torch.nn.functional.mse_loss(v,z);loss=lp+lv
   opt.zero_grad();loss.backward();opt.step();pl+=lp.item();vl+=lv.item();n+=1
  print(f'legacy epoch={ep} policy={pl/n:.4f} value={vl/n:.4f}',flush=True)
 out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':net.state_dict(),'legacy_samples':len(data),'channels':channels,'blocks':blocks,'encoding':1},out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy',default='legacy');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch',type=int,default=512);a=p.parse_args();run(a.legacy,a.output,epochs=a.epochs,batch=a.batch)
if __name__=='__main__':main()

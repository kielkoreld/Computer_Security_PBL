"""공식 모델 손실, 독립 경로 배치, 정규화 LUT의 정확 복원."""
import collections,pickle,random,sys,time
import numpy as np
import torch
from paths import Q,R,S,read
import model_builder as original

def seed():
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def build():return original.build()
def reset(model):original.reset(model)
def optimizer(model):
 return torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('img_backbone')],'lr':2e-6},{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('img_backbone')],'lr':2e-5}],weight_decay=.001)

def groups(eps,batch,epoch=0):
 bylen=collections.defaultdict(list)
 for e in eps:bylen[read(Q/'data'/e/'complete.json')['counts']['frames']].append(e)
 out=[];rng=random.Random(epoch)
 for length in sorted(bylen):
  rows=sorted(bylen[length])
  if epoch:rng.shuffle(rows)
  i=0
  while i<len(rows):
   size=min(batch,2**int((len(rows)-i).bit_length()-1))
   out.append(rows[i:i+size]);i+=size
 if epoch:rng.shuffle(out)
 assert len([e for g in out for e in g])==len(eps) and set(e for g in out for e in g)==set(eps)
 return out

class Group:
 def __init__(self,eps):
  self.eps=eps;self.rgb=[];self.meta=[];self.target=[]
  for e in eps:
   d=Q/'cache'/e;assert read(d/'complete.json')['complete']
   self.rgb.append(np.load(d/'rgb.npy',mmap_mode='r'))
   with (d/'meta.pkl').open('rb') as f:self.meta.append(pickle.load(f))
   with (Q/'data'/e/'targets.pkl').open('rb') as f:self.target.append(pickle.load(f))
  self.length=len(self.meta[0]);assert all(len(m)==self.length for m in self.meta)
  for e,m,t,r in zip(eps,self.meta,self.target,self.rgb):
   assert len(t)==len(r)==self.length and all(x['token'].startswith(e+'/') for x in m)
   assert all(abs(b['timestamp']-a['timestamp']-.5)<1e-6 for a,b in zip(m,m[1:]))
 def cpu(self,i):
  rgb=torch.from_numpy(np.stack([x[i] for x in self.rgb])).pin_memory()
  return rgb,[x[i] for x in self.meta],[x[i] for x in self.target]

class Transfer:
 def __init__(self):
  self.lut=torch.from_numpy(np.load(Q/'cache/normalization_lut.npy')).cuda()
  self.channel=torch.arange(3,device='cuda')[None,None,:,None,None]
 def __call__(self,item):
  raw,meta,target=item;raw=raw.cuda(non_blocking=True)
  im=self.lut[self.channel,raw.long()]
  data=dict(img=im,projection_mat=torch.from_numpy(np.stack([m['projection_mat'] for m in meta])).cuda(),image_wh=torch.from_numpy(np.stack([m['image_wh'] for m in meta])).cuda(),timestamp=torch.tensor([m['timestamp'] for m in meta],dtype=torch.float64,device='cuda'),img_metas=[dict(T_global=m['T_global'],T_global_inv=m['T_global_inv'],instance_id=t['ids']) for m,t in zip(meta,target)],gt_bboxes_3d=[torch.from_numpy(t['boxes']).cuda() for t in target],gt_labels_3d=[torch.from_numpy(t['labels']).cuda() for t in target])
  return data

def losses(model,data):
 parts=model.forward_train(**data);keys=list(parts);v=torch.stack([parts[k].mean() for k in keys]);total=sum(parts[k].mean() for k in keys if 'loss' in k)
 if not torch.isfinite(total):raise FloatingPointError('비유한 학습/검증 손실')
 return total,keys,v

def train_group(model,group,optim,transfer,limit=None,threads=1):
 import concurrent.futures
 train=optim is not None;model.train(train);model.depth_branch.eval();reset(model)
 n=group.length if limit is None else min(group.length,limit);batch=len(group.eps);frames=0;total=0.;parts_sum={};pending=0;times=[]
 if train:optim.zero_grad(set_to_none=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
  future=pool.submit(group.cpu,0)
  for i in range(n):
   t=time.perf_counter();item=future.result()
   if i+1<n:future=pool.submit(group.cpu,i+1)
   data=transfer(item)
   with torch.set_grad_enabled(train):value,keys,parts=losses(model,data)
   if train:
    (value*batch/8).backward();pending+=batch
    if pending>=8:
     if pending!=8:
      for p in model.parameters():
       if p.grad is not None:p.grad.mul_(8/pending)
     torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],25.,error_if_nonfinite=True);optim.step();optim.zero_grad(set_to_none=True);pending=0
   # 첫 프레임의 속도 미상/빈 GT에서는 DN 항이 없을 수 있다.
   for k,v in zip(keys,parts.detach()*batch):parts_sum[k]=parts_sum.get(k,0.)+v
   total+=float(value.detach())*batch;frames+=batch
   torch.cuda.synchronize();times.append(time.perf_counter()-t)
   del item,data,value,parts
 if train and pending:
  for p in model.parameters():
   if p.grad is not None:p.grad.mul_(8/pending)
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],25.,error_if_nonfinite=True);optim.step();optim.zero_grad(set_to_none=True)
 component={k:float(v)/frames for k,v in zip(parts_sum,torch.stack(list(parts_sum.values())).cpu())};reset(model)
 return dict(loss=total/frames,frames=frames,components=component,elapsed_s=sum(times),frame_s=sum(times)/frames,steps=n,batch=batch)

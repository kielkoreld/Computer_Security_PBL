"""실제10Hz 시퀀스 학습. 배치 행 제거 시 시간 상태도 같은 행만 보존한다."""
import collections,concurrent.futures,contextlib,pickle,random,sys,time,types
import numpy as np
import torch
from common import *
from cache_io import RGB
sys.path.insert(0,str(R/'runtime/Sparse4D-main'))
from mmcv import Config
import projects.mmdet3d_plugin
from mmdet.models import build_detector
from model_inputs import CLASS_NAMES
COMMON=['car','truck','bus','pedestrian','bicycle','motorcycle']
def seed(n=0,precision='fp32'):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
 torch.backends.cuda.matmul.allow_tf32=precision=='tf32';torch.backends.cudnn.allow_tf32=precision=='tf32';torch.backends.cudnn.benchmark=False
def reset(m):m.head.instance_bank.reset();m.head.sampler.dn_metas=None
class CommonFocal(torch.nn.Module):
 def __init__(self,base):super().__init__();self.original=base
 def forward(self,pred,target,weight=None,**kw):
  mask=pred.new_tensor([float(c in COMMON) for c in CLASS_NAMES]).expand_as(pred)
  if weight is not None:mask=mask*weight.reshape(-1,1) if weight.ndim==1 else mask*weight
  return self.original(pred,target,weight=mask,**kw)
def build(checkpoint=INITIAL,precision='fp32',default_dt=.1):
 cfg=Config.fromfile(str(R/'runtime/Sparse4D-main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
 cfg.model.img_backbone.pretrained=None;cfg.model.head.instance_bank.anchor=str(R/'artifacts/nuscenes_kmeans900_from_checkpoint.npy');cfg.model.head.instance_bank.default_time_interval=default_dt
 m=build_detector(cfg.model);m.head.loss_cls=CommonFocal(m.head.loss_cls)
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ck['state_dict'],strict=True);del ck
 # 공식 입력은 NaN velocity를 0으로 치환하지만 이 프로젝트는 미상 성분을
 # 손실에서 제외한다. NaN*0은 NaN이므로 cost 계산 전에 값을 제거해야 한다.
 install_finite_target_cost(m.head.sampler)
 for p in m.depth_branch.parameters():p.requires_grad_(False)
 base=m.head.sampler.get_dn_anchors
 def finite_dn(labels,boxes,ids=None):
  masks=[torch.isfinite(b).all(-1) for b in boxes]
  return base([x[t] for x,t in zip(labels,masks)],[x[t] for x,t in zip(boxes,masks)],None if ids is None else [x[t] for x,t in zip(ids,masks)])
 m.head.sampler.get_dn_anchors=finite_dn
 for op,layer in zip(m.head.operation_order,m.head.layers):
  if op=='refine':
   def cls(*args,_base=layer.forward,**kw):kw['return_cls']=True;return _base(*args,**kw)
   layer.forward=cls
 extract=m.extract_feat
 def tensor32(x):
  if isinstance(x,torch.Tensor):return x.float() if x.is_floating_point() else x
  if isinstance(x,(tuple,list)):return type(x)(tensor32(y) for y in x)
  return x
 def extract_new(img,return_depth=False,metas=None):
  # 깊이 정답이 없는 기존 실험. 사용되지 않는 depth 계산만 생략한다.
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=m._precision=='bf16_backbone'):
   feat=extract(img,False,metas)
  feat=tensor32(feat)
  return (feat,None) if return_depth else feat
 m.extract_feat=extract_new;m._precision=precision;m._train_precision=precision
 return m.cuda(),cfg
def install_finite_target_cost(sampler):
 base=sampler._box_cost
 def masked(box_pred,box_target,instance_reg_weights):
  finite=[torch.isfinite(t) for t in box_target]
  safe=[torch.where(f,t,torch.zeros_like(t)) for t,f in zip(box_target,finite)]
  weights=[w*f.to(w.dtype) for w,f in zip(instance_reg_weights,finite)]
  return base(box_pred,safe,weights)
 sampler._box_cost=masked
def optimizer(m,lr=5e-6):
 return torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('img_backbone')],'lr':lr*.1},{'params':[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('img_backbone')],'lr':lr}],weight_decay=.001)
def groups(eps,batch=4,epoch=0):
 counts={x['episode']:x['frames'] for x in read(P/'protocol/inventory.json')['episodes']};rows={e:counts[e] for e in eps}
 rng=random.Random(epoch);order=list(eps)
 if epoch:rng.shuffle(order)
 order.sort(key=lambda e:rows[e]);gs=[order[i:i+batch] for i in range(0,len(order),batch)]
 if epoch:rng.shuffle(gs)
 assert len(order)==len(set(order))==len(eps)
 return gs
class Group:
 def __init__(self,eps):
  self.eps=eps;self.rgb=[];self.meta=[];self.targets=[]
  for e in eps:
   d=P/'cache'/e;assert read(d/'complete.json')['complete'];self.rgb.append(RGB(e))
   with (d/'meta.pkl').open('rb') as f:self.meta.append(pickle.load(f))
   with (d/'targets.pkl').open('rb') as f:self.targets.append(pickle.load(f))
  self.lengths=list(map(len,self.meta));self.length=max(self.lengths)
  for e,rs,ms,ts in zip(eps,self.rgb,self.meta,self.targets):
   assert len(rs)==len(ms)==len(ts) and all(m['token']==e+f'/{i:05d}' for i,m in enumerate(ms))
   assert all(abs(b['timestamp']-a['timestamp']-.1)<1e-7 for a,b in zip(ms,ms[1:]))
 def cpu(self,i):
  active=[j for j,n in enumerate(self.lengths) if i<n]
  return (torch.from_numpy(np.stack([self.rgb[j][i] for j in active])).pin_memory(),[self.meta[j][i] for j in active],[self.targets[j][i] for j in active],active)
 def close(self):
  for r in self.rgb:r.close()
class Transfer:
 def __init__(self):self.lut=torch.from_numpy(np.load(Q/'cache/normalization_lut.npy')).cuda();self.ch=torch.arange(3,device='cuda')[None,None,:,None,None]
 def __call__(self,item,with_gt=True):
  raw,ms,ts,_=item
  data=dict(img=self.lut[self.ch,raw.cuda(non_blocking=True).long()],projection_mat=torch.from_numpy(np.stack([m['projection_mat'] for m in ms])).cuda(),image_wh=torch.from_numpy(np.stack([m['image_wh'] for m in ms])).cuda(),timestamp=torch.tensor([m['timestamp'] for m in ms],dtype=torch.float64,device='cuda'),img_metas=[dict(T_global=m['T_global'],T_global_inv=m['T_global_inv']) for m in ms])
  if with_gt:
   for m,t in zip(data['img_metas'],ts):m['instance_id']=t['ids']
   data.update(gt_bboxes_3d=[torch.from_numpy(t['boxes']).cuda() for t in ts],gt_labels_3d=[torch.from_numpy(t['labels']).cuda() for t in ts])
  return data
def retain_streams(m,indices):
 bank=m.head.instance_bank;idx=torch.tensor(indices,device='cuda',dtype=torch.long);old=bank.cached_anchor.shape[0]
 assert len(indices)==len(set(indices)) and all(0<=i<old for i in indices)
 for name in ['cached_feature','cached_anchor','mask','confidence','temp_confidence','instance_id']:
  v=getattr(bank,name,None)
  if v is not None:assert v.shape[0]==old;setattr(bank,name,v.index_select(0,idx))
 meta=bank.metas.copy()
 for k,v in meta.items():
  if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==old:meta[k]=v.index_select(0,idx)
  elif isinstance(v,list) and len(v)==old:meta[k]=[v[i] for i in indices]
 bank.metas=meta
 if m.head.sampler.dn_metas is not None:
  for k,v in m.head.sampler.dn_metas.items():
   assert isinstance(v,torch.Tensor) and v.shape[0]==old,(k,type(v))
   m.head.sampler.dn_metas[k]=v.index_select(0,idx)
def losses(m,d):
 parts=m.forward_train(**d);total=sum(v.mean() for k,v in parts.items() if 'loss' in k)
 if not torch.isfinite(total):raise FloatingPointError('비유한 loss')
 return total,parts
def run_group(m,g,opt=None,transfer=None,limit=None,step=1):
 train=opt is not None;m._precision=m._train_precision if train else 'fp32'
 torch.backends.cuda.matmul.allow_tf32=m._precision=='tf32';torch.backends.cudnn.allow_tf32=m._precision=='tf32'
 m.train(train);m.depth_branch.eval();reset(m);transfer=transfer or Transfer()
 n=g.length if limit is None else min(limit,g.length);frames=0;total=0.;parts_sum={};pending=0;updates=0;hist=collections.Counter();start=time.perf_counter();prev=None
 if train:opt.zero_grad(set_to_none=True)
 def update():
  nonlocal pending,updates
  for p in m.parameters():
   if p.grad is not None:p.grad.mul_(8/pending)
  norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],25.,error_if_nonfinite=True)
  opt.step();opt.zero_grad(set_to_none=True);hist[pending]+=1;pending=0;updates+=1
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
  future=pool.submit(g.cpu,0)
  for i in range(0,n,step):
   item=future.result()
   if i+step<n:future=pool.submit(g.cpu,i+step)
   active=item[3]
   if prev is not None and active!=prev:retain_streams(m,[prev.index(j) for j in active])
   prev=active;data=transfer(item);batch=len(active)
   with torch.set_grad_enabled(train):value,parts=losses(m,data)
   if train:
    (value*batch/8).backward();pending+=batch
    if pending>=8:update()
   total+=float(value.detach())*batch;frames+=batch
   for k,v in parts.items():parts_sum[k]=parts_sum.get(k,0.)+v.detach().mean()*batch
   del data,item,value,parts
 if train and pending:update()
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 result=dict(loss=total/frames,frames=frames,components={k:float(v)/frames for k,v in parts_sum.items()},elapsed_s=elapsed,frame_s=elapsed/frames,updates=updates,effective_batch_histogram=dict(hist),initial_batch=len(g.eps),episodes=g.eps,step=step)
 reset(m);return result

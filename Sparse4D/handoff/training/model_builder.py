"""Original Sparse4D fine-tuning model construction; path layout only adapted."""
import sys
import torch
from paths import ROOT,Q
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'handoff/core'))
from model_inputs import CLASS_NAMES
from mmcv import Config
import projects.mmdet3d_plugin
from mmdet.models import build_detector
COMMON=['car','truck','bus','pedestrian','bicycle','motorcycle']

class CommonFocal(torch.nn.Module):
 def __init__(self,original):super().__init__();self.original=original
 def forward(self,pred,target,weight=None,**kwargs):
  mask=pred.new_tensor([float(c in COMMON) for c in CLASS_NAMES]).expand_as(pred)
  if weight is not None:mask=mask*weight.reshape(-1,1) if weight.ndim==1 else mask*weight
  return self.original(pred,target,weight=mask,**kwargs)


def reset(model):
 model.head.instance_bank.reset();model.head.sampler.dn_metas=None


def build():
 cfg=Config.fromfile(str(ROOT/'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
 cfg.model.img_backbone.pretrained=None;cfg.model.head.instance_bank.anchor=str(Q/'artifacts/nuscenes_kmeans900_from_checkpoint.npy')
 model=build_detector(cfg.model)
 initial=ROOT/'checkpoints/sparse4dv3_r50.pth'
 model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['state_dict'],strict=True)
 for p in model.depth_branch.parameters():p.requires_grad_(False)
 model.head.loss_cls=CommonFocal(model.head.loss_cls)
 original=model.head.sampler.get_dn_anchors
 def finite_dn(labels,boxes,ids=None):
  masks=[torch.isfinite(b).all(-1) for b in boxes]
  return original([x[m] for x,m in zip(labels,masks)],[x[m] for x,m in zip(boxes,masks)],None if ids is None else [x[m] for x,m in zip(ids,masks)])
 model.head.sampler.get_dn_anchors=finite_dn
 # eval 모드의 dropout/BN/DN 동작은 유지하면서 모든 decoder loss에 필요한 cls를 반환.
 for op,layer in zip(model.head.operation_order,model.head.layers):
  if op=='refine':
   def with_cls(*args,_original=layer.forward,**kwargs):
    kwargs['return_cls']=True;return _original(*args,**kwargs)
   layer.forward=with_cls
 return model.cuda(),cfg


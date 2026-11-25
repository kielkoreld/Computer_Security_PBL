"""동봉된 실제 공식 입력 3시점. 기본 CPU strict load; --forward만 GPU 추론."""
import argparse,json,sys,datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'handoff'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(R/'core'))
from mmcv import Config
from mmdet.models import build_detector
import projects.mmdet3d_plugin
from model_inputs import export_boxes

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--forward',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 out=a.output or ROOT/'outputs'/('sparse4d_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'))
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 cfg=Config.fromfile(str(ROOT/'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
 cfg.model.img_backbone.pretrained=None;cfg.model.head.instance_bank.anchor=str(R/'artifacts/nuscenes_kmeans900_from_checkpoint.npy')
 model=build_detector(cfg.model).eval()
 path=ROOT/'checkpoints/best_epoch_012.pth'
 ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'],strict=True)
 assert ck['epoch']==12 and ck['validation_loss']==8.487784253068527
 values=dict(np.load(ROOT/'handoff/examples/sparse4d_3frames.npz'));infos=json.loads((ROOT/'handoff/examples/sparse4d_3frames_metadata.json').read_text())['infos']
 lut=np.load(R/'cache/normalization_lut.npy')
 inputs=[]
 for i in range(3):
  image=lut[np.arange(3)[None,:,None,None],values['rgb_uint8'][i]][None]
  data=dict(img=torch.from_numpy(image),projection_mat=torch.from_numpy(values['projection_mat'][i:i+1]),
    image_wh=torch.from_numpy(values['image_wh'][i:i+1]),timestamp=torch.from_numpy(values['timestamp'][i:i+1]),
    img_metas=[dict(T_global=values['T_global'][i],T_global_inv=values['T_global_inv'][i])])
  assert data['img'].shape==(1,6,3,256,704) and data['img'].dtype==torch.float32
  assert data['timestamp'].dtype==torch.float64 and set(data)=={'img','projection_mat','image_wh','timestamp','img_metas'}
  inputs.append(data)
 result=dict(pass_=True,strict_load=True,epoch=12,checkpoint_state_entries=len(ck['state_dict']),
   input_frames=3,timestamps_s=values['timestamp'].tolist(),input_shape=list(inputs[0]['img'].shape),
   gt_in_forward_input=False,actual_forward=a.forward,validation_scope='CPU 엄격한 가중치 로딩과 실제 입력 계약 검사')
 if a.forward:
  model=model.cuda()
  for data in inputs:
   for k,v in data.items():
    if torch.is_tensor(v):data[k]=v.cuda()
  with torch.inference_mode():
   model.head.instance_bank.reset();model.head.sampler.dn_metas=None
   for _ in range(3):model.simple_test(**inputs[0])
   torch.cuda.synchronize();model.head.instance_bank.reset()
   rows=[]
   for data,info in zip(inputs,infos):
    native=model.simple_test(**data)[0]['img_bbox'];boxes=native['boxes_3d'];boxes=getattr(boxes,'tensor',boxes)
    ids=native.get('instance_ids')
    rows.append(export_boxes(boxes.detach().cpu().numpy(),native['scores_3d'].detach().cpu().numpy(),
      native['labels_3d'].detach().cpu().numpy(),info,'sparse4d',None if ids is None else ids.detach().cpu().numpy()))
  (out/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False))
  result['validation_scope']='실제 GPU3시점 순차 추론. 저장 결과와 비트 단위 동일성을 보장하지 않음.'
 (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

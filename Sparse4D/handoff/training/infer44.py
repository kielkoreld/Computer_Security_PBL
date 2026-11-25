"""수정GT best/이전epoch7을 같은44경로에 GT 없는 공식 입력으로 추론."""
import argparse,concurrent.futures,json,pickle,signal,sys,time
import numpy as np
import torch
from paths import Q,R,ROOT,source,read,save,now,progress
sys.path.insert(0,str(ROOT/'handoff/core'));sys.path.insert(0,str(ROOT))
from model_inputs import export_boxes
from mmcv import Config
import projects.mmdet3d_plugin
from mmdet.models import build_detector

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',choices=['best','pretrained'],default='best');p.add_argument('--pilot',action='store_true');p.add_argument('--wait-for-cache',action='store_true');a=p.parse_args()
 if a.tag=='best':assert read(Q/'training/status.json')['complete']
 cp=Q/'training/best.pth' if a.tag=='best' else ROOT/'checkpoints/sparse4dv3_r50.pth'
 out=Q/('after_predictions' if a.tag=='best' else 'official_pretrained_predictions')
 if a.pilot:out=Q/'inference_smoke'
 eps=read(Q/'protocol/pilot.json')['episodes'][:1] if a.pilot else read(Q/'protocol/split.json')['evaluation44']
 if a.wait_for_cache:eps=sorted(eps,key=lambda e:(not (Q/'cache'/e/'complete.json').exists(),e))
 torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 cfg=Config.fromfile(str(ROOT/'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'));cfg.model.img_backbone.pretrained=None;cfg.model.head.instance_bank.anchor=str(Q/'artifacts/nuscenes_kmeans900_from_checkpoint.npy')
 model=build_detector(cfg.model).cuda().eval();ck=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'],strict=True);epoch=ck.get('epoch');del ck
 lut=torch.from_numpy(np.load(Q/'cache/normalization_lut.npy')).cuda();channels=torch.arange(3,device='cuda')[None,None,:,None,None]
 for ep in eps:
  if a.wait_for_cache:
   deadline=time.monotonic()+7200
   while not (Q/'cache'/ep/'complete.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('공식 입력 캐시 준비 대기2시간 초과: '+ep)
    time.sleep(2)
  dest=out/'sparse4d'/ep;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'status.json').exists() and read(dest/'status.json').get('complete'):continue
  with (Q/'cache'/ep/'meta.pkl').open('rb') as f:meta=pickle.load(f)
  with (source(ep)/'sparse4d_infos.pkl').open('rb') as f:infos=pickle.load(f)['infos']
  rgb=np.load(Q/'cache'/ep/'rgb.npy',mmap_mode='r');assert len(meta)==len(infos)==len(rgb)
  def prepare(i):return torch.from_numpy(np.array(rgb[i:i+1],copy=True)).pin_memory(),meta[i]
  def forward(item):
   raw,m=item;data=dict(img=lut[channels,raw.cuda(non_blocking=True).long()],projection_mat=torch.from_numpy(m['projection_mat'][None]).cuda(),image_wh=torch.from_numpy(m['image_wh'][None]).cuda(),timestamp=torch.tensor([m['timestamp']],dtype=torch.float64,device='cuda'),img_metas=[dict(T_global=m['T_global'],T_global_inv=m['T_global_inv'])])
   assert set(data)=={'img','projection_mat','image_wh','timestamp','img_metas'}
   return model.simple_test(**data)[0]['img_bbox']
  status=dict(complete=False,strict_load=True,full_episode=False,checkpoint=str(cp),epoch=epoch,episode=ep,frames=0,started_utc=now(),input='official cached RGB and pose; GT never loaded')
  timings=[]
  try:
   with torch.inference_mode(),concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    model.head.instance_bank.reset();model.head.sampler.dn_metas=None;first=prepare(0)
    for _ in range(3):forward(first)
    torch.cuda.synchronize();model.head.instance_bank.reset();torch.cuda.reset_peak_memory_stats();future=pool.submit(prepare,0);start=time.perf_counter()
    with (dest/'predictions.partial.jsonl').open('w') as f:
     for i,info in enumerate(infos):
      t=time.perf_counter();item=future.result()
      if i+1<len(infos):future=pool.submit(prepare,i+1)
      native=forward(item);torch.cuda.synchronize();box=native['boxes_3d'];box=box.tensor if hasattr(box,'tensor') else box
      ids=native.get('instance_ids');record=export_boxes(box.detach().cpu().numpy(),native['scores_3d'].detach().cpu().numpy(),native['labels_3d'].detach().cpu().numpy(),info,'sparse4d',None if ids is None else ids.detach().cpu().numpy())
      f.write(json.dumps(record,allow_nan=False)+'\n');timings.append(time.perf_counter()-t);status['frames']+=1
    assert status['frames']==len(infos);(dest/'predictions.partial.jsonl').replace(dest/'predictions.jsonl');status.update(complete=True,full_episode=True,elapsed_s=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
   print(json.dumps(dict(episode=ep,tag=a.tag,frames=len(infos),elapsed_s=status['elapsed_s'])),flush=True)
  except BaseException as e:status['error']=repr(e);raise
  finally:status['finished_utc']=now();save(dest/'status.json',status);save(dest/'timing.json',timings)
 save(out/'status.json',dict(complete=True,tag=a.tag,routes=len(eps),epoch=epoch,checkpoint=str(cp),finished_utc=now()))
if __name__=='__main__':main()

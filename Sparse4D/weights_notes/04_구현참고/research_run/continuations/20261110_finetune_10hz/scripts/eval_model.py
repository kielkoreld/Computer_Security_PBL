"""검증/평가 순차 추론과 공식 거리 AP. 평가 GT는 추론 완료 후 별도 로드한다."""
import argparse,collections,concurrent.futures,gc,json,pickle,signal,sys,time,fcntl
import numpy as np
import torch
from common import *
from engine import seed,build,reset
from cache_io import RGB
from model_inputs import export_boxes
from pyquaternion import Quaternion
sys.path.insert(0,str(R/'continuations/20261102_nuscenes44/scripts'))
import nu_metrics as nm
STOP=False
def interrupt(sig,frame):
 global STOP
 STOP=True
def infer(eps,out,checkpoint,precision='fp32',dt=.1,step=1,no_history=False):
 seed(0,precision);torch.set_num_threads(4);m,_=build(checkpoint,precision,dt);m.eval();status=[]
 lut=torch.from_numpy(np.load(Q/'cache/normalization_lut.npy')).cuda();ch=torch.arange(3,device='cuda')[None,None,:,None,None]
 for ep in eps:
  dest=out/'predictions'/ep;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():status.append(read(dest/'complete.json'));continue
  d=P/'cache'/ep
  if d.exists():rgb=RGB(ep)
  else:d=OLD10/'cache'/ep;rgb=np.load(d/'rgb.npy',mmap_mode='r')
  with (d/'meta.pkl').open('rb') as f:meta=pickle.load(f)
  with (d/'infos.pkl').open('rb') as f:infos=pickle.load(f)
  def cpu(i):return torch.from_numpy(np.array(rgb[i],copy=True)[None]).pin_memory(),meta[i]
  reset(m);start=time.perf_counter();count=0
  with torch.inference_mode(),concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,(dest/'predictions.partial.jsonl').open('w') as f:
   future=pool.submit(cpu,0)
   for i in range(0,len(meta),step):
    if STOP:raise InterruptedError('추론 중단: 현재 부분 예측 보존')
    raw,md=future.result()
    if i+step<len(meta):future=pool.submit(cpu,i+step)
    if no_history:reset(m)
    data=dict(img=lut[ch,raw.cuda(non_blocking=True).long()],projection_mat=torch.from_numpy(md['projection_mat'][None]).cuda(),image_wh=torch.from_numpy(md['image_wh'][None]).cuda(),timestamp=torch.tensor([md['timestamp']],device='cuda',dtype=torch.float64),img_metas=[dict(T_global=md['T_global'],T_global_inv=md['T_global_inv'])])
    o=m.simple_test(**data)[0]['img_bbox'];b=o['boxes_3d'];b=b.tensor if hasattr(b,'tensor') else b
    assert torch.isfinite(b).all() and torch.isfinite(o['scores_3d']).all() and (b[:,3:6]>0).all()
    row=export_boxes(b.cpu().numpy(),o['scores_3d'].cpu().numpy(),o['labels_3d'].cpu().numpy(),infos[i],'sparse4d');assert row['sample_token']==md['token']
    f.write(json.dumps(row,allow_nan=False)+'\n');count+=1
  if isinstance(rgb,RGB):rgb.close()
  (dest/'predictions.partial.jsonl').replace(dest/'predictions.jsonl');record=dict(complete=True,episode=ep,frames=count,elapsed_s=time.perf_counter()-start,checkpoint=str(checkpoint),precision=precision,default_dt=dt,step=step,no_history=no_history,gt_used_by_inference=False,finished_utc=now());save(dest/'complete.json',record);status.append(record)
  print('추론',out.name,ep,count,flush=True)
 save(out/'inference.json',dict(complete=True,runs=status,frames=sum(x['frames'] for x in status),finished_utc=now()));del m;gc.collect();torch.cuda.empty_cache()
def load(ep,out,only=None,common=False):
 d=P/'cache'/ep
 if not d.exists():d=OLD10/'cache'/ep
 gt={g['sample_token']:g for g in map(json.loads,(d/'gt_common.jsonl').open())};G={};B={}
 for pred in map(json.loads,(out/'predictions'/ep/'predictions.jsonl').open()):
  g=gt[pred['sample_token']]
  if common and g['frame_id']%5:continue
  tok=g['sample_token'];assert abs(pred['timestamp']-g['timestamp'])<1e-8;ego=np.array(g['ego_to_world'])[:3,3];G[tok]=[];B[tok]=[]
  for truth,objects,dst in [(True,g['objects'],G[tok]),(False,pred['objects'],B[tok])]:
   for o in objects:
    c=o['class']
    if c not in nm.CLASSES or only and c!=only or truth and not o['valid_mask']:continue
    rel=np.array(o['center_xyz'])-ego
    if np.linalg.norm(rel[:2])>=nm.RANGES[c]:continue
    sz=o['size_lwh'];dst.append(nm.make_box(dict(sample_token=tok,translation=o['center_xyz'],size=[sz[1],sz[0],sz[2]],rotation=Quaternion(axis=[0,0,1],angle=o['yaw']).elements.tolist(),velocity=o['velocity_xy'] if not truth or o['velocity_valid'] else None,ego_translation=rel.tolist(),num_pts=o.get('num_lidar_points',-1),detection_name=c,detection_score=o['score']),truth))
 return G,B
def evaluate(eps,out,common=False):
 where=out/('metrics_common2hz' if common else 'metrics');classes=list(nm.CLASSES);result=dict(classes={},NDS=None,metric='CARLA common-class nuScenes-devkit AP',min_precision=.1,min_recall=.1,distance_thresholds_m=[.5,1,2,4],gt_policy='static_inclusive_v1')
 # 검증 부분집합은 한 번 파싱. 큰 최종 집합은 클래스별 순차 평가로 RAM 제한.
 if len(eps)<=16:
  G={};B={}
  for ep in eps:g,b=load(ep,out,common=common);G.update(g);B.update(b)
  result,curves=nm.official(G,B,curves=True);save(where/'curves.json',curves)
 else:
  for c in classes:
   G={};B={}
   for ep in eps:g,b=load(ep,out,c,common);G.update(g);B.update(b)
   nm.CLASSES=[c]
   try:res,curves=nm.official(G,B,curves=True)
   finally:nm.CLASSES=classes
   result['classes'][c]=res['classes'][c];result['frames']=res['frames'];save(where/f'class_{c}.json',res);save(where/f'curves_{c}.json',curves);del G,B;gc.collect()
  values=[v['mean_ap'] for v in result['classes'].values() if v['mean_ap'] is not None];result['mean_ap']=float(np.mean(values)) if values else None
 save(where/'pooled.json',result);print('AP',out.name,'common' if common else 'all',result['mean_ap'],flush=True);return result
def main():
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--checkpoint',type=Path,default=INITIAL);a.add_argument('--precision',default='fp32');a.add_argument('--dt',type=float,default=.1);a.add_argument('--step',type=int,default=1);a.add_argument('--no-history',action='store_true');a.add_argument('--split',choices=['pilot','evaluation44','validation'],default='pilot');a.add_argument('--metrics-only',action='store_true');a.add_argument('--inference-only',action='store_true');args=a.parse_args()
 eps=read(P/'protocol/pilot.json')['validation'] if args.split=='pilot' else read(P/'protocol/split.json')[args.split];out=P/'evaluations'/args.tag
 assert not(args.metrics_only and args.inference_only);out.mkdir(parents=True,exist_ok=True)
 # GPU 추론과 CPU 지표 집계의 독립 작업을 겹칠 때 같은 tag의 중복 실행 방지.
 contract=dict(checkpoint=str(args.checkpoint.resolve()),precision=args.precision,dt=args.dt,step=args.step,no_history=args.no_history,episodes=eps)
 with (out/'execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (out/'contract.json').exists():assert read(out/'contract.json')==contract,'동일 tag의 실험 조건 변경 금지'
  else:save(out/'contract.json',contract)
  for ep in eps:
   done=out/'predictions'/ep/'complete.json'
   if done.exists():
    s=read(done);assert s['checkpoint']==str(args.checkpoint) and s['precision']==args.precision and s['default_dt']==args.dt and s['step']==args.step and s['no_history']==args.no_history
  if not args.metrics_only:infer(eps,out,args.checkpoint,args.precision,args.dt,args.step,args.no_history)
  if not args.inference_only:evaluate(eps,out);evaluate(eps,out,True)
if __name__=='__main__':main()

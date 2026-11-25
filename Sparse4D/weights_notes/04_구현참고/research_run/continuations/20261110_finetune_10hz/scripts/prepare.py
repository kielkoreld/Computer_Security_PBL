"""GT/정보 변환, 공식 pipeline 전수 대조, 2Hz 재사용/10Hz 무손실 캐시."""
import argparse,collections,concurrent.futures,json,math,multiprocessing,pickle,signal,subprocess,sys,time,zlib
import numpy as np
from common import *
from cache_io import SHAPE
sys.path.insert(0,str(R/'runtime/Sparse4D-main'))
import torch,mmcv
from mmcv import Config
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.nuscenes_3d_det_track_dataset import NuScenes3DDetTrackDataset
from model_inputs import T,CLASS_NAMES
COMMON=['car','truck','bus','pedestrian','bicycle','motorcycle']
STOP=False
CHILD=None
def interrupt(sig,frame):
 global STOP
 STOP=True
 if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)
def prepare(ep):
 global CHILD
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 torch.set_num_threads(1);mmcv.use_backend('cv2')
 import cv2
 cv2.setNumThreads(1)
 item=next(x for x in read(P/'protocol/inventory.json')['episodes'] if x['episode']==ep)
 out=P/'cache'/ep;data=P/'data'/ep;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return read(out/'complete.json')
 start=time.perf_counter()
 if not (data/'conversion_report.json').exists():
  cmd=[str(PY),str(R/'scripts/convert_dataset_extended.py'),'--episode',ep,'--frame-step','1','--extracted-root',str(Path(item['raw_path']).parent),'--audit',str(P/'protocol/raw_inventory_audit.json'),'--output-root',str(P/'data'),'--records-root',str(P/'records/conversion')]
  with (P/'logs'/f'convert_{ep}.log').open('a') as f:
   CHILD=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);code=CHILD.wait();CHILD=None
  if code or STOP:raise RuntimeError(('원본 변환 실패/중단',ep,code))
 cfg=Config.fromfile(str(R/'runtime/Sparse4D-main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
 ds=NuScenes3DDetTrackDataset(ann_file=str(data/'sparse4d_infos.pkl'),pipeline=cfg.test_pipeline,data_root='',classes=cfg.class_names,modality=cfg.input_modality,test_mode=True,data_aug_conf=cfg.data_aug_conf)
 raw=[json.loads(x) for x in (data/'gt_common.jsonl').open()]
 old_gt=[json.loads(x) for x in (Q/'data'/ep/'gt_common.jsonl').open()]
 with (Q/'cache'/ep/'meta.pkl').open('rb') as f:old_meta=pickle.load(f)
 old_rgb=np.load(Q/'cache'/ep/'rgb.npy',mmap_mode='r');lut=np.load(Q/'cache/normalization_lut.npy')
 mean=np.array(cfg.img_norm_cfg['mean'],np.float32);std=np.array(cfg.img_norm_cfg['std'],np.float32)
 ids={x:i for i,x in enumerate(sorted({o['track_id'] for g in raw for o in g['objects']}))}
 n=len(ds);assert n==item['frames']==len(raw)
 meta=[];targets=[];index=[];stats=collections.Counter();cl=collections.Counter();geommax=velmax=0.
 with (out/'rgb.partial.zlib').open('wb') as blob,(out/'gt_common.partial.jsonl').open('w') as gf:
  for i in range(n):
   if STOP:raise InterruptedError('입력 준비 중단')
   native=ds[i];im=native['img'].data.numpy();m=native['img_metas'].data
   u=np.rint(im*std[None,:,None,None]+mean[None,:,None,None]);assert u.min()>=0 and u.max()<=255
   u=u.astype(np.uint8);assert u.shape==SHAPE and np.array_equal(lut[np.arange(3)[None,:,None,None],u],im)
   md=dict(token=ds.data_infos[i]['token'],projection_mat=np.array(native['projection_mat']),image_wh=np.array(native['image_wh']),timestamp=float(native['timestamp']),T_global=np.array(m['T_global']),T_global_inv=np.array(m['T_global_inv']));meta.append(md)
   g=raw[i];assert md['token']==g['sample_token'] and g['frame_id']==i and abs(g['timestamp']-i*.1)<1e-8 and abs(md['timestamp']-i*.1)<1e-8
   assert np.max(abs(md['T_global']@md['T_global_inv']-np.eye(4)))<1e-8
   boxes=[];labels=[];instance=[];G=md['T_global'];inv=md['T_global_inv']
   for o in g['objects']:
    o['legacy_valid_mask']=o['valid_mask'];o['legacy_ignore_reasons']=o['ignore_reasons'][:]
    o['ignore_reasons']=[s for s in o['ignore_reasons'] if s!='static_geometry_unverified'];o['valid_mask']=not o['ignore_reasons']
    if not o['valid_mask'] or o['class'] not in COMMON:continue
    c=(inv@np.r_[o['center_xyz'],1])[:3];h=np.linalg.solve(G[:2,:2],[math.cos(o['yaw']),math.sin(o['yaw'])]);v=np.linalg.solve(G[:2,:2],o['velocity_xy']) if o['velocity_valid'] else [np.nan,np.nan]
    b=np.array([*c,*o['size_lwh'],math.atan2(h[1],h[0]),*v],np.float32)
    geommax=max(geommax,float(np.linalg.norm((G@np.r_[b[:3].astype(float),1])[:3]-o['center_xyz'])))
    if o['velocity_valid']:velmax=max(velmax,float(np.max(abs(G[:2,:2]@b[7:9]-o['velocity_xy']))))
    boxes.append(b);labels.append(CLASS_NAMES.index(o['class']));instance.append(ids[o['track_id']]);cl[o['class']]+=1
   boxes=np.array(boxes,np.float32).reshape(-1,9);assert np.isfinite(boxes[:,:7]).all() and (boxes[:,3:6]>0).all()
   targets.append(dict(boxes=boxes,labels=np.array(labels,np.int64),ids=np.array(instance,np.int64)))
   stats['frames']+=1;stats['positive_objects']+=len(boxes);stats['empty_positive_frames']+=int(len(boxes)==0)
   if i%5==0:
    assert np.array_equal(u,old_rgb[i//5]);old=old_meta[i//5]
    for k in ['projection_mat','image_wh','T_global','T_global_inv']:assert np.allclose(md[k],old[k],atol=1e-8,rtol=0),(ep,i,k)
    assert {k:v for k,v in g.items() if k!='split'}=={k:v for k,v in old_gt[i//5].items() if k!='split'},(ep,i,'기존2Hz GT 불일치')
    index.append(None);stats['old2hz_exact_frames']+=1
   else:
    b=zlib.compress(u.tobytes(),level=1);assert np.array_equal(np.frombuffer(zlib.decompress(b),np.uint8).reshape(SHAPE),u)
    index.append([blob.tell(),len(b)]);blob.write(b)
   gf.write(json.dumps(g,allow_nan=False)+'\n')
 assert geommax<1e-4 and velmax<1e-4
 for name,val in [('meta',meta),('targets',targets),('infos',ds.data_infos)]:
  with (out/f'{name}.partial.pkl').open('wb') as f:pickle.dump(val,f,protocol=5)
  (out/f'{name}.partial.pkl').replace(out/f'{name}.pkl')
 (out/'rgb.partial.zlib').replace(out/'rgb.zlib');(out/'gt_common.partial.jsonl').replace(out/'gt_common.jsonl');save(out/'rgb_index.json',index)
 result=dict(complete=True,episode=ep,split=item['split'],frames=n,counts=dict(stats),classes=dict(cl),all_official_pixels_exact=True,all_new_compressed_frames_exact=True,old2hz_input_and_gt_exact=True,world_roundtrip_max_m=geommax,velocity_roundtrip_max_mps=velmax,compressed_bytes=(out/'rgb.zlib').stat().st_size,elapsed_s=time.perf_counter()-start,finished_utc=now())
 save(out/'complete.json',result);return result
def main():
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');a.add_argument('--workers',type=int,default=6);a.add_argument('--episode');args=a.parse_args()
 split=read(P/'protocol/split.json')
 if args.episode:eps=[args.episode]
 elif args.pilot:
  train=read(Q/'protocol/pilot.json')['episodes']
  wanted=['Accident','ControlLoss','PedestrianCrossing','ParkingCrossingPedestrian','CrossingBicycleFlow','HighwayCutIn','StaticCutIn','ConstructionObstacle']
  val=[sorted(e for e in split['validation'] if e.startswith(k+'_'))[0] for k in wanted]
  eps=train+val;save(P/'protocol/pilot.json',dict(train=train,validation=val,selection='기존 기능시험 학습4클립 및 사전 지정8시나리오의 검증클립 이름순 첫번째. 점수 미사용.'))
 else:eps=split['train']+split['validation']
 rows=[];start=time.perf_counter();status=P/'records'/('preparation_pilot.json' if args.pilot else 'preparation.json')
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  futures={pool.submit(prepare,ep):ep for ep in eps}
  try:
   for f in concurrent.futures.as_completed(futures):
    if STOP:raise InterruptedError('입력 준비 감독 중단')
    row=f.result();rows.append(row)
    save(status,dict(complete=len(rows)==len(eps),done=len(rows),expected=len(eps),frames=sum(x['frames'] for x in rows),elapsed_s=time.perf_counter()-start,updated_utc=now()))
    print(f'입력 검증 {len(rows)}/{len(eps)}: {row["episode"]} ({row["frames"]})',flush=True)
  except BaseException as ex:
   save(status,dict(complete=False,error=repr(ex),done=len(rows),time=now()))
   for proc in pool._processes.values():proc.terminate()
   raise
 save(status,dict(complete=True,done=len(rows),frames=sum(x['frames'] for x in rows),elapsed_s=time.perf_counter()-start,routes=rows,finished_utc=now()))
if __name__=='__main__':main()

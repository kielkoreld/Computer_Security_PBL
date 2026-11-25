"""원래 GT에서 정적 객체 일괄 제외 조건만 해제. 좌표와 다른 마스크 보존."""
import collections,concurrent.futures,json,math,pickle,sys,time
import numpy as np
from paths import Q,R,S,ROOT,source,save,read,now,progress
sys.path.insert(0,str(ROOT/'handoff/core'))
from model_inputs import T,CLASS_NAMES
COMMON=['car','truck','bus','pedestrian','bicycle','motorcycle']

def prepare(ep):
 out=Q/'data'/ep;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return read(out/'complete.json')
 d=source(ep);raw=[json.loads(x) for x in (d/'gt_common.jsonl').read_text().splitlines()]
 with (d/'sparse4d_infos.pkl').open('rb') as f:infos=pickle.load(f)['infos']
 assert len(raw)==len(infos)>0
 ids={s:i for i,s in enumerate(sorted({o['track_id'] for g in raw for o in g['objects']}))}
 targets=[];stats=collections.Counter();classes=collections.Counter();times=[]
 for info,g in zip(infos,raw):
  assert info['token']==g['sample_token'];times.append(g['timestamp']);stats['frames']+=1
  boxes=[];labels=[];instance=[];G=T(info['ego2global_rotation'],info['ego2global_translation'])@T(info['lidar2ego_rotation'],info['lidar2ego_translation']);inv=np.linalg.inv(G)
  for o in g['objects']:
   stats['objects']+=1;old=bool(o['valid_mask']);reasons=list(o['ignore_reasons']);assert old==(not reasons)
   o['legacy_valid_mask']=old;o['legacy_ignore_reasons']=reasons
   o['ignore_reasons']=[s for s in reasons if s!='static_geometry_unverified'];o['valid_mask']=not o['ignore_reasons']
   if old:stats['legacy_valid']+=1
   if not old and o['valid_mask']:
    assert set(reasons)=={'static_geometry_unverified'} and o['num_lidar_points']>0 and o['camera_frustum_overlap'];stats['static_added']+=1
   if o['valid_mask']:
    stats['valid']+=1;classes[o['class']]+=1
   if not o['valid_mask'] or o['class'] not in COMMON:continue
   c=(inv@np.r_[o['center_xyz'],1])[:3]
   h=np.linalg.solve(G[:2,:2],[math.cos(o['yaw']),math.sin(o['yaw'])]);yaw=math.atan2(h[1],h[0])
   v=np.linalg.solve(G[:2,:2],o['velocity_xy']) if o['velocity_valid'] else [np.nan,np.nan]
   boxes.append([*c,*o['size_lwh'],yaw,*v]);labels.append(CLASS_NAMES.index(o['class']));instance.append(ids[o['track_id']])
  b=np.asarray(boxes,dtype=np.float32).reshape(-1,9);assert np.isfinite(b[:,:7]).all() and (b[:,3:6]>0).all()
  targets.append(dict(boxes=b,labels=np.asarray(labels,np.int64),ids=np.asarray(instance,np.int64)))
  if not len(boxes):stats['empty_positive_frames']+=1
 assert len(set(times))==len(times) and all(b>a for a,b in zip(times,times[1:]))
 p=out/'gt_common.jsonl';t=p.with_suffix('.tmp')
 with t.open('w') as f:
  for g in raw:f.write(json.dumps(g,allow_nan=False)+'\n')
 t.replace(p)
 for n in ['sparse4d_infos.pkl','bevnext_infos.pkl']:
  if not (out/n).exists():(out/n).symlink_to(d/n)
 with (out/'targets.tmp').open('wb') as f:pickle.dump(targets,f,protocol=5)
 (out/'targets.tmp').replace(out/'targets.pkl')
 result=dict(episode=ep,counts=dict(stats),classes=dict(classes),source=str(d),complete=True,finished_utc=now())
 save(out/'complete.json',result);return result

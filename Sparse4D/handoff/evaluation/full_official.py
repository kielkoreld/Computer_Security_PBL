"""Official pooled CARLA common-class evaluation functions."""
import collections,gc,hashlib,json,math,resource
import numpy as np
from pyquaternion import Quaternion
from paths import read,save,source
import nu_metrics as nm
CLASSES=list(nm.CLASSES)

def token(s):return hashlib.md5(('sample|'+s).encode()).hexdigest()


def load(ep,path,only=None):
 rawgt=[json.loads(x) for x in (source(ep)/'gt_common.jsonl').read_text().splitlines()]
 rawpred=[json.loads(x) for x in path.read_text().splitlines()]
 assert len(rawgt)==len(rawpred);G={};P={}
 for g,p in zip(rawgt,rawpred):
  assert g['sample_token']==p['sample_token'] and g['timestamp']==p['timestamp'];tok=token(g['sample_token']);ego=np.array(g['ego_to_world'])[:3,3];G[tok]=[];P[tok]=[]
  for o in g['objects']:
   c=o['class']
   if not o['valid_mask'] or c not in CLASSES or only and c!=only:continue
   rel=np.asarray(o['center_xyz'])-ego
   if np.linalg.norm(rel[:2])>=nm.RANGES[c]:continue
   sz=o['size_lwh'];G[tok].append(nm.make_box(dict(sample_token=tok,translation=o['center_xyz'],size=[sz[1],sz[0],sz[2]],rotation=Quaternion(axis=[0,0,1],angle=o['yaw']).elements.tolist(),velocity=o['velocity_xy'] if o['velocity_valid'] else None,ego_translation=rel.tolist(),num_pts=o['num_lidar_points'],detection_name=c),True))
  assert len(p['objects'])<=500
  for o in p['objects']:
   c=o['class']
   if c not in CLASSES or only and c!=only:continue
   rel=np.asarray(o['center_xyz'])-ego
   if np.linalg.norm(rel[:2])>=nm.RANGES[c]:continue
   sz=o['size_lwh'];P[tok].append(nm.make_box(dict(sample_token=tok,translation=o['center_xyz'],size=[sz[1],sz[0],sz[2]],rotation=Quaternion(axis=[0,0,1],angle=o['yaw']).elements.tolist(),velocity=o['velocity_xy'],ego_translation=rel.tolist(),detection_name=c,detection_score=o['score'])))
 return G,P


def pooled(eps,model,predroot,out):
 final={'metric':'CARLA common-class nuScenes-devkit AP','devkit':'1.2.0','classes':{},'NDS':None,'mAAE':None,'min_precision':.1,'min_recall':.1}
 for cls in CLASSES:
  path=out/model/f'class_{cls}.json'
  if path.exists():entry=read(path)
  else:
   G={};P={}
   for ep in eps:
    g,p=load(ep,predroot/model/ep/'predictions.jsonl',cls);G.update(g);P.update(p)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>25*2**30:raise MemoryError('공식 pooled 평가 메모리 한계25GiB; 중단 후 방식 검토 필요')
   nm.CLASSES=[cls]
   try:m,curve=nm.official(G,P,curves=True)
   finally:nm.CLASSES=CLASSES
   entry={'class':m['classes'][cls],'frames':m['frames'],'curves':curve};save(path,entry);del G,P;gc.collect()
  final['classes'][cls]=entry['class'];final['frames']=entry['frames']
 final['mean_ap']=float(np.mean([x['mean_ap'] for x in final['classes'].values() if x['mean_ap'] is not None]));save(out/model/'official_metrics.json',final);return final


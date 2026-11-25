"""Unmodified nuScenes-devkit metric calls for CARLA common classes."""
import json,math
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate,calc_ap,calc_tp
CLASSES=['car','truck','bus','pedestrian','bicycle','motorcycle'];RANGES={c:40 if c in ['pedestrian','bicycle','motorcycle'] else 50 for c in CLASSES};TH=[.5,1.,2.,4.]

def clean(v):
 if isinstance(v,np.bool_):return bool(v)
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,(float,np.floating)):return float(v) if math.isfinite(v) else None
 if isinstance(v,(np.integer,)):return int(v)
 return v


def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(clean(d),ensure_ascii=False,allow_nan=False,indent=2));q.replace(p)


def make_box(row,gt=False):
 velocity=row.get('velocity');velocity=[np.nan,np.nan] if velocity is None else velocity
 return DetectionBox(sample_token=row['sample_token'],translation=tuple(row['translation']),size=tuple(row['size']),rotation=tuple(row['rotation']),velocity=tuple(velocity),ego_translation=tuple(row.get('ego_translation',[0,0,0])),num_pts=int(row.get('num_pts',-1)),detection_name=row['detection_name'],detection_score=-1.0 if gt else float(row['detection_score']),attribute_name='')


def official(gt,pr,curves=False):
 # 클래스별 선별은 공식함수 내부 class 검사와 동등하며 비대상GT 반복 비용을 줄인다.
 result={'metric':'CARLA common-class nuScenes-devkit AP','devkit':'1.2.0','frames':len(gt),'min_recall':.1,'min_precision':.1,'mAAE':None,'NDS':None,'classes':{}}
 arrays={}
 for cls in CLASSES:
  gb,pb=EvalBoxes(),EvalBoxes()
  for token,rows in gt.items():gb.add_boxes(token,[b for b in rows if b.detection_name==cls])
  for token,rows in pr.items():pb.add_boxes(token,[b for b in rows if b.detection_name==cls])
  count=len(gb.all);predn=len(pb.all);aps={};tps={};recall=None
  if count:
   for th in TH:
    md=accumulate(gb,pb,cls,center_distance,th,verbose=False);aps[str(th)]=calc_ap(md,.1,.1)
    if curves:arrays[cls+'_'+str(th)]={k:getattr(md,k).tolist() for k in ['recall','precision','confidence','trans_err','scale_err','orient_err','vel_err']}
    if th==2.:
     tps={k:calc_tp(md,.1,k) for k in ['trans_err','scale_err','orient_err','vel_err']};recall=md.max_recall
   avg=float(np.mean(list(aps.values())))
  else:avg=None;aps={str(th):None for th in TH}
  result['classes'][cls]={'gt':count,'predictions':predn,'mean_ap':avg,'threshold_ap':aps,'tp_errors':tps,'maximum_recall_at_2m':recall,'TP_insufficient_recall_returns_default_1':None if recall is None else recall<=.1,'valid_velocity_GT':sum(np.isfinite(b.velocity).all() for b in gb.all)}
 vals=[r['mean_ap'] for r in result['classes'].values() if r['mean_ap'] is not None];result['mean_ap']=float(np.mean(vals)) if vals else None
 result['macro_tp_errors']={k:float(np.mean([r['tp_errors'][k] for r in result['classes'].values() if r['gt']])) for k in ['trans_err','scale_err','orient_err','vel_err']} if vals else {}
 return clean(result),clean(arrays)


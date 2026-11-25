"""CARLA common-class XY center-distance AP. This is explicitly not nuScenes NDS."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from geometry import transform

CLASSES=['car','truck','bus','pedestrian','bicycle','motorcycle']
RANGES={c:40 if c in ['pedestrian','bicycle','motorcycle'] else 50 for c in CLASSES}

def validate_prediction(obj):
    for key,n in [('center_xyz',3),('size_lwh',3)]:
        x=np.asarray(obj[key]); assert x.shape==(n,) and np.isfinite(x).all(),key
    assert (np.asarray(obj['size_lwh'])>0).all()
    assert np.isfinite(obj['yaw']) and np.isfinite(obj['score']) and 0<=obj['score']<=1
    assert obj['coordinate_frame']=='world_rh'
    if obj.get('velocity_xy') is not None:
        assert np.asarray(obj['velocity_xy']).shape==(2,) and np.isfinite(obj['velocity_xy']).all()

def ap_integral(tp,fp,total):
    if total==0:return None
    if not tp:return 0.
    tp=np.cumsum(tp);fp=np.cumsum(fp)
    recall=tp/total;precision=tp/np.maximum(tp+fp,1)
    r=np.r_[0,recall,1];p=np.r_[0,precision,0]
    p=np.maximum.accumulate(p[::-1])[::-1]
    k=np.where(r[1:]!=r[:-1])[0]
    return float(np.sum((r[k+1]-r[k])*p[k+1]))

def evaluate(gt_frames,pred_frames,thresholds=(.5,1.,2.,4.)):
    gt={f['sample_token']:f for f in gt_frames};pr={f['sample_token']:f for f in pred_frames}
    if len(gt)!=len(gt_frames) or len(pr)!=len(pred_frames):raise ValueError('중복 frame token')
    if set(gt)!=set(pr):raise ValueError('누락/추가 예측 프레임: 빈 예측도 명시적으로 저장해야 합니다')
    if not gt:raise ValueError('평가 프레임이 없습니다')
    for token in gt:
        g,p=gt[token],pr[token]
        if p['episode_id']!=g['episode_id'] or abs(p['timestamp']-g['timestamp'])>1e-6:raise ValueError('예측 episode/시각 불일치')
        for x in p['objects']:
            validate_prediction(x)
            if x['episode_id']!=p['episode_id'] or abs(x['timestamp']-p['timestamp'])>1e-6:
                raise ValueError('프레임과 객체의 episode/시각 불일치')
    results={};matches=[]
    for cls in CLASSES:
        valid={k:[b for b in f['objects'] if b['class']==cls and b['valid_mask']] for k,f in gt.items()}
        ignore={k:[b for b in f['objects'] if b['class']==cls and not b['valid_mask']] for k,f in gt.items()}
        total=sum(map(len,valid.values()));scores=[]
        for token,f in pr.items():
            E_G=np.linalg.inv(np.asarray(gt[token]['ego_to_world']))
            for i,b in enumerate(f['objects']):
                if b['class']!=cls:continue
                dist=np.linalg.norm(transform([b['center_xyz']],E_G)[0,:2])
                if dist<=RANGES[cls]:scores.append((b['score'],token,i,b))
        scores.sort(key=lambda x:(-x[0],x[1],x[2]))
        per={}
        for threshold in thresholds:
            used=defaultdict(set);tp=[];fp=[];ignored=0;errors=[]
            for score,token,i,b in scores:
                distances=[np.linalg.norm(np.asarray(b['center_xyz'])[:2]-np.asarray(g['center_xyz'])[:2]) for g in valid[token]]
                candidates=[j for j,d in enumerate(distances) if d<threshold and j not in used[token]]
                if candidates:
                    j=min(candidates,key=lambda j:distances[j]);used[token].add(j);g=valid[token][j]
                    tp.append(1);fp.append(0)
                    angle=abs(float((b['yaw']-g['yaw']+np.pi)%(2*np.pi)-np.pi))
                    a=np.asarray(b['size_lwh']);c=np.asarray(g['size_lwh']);inter=np.minimum(a,c).prod()
                    err={'position_m':float(distances[j]),'yaw_rad':angle,'size_error_1_minus_aligned_iou':float(1-inter/(a.prod()+c.prod()-inter))}
                    if g.get('velocity_valid') and b.get('velocity_xy') is not None:
                        err['velocity_mps']=float(np.linalg.norm(np.asarray(b['velocity_xy'])-g['velocity_xy']))
                    errors.append(err)
                    outcome='TP';gt_id=g['track_id']
                elif not any(d<threshold for d in distances) and any(np.linalg.norm(np.asarray(b['center_xyz'])[:2]-np.asarray(g['center_xyz'])[:2])<threshold for g in ignore[token]):
                    ignored+=1;outcome='ignored';gt_id=None
                else:tp.append(0);fp.append(1);outcome='FP';gt_id=None
                if threshold==2.:matches.append({'sample_token':token,'class':cls,'prediction_index':i,'score':score,'outcome':outcome,'gt_track_id':gt_id})
            errkeys=set().union(*(e.keys() for e in errors)) if errors else set()
            per[str(threshold)]={'ap':ap_integral(tp,fp,total),'gt':total,'tp':sum(tp),'fp':sum(fp),'fn':total-sum(tp),'ignored':ignored,
               'errors':{k:{'mean':float(np.mean([e[k] for e in errors if k in e])),'count':sum(k in e for e in errors)} for k in errkeys}}
            if threshold==2.:
                for token,boxes in valid.items():
                    for j,g in enumerate(boxes):
                        if j not in used[token]:matches.append({'sample_token':token,'class':cls,'outcome':'FN','gt_track_id':g['track_id']})
        aps=[v['ap'] for v in per.values() if v['ap'] is not None]
        results[cls]={'mean_ap':float(np.mean(aps)) if aps else None,'thresholds_m':per}
    aps=[v['mean_ap'] for v in results.values() if v['mean_ap'] is not None]
    return {'metric':'CARLA common-class all-points interpolated XY center-distance AP',
            'NDS':None,'frames':len(gt),'mean_ap':float(np.mean(aps)) if aps else None,'classes':results},matches

def read_jsonl(path):
    with open(path) as f:return [json.loads(line) for line in f if line.strip()]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gt',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    metrics,matches=evaluate(read_jsonl(a.gt),read_jsonl(a.predictions))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    with (out/'matches.jsonl').open('w') as f:
        for row in matches:f.write(json.dumps(row)+'\n')
    print(json.dumps(metrics,indent=2))

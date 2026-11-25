"""GT-free world-coordinate constant-velocity Hungarian tracker and explicit tracking metrics."""
from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluate import validate_prediction,CLASSES,RANGES
from geometry import transform

class CommonTracker:
    def __init__(self,gate_m=4.,max_age_s=1.,score_threshold=.2):
        self.gate=gate_m;self.max_age=max_age_s;self.score_threshold=score_threshold;self.reset()
    def reset(self):
        self.episode=None;self.timestamp=None;self.tracks={};self.next_id=0
    def update(self,episode,timestamp,detections):
        if self.episode!=episode:
            self.reset();self.episode=episode
        if self.timestamp is not None and timestamp<=self.timestamp:raise ValueError('시간 역전/중복')
        self.timestamp=timestamp
        detections=[deepcopy(b) for b in detections if b['score']>=self.score_threshold]
        for b in detections:
            validate_prediction(b)
            if b['episode_id']!=episode or abs(b['timestamp']-timestamp)>1e-6:raise ValueError('관측 episode/시간 혼합')
        self.tracks={k:v for k,v in self.tracks.items() if timestamp-v['time']<=self.max_age}
        ids=list(self.tracks);cost=np.full((len(ids),len(detections)),1e9)
        for i,k in enumerate(ids):
            t=self.tracks[k];center=t['center']+t['velocity']*(timestamp-t['time'])
            for j,b in enumerate(detections):
                if b['class']==t['class']:
                    distance=np.linalg.norm(center-np.asarray(b['center_xyz'])[:2])
                    if distance<self.gate:cost[i,j]=distance
        assigned={}
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            assigned={int(j):ids[i] for i,j in zip(rows,cols) if cost[i,j]<self.gate}
        output=[]
        for j,b in enumerate(detections):
            k=assigned.get(j)
            if k is None:k=self.next_id;self.next_id+=1
            old=self.tracks.get(k);center=np.asarray(b['center_xyz'])[:2]
            if b.get('velocity_xy') is not None:velocity=np.asarray(b['velocity_xy'])
            elif old is not None:velocity=(center-old['center'])/(timestamp-old['time'])
            else:velocity=np.zeros(2)
            self.tracks[k]={'center':center,'velocity':velocity,'time':timestamp,'class':b['class']}
            b['track_id']=episode+':pred:'+str(k);b['valid_mask']=True;output.append(b)
        return output

def tracking_metrics(gt_frames,pred_frames,gate_m=2.):
    """Simple CLEAR-style MOTA/IDS/FRAG; no AMOTA claim; missing-frame rejection."""
    gt={x['sample_token']:x for x in gt_frames};pred={x['sample_token']:x for x in pred_frames}
    if set(gt)!=set(pred) or len(gt)!=len(gt_frames) or len(pred)!=len(pred_frames):raise ValueError('추적 frame token 불일치')
    previous_id={};seen=set();was_matched={};counts=dict(gt=0,tp=0,fp=0,fn=0,ids=0,fragments=0);distances=[]
    for token,f in sorted(gt.items(),key=lambda kv:(kv[1]['episode_id'],kv[1]['timestamp'])):
        if pred[token]['episode_id']!=f['episode_id'] or abs(pred[token]['timestamp']-f['timestamp'])>1e-6:raise ValueError('추적 episode/시각 불일치')
        boxes=[b for b in f['objects'] if b['valid_mask'] and b['class'] in CLASSES]
        E_G=np.linalg.inv(np.asarray(f['ego_to_world']))
        ps=[]
        for b in pred[token]['objects']:
            validate_prediction(b)
            if b['episode_id']!=f['episode_id'] or abs(b['timestamp']-f['timestamp'])>1e-6:raise ValueError('추적 객체 episode/시각 불일치')
            if b['track_id'] is None:raise ValueError('예측 track ID 없음')
            if b['class'] in CLASSES and np.linalg.norm(transform([b['center_xyz']],E_G)[0,:2])<=RANGES[b['class']]:ps.append(b)
        cost=np.full((len(boxes),len(ps)),1e9)
        pids=[b['track_id'] for b in ps]
        if len(pids)!=len(set(pids)):raise ValueError('프레임 내 예측 ID 중복')
        for i,g in enumerate(boxes):
            for j,p in enumerate(ps):
                if g['class']==p['class']:
                    d=np.linalg.norm(np.asarray(g['center_xyz'])[:2]-np.asarray(p['center_xyz'])[:2])
                    if d<gate_m:cost[i,j]=d
        matched={}
        if cost.size:
            rows,cols=linear_sum_assignment(cost);matched={i:j for i,j in zip(rows,cols) if cost[i,j]<gate_m}
        counts['gt']+=len(boxes);counts['tp']+=len(matched);counts['fn']+=len(boxes)-len(matched)
        ignored=[g for g in f['objects'] if not g['valid_mask']]
        for j,p in enumerate(ps):
            if j in matched.values():continue
            duplicate=any(cost[i,j]<gate_m for i in range(len(boxes)))
            near_ignore=any(g['class']==p['class'] and np.linalg.norm(np.asarray(g['center_xyz'])[:2]-np.asarray(p['center_xyz'])[:2])<gate_m for g in ignored)
            if duplicate or not near_ignore:counts['fp']+=1
        for i,g in enumerate(boxes):
            k=(f['episode_id'],g['track_id'])
            if i in matched:
                j=matched[i];p=ps[j];distances.append(float(cost[i,j]))
                if k in previous_id and previous_id[k]!=p['track_id']:counts['ids']+=1
                if k in seen and not was_matched.get(k,False):counts['fragments']+=1
                previous_id[k]=p['track_id'];seen.add(k);was_matched[k]=True
            else:was_matched[k]=False
    counts['MOTA']=1-(counts['fn']+counts['fp']+counts['ids'])/counts['gt'] if counts['gt'] else None
    counts['mean_matched_distance_m']=float(np.mean(distances)) if distances else None
    return counts

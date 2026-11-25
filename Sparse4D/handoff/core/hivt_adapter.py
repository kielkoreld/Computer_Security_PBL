"""History-only numeric adapter matching official HiVT fields; map metadata is mandatory."""
from itertools import permutations
import numpy as np

def history_tensors(records,episode_id,end_timestamp,av_track_id,agent_track_id,dt=.1):
    if abs(dt-.1)>1e-9:raise ValueError('공식 10Hz 계약: 2Hz 결과를 관측 10Hz로 위장할 수 없습니다')
    records=[r for r in records if r['timestamp']<=end_timestamp+1e-7 and r['timestamp']>=end_timestamp-19*dt-1e-7]
    if not records:raise ValueError('이력 없음')
    if any(r['episode_id']!=episode_id for r in records):raise ValueError('episode 혼합')
    if any(r['coordinate_frame']!='world_rh' for r in records):raise ValueError('world_rh로 변환 필요')
    ids=sorted({r['track_id'] for r in records})
    if av_track_id not in ids or agent_track_id not in ids:raise ValueError('AV/AGENT 누락')
    index={x:i for i,x in enumerate(ids)};n=len(ids)
    positions=np.zeros((n,50,2),dtype=np.float32);mask=np.ones((n,50),dtype=bool);seen=set()
    for r in records:
        t=(r['timestamp']-(end_timestamp-19*dt))/dt;step=int(round(t))
        if abs(t-step)>1e-5 or not 0<=step<20:raise ValueError('0.1초 격자 불일치')
        i=index[r['track_id']]
        if (i,step) in seen:raise ValueError('중복 ID/시각')
        seen.add((i,step))
        if r['valid_mask']:
            xy=np.asarray(r['center_xyz'])[:2]
            if not np.isfinite(xy).all():raise ValueError('좌표 NaN')
            positions[i,step]=xy;mask[i,step]=False
    av=index[av_track_id]
    if mask[av,:20].any():raise ValueError('20개 AV pose 필요')
    origin=positions[av,19].astype(np.float64);heading=positions[av,19]-positions[av,18]
    if np.linalg.norm(heading)<1e-4:raise ValueError('정지 AV의 이동 heading 불명확: 별도 pose-yaw 정책 필요')
    theta=float(np.arctan2(heading[1],heading[0]));c,s=np.cos(theta),np.sin(theta)
    rotate=np.array([[c,-s],[s,c]])
    for i in range(n):
        valid=~mask[i,:20];positions[i,:20][valid]=(positions[i,:20][valid]-origin)@rotate
    x=np.zeros((n,20,2),dtype=np.float32)
    valid_pair=~(mask[:,:19]|mask[:,1:20]);x[:,1:]=np.where(valid_pair[...,None],positions[:,1:20]-positions[:,:19],0)
    bos=np.zeros((n,20),dtype=bool);bos[:,0]=~mask[:,0];bos[:,1:]=mask[:,:19]&~mask[:,1:20]
    angles=np.zeros(n,dtype=np.float32);eligible=np.zeros(n,dtype=bool)
    for i in range(n):
        valid=np.where(~mask[i,:20])[0]
        if len(valid)>1:
            vec=positions[i,valid[-1]]-positions[i,valid[-2]];angles[i]=np.arctan2(vec[1],vec[0])
            eligible[i]=not mask[i,19]
    edges=np.asarray(list(permutations(range(n),2)),dtype=np.int64).reshape(-1,2).T
    return {'x':x,'positions':positions,'padding_mask':mask,'bos_mask':bos,'rotate_angles':angles,
            'edge_index':edges,'num_nodes':n,'av_index':av,'agent_index':index[agent_track_id],
            'origin':origin[None],'theta':theta,'y':None,'track_ids':ids,'prediction_eligible':eligible,
            'episode_id':episode_id,'end_timestamp':end_timestamp}

def attach_map(history,map_features):
    if map_features is None:raise ValueError('실제 차선·교차로·회전·교통제어 지도 입력이 없습니다')
    required=['lane_vectors','is_intersections','turn_directions','traffic_controls','lane_actor_index','lane_actor_vectors']
    if any(k not in map_features for k in required):raise ValueError('HiVT 지도 필드 결손')
    if map_features.get('coordinate_frame')!='hivt_local':raise ValueError('지도 로컬 좌표 변환 필요')
    if map_features.get('episode_id')!=history['episode_id'] or abs(map_features.get('end_timestamp',float('inf'))-history['end_timestamp'])>1e-6:raise ValueError('지도 episode/시각 불일치')
    n=len(map_features['lane_vectors']);e=np.asarray(map_features['lane_actor_index'])
    assert np.asarray(map_features['lane_vectors']).shape==(n,2)
    assert e.ndim==2 and e.shape[0]==2 and np.issubdtype(e.dtype,np.integer)
    assert np.asarray(map_features['lane_actor_vectors']).shape==(e.shape[1],2)
    for k in ['is_intersections','turn_directions','traffic_controls']:assert np.asarray(map_features[k]).shape==(n,)
    assert np.isin(map_features['is_intersections'],[0,1]).all() and np.isin(map_features['traffic_controls'],[0,1]).all()
    assert np.isin(map_features['turn_directions'],[0,1,2]).all()
    if e.size:assert e.min()>=0 and e[0].max()<n and e[1].max()<history['num_nodes']
    for k in ['lane_vectors','lane_actor_vectors']:assert np.isfinite(map_features[k]).all()
    return {**history,**{k:np.asarray(map_features[k]) for k in required}}

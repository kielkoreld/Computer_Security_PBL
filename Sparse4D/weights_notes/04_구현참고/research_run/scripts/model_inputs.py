"""GT-free NumPy input preparation; source-compatible array layout, not a validated GPU runner."""
from pathlib import Path
import json
import pickle
from bisect import bisect_right
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from geometry import SPARSE_CAMS,BEV_CAMS,augmentation_matrix,transform

MEAN=np.array([123.675,116.28,103.53],dtype=np.float32)
STD=np.array([58.395,57.12,57.375],dtype=np.float32)
CLASS_NAMES=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
PROFILE_EVENTS=None

def T(q,t):
    out=np.eye(4);out[:3,:3]=Rotation.from_quat(np.array(q)[[1,2,3,0]]).as_matrix();out[:3,3]=t;return out

def image_input(path,reverse_channels=False):
    start=time.perf_counter() if PROFILE_EVENTS is not None else None
    im=Image.open(path).convert('RGB')
    decoded=time.perf_counter() if PROFILE_EVENTS is not None else None
    if im.size!=(1600,900):raise ValueError('고정된 영상 크기와 다릅니다')
    # Both published pipelines use PIL resize default for this transform.
    rgb=np.asarray(im.resize((704,396)).crop((0,140,704,396)),dtype=np.float32)
    normalized_rgb=rgb[...,::-1] if reverse_channels else rgb
    normalized=((normalized_rgb-MEAN)/STD).transpose(2,0,1).copy()
    if PROFILE_EVENTS is not None:
        PROFILE_EVENTS.append({'read_decode_s':decoded-start,'image_preprocess_s':time.perf_counter()-decoded})
    return normalized,rgb

def check_sequence(infos):
    if not infos:raise ValueError('빈 episode')
    ep=infos[0]['scene_token'];prev=None
    for f in infos:
        if f['scene_token']!=ep:raise ValueError('episode 혼합')
        if prev is not None and f['timestamp']<=prev:raise ValueError('시간 역전/중복')
        prev=f['timestamp']
        if any(k.startswith('gt_') or k=='ann_infos' for k in f):raise ValueError('모델 metadata에 GT가 섞임')
        if 'ego2global_rotation' not in f or 'lidar2ego_rotation' not in f:raise ValueError('pose 누락')

def sparse_input(info):
    if list(info['cams'])!=SPARSE_CAMS:raise ValueError('Sparse4D 카메라 순서 오류')
    imgs=[];projections=[];aug=augmentation_matrix()
    for cam in SPARSE_CAMS:
        c=info['cams'][cam];im,_=image_input(c['data_path']);imgs.append(im)
        L_C=np.eye(4);L_C[:3,:3]=c['sensor2lidar_rotation'];L_C[:3,3]=c['sensor2lidar_translation']
        K4=np.eye(4);K4[:3,:3]=c['cam_intrinsic'];projections.append(aug@K4@np.linalg.inv(L_C))
    G_E=T(info['ego2global_rotation'],info['ego2global_translation']);E_L=T(info['lidar2ego_rotation'],info['lidar2ego_translation'])
    G_L=G_E@E_L
    return {'img':np.stack(imgs)[None],'projection_mat':np.asarray(projections,dtype=np.float32)[None],
            'image_wh':np.array([[[704,256]]*6],dtype=np.float32),'timestamp':np.array([info['timestamp']/1e6]),
            'T_global':G_L[None],'T_global_inv':np.linalg.inv(G_L)[None]}

def history_indices(infos,index):
    times=[f['timestamp'] for f in infos[:index+1]]
    return [max(0,bisect_right(times,times[-1]-i*500000)-1) for i in range(9)]

def bev_input(infos,index):
    check_sequence(infos);current=infos[index]
    if list(current['cams'])!=BEV_CAMS:raise ValueError('BEVNeXt 카메라 순서 오류')
    history=[infos[i] for i in history_indices(infos,index)]
    G_Ekey=T(current['ego2global_rotation'],current['ego2global_translation'])
    imgs=[];canvas=[]
    for cam in BEV_CAMS:
        for j,f in enumerate(history):
            # Official BEVNeXt mmlabNormalize applies imnormalize(to_rgb=True)
            # to a PIL RGB array, so its checkpoint expects this R/B reversal.
            im,rgb=image_input(f['cams'][cam]['data_path'],reverse_channels=True);imgs.append(im)
            if j==0:canvas.append(rgb)
    rots=[];trans=[];intrins=[];post_rots=[];post_trans=[];sensor2sensors=[]
    for f in history:
        G_E=T(f['ego2global_rotation'],f['ego2global_translation'])
        for cam in BEV_CAMS:
            c=f['cams'][cam];E_C=T(c['sensor2ego_rotation'],c['sensor2ego_translation'])
            Ekey_C=np.linalg.inv(G_Ekey)@G_E@E_C
            key=current['cams'][cam];Ekey_Ckey=T(key['sensor2ego_rotation'],key['sensor2ego_translation'])
            sensor2sensors.append(np.linalg.inv(Ekey_C)@Ekey_Ckey)
            rots.append(Ekey_C[:3,:3]);trans.append(Ekey_C[:3,3]);intrins.append(c['cam_intrinsic'])
            post_rots.append(np.diag([.44,.44,1.]));post_trans.append([0,-140,0])
    return {'imgs':np.stack(imgs)[None],'rots':np.asarray(rots,dtype=np.float32)[None],
            'trans':np.asarray(trans,dtype=np.float32)[None],'intrins':np.asarray(intrins,dtype=np.float32)[None],
            'post_rots':np.asarray(post_rots,dtype=np.float32)[None],'post_trans':np.asarray(post_trans,dtype=np.float32)[None],
            'bda':np.eye(3,dtype=np.float32)[None],'sensor2sensors':np.asarray(sensor2sensors,dtype=np.float32)[None],
            'canvas':np.asarray(canvas,dtype=np.float32)[None],
            'timestamps_s':np.array([f['timestamp']/1e6 for f in history]),'history_ready':np.asarray(current['timestamp']-infos[0]['timestamp']>=4000000)}

def export_boxes(boxes,scores,labels,info,model,native_ids=None):
    """boxes: decoded [x,y,z,l,w,h,yaw,vx,vy,(vz)] array; BEV raw tensor uses bottom center."""
    boxes=np.asarray(boxes).copy();scores=np.asarray(scores);labels=np.asarray(labels)
    if boxes.ndim!=2 or boxes.shape[1]<9 or len(boxes)!=len(scores) or len(boxes)!=len(labels):raise ValueError('출력 형상 오류')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('출력 NaN/Inf')
    if not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=len(CLASS_NAMES)):raise ValueError('클래스 인덱스 오류')
    if np.any(boxes[:,3:6]<=0) or np.any(scores<0) or np.any(scores>1):raise ValueError('크기/score 오류')
    G_E=T(info['ego2global_rotation'],info['ego2global_translation'])
    if model=='sparse4d':G_M=G_E@T(info['lidar2ego_rotation'],info['lidar2ego_translation'])
    elif model=='bevnext':G_M=G_E;boxes[:,2]+=boxes[:,5]/2
    else:raise ValueError('모델 종류 오류')
    centers=transform(boxes[:,:3],G_M);rows=[]
    for i,b in enumerate(boxes):
        heading=G_M[:3,:3]@np.array([np.cos(b[6]),np.sin(b[6]),0.])
        velocity=G_M[:3,:3]@np.r_[b[7:9],b[9] if len(b)>9 else 0.]
        native=None if native_ids is None or int(native_ids[i])<0 else info['scene_token']+':native:'+str(int(native_ids[i]))
        rows.append({'episode_id':info['scene_token'],'timestamp':info['timestamp']/1e6,'track_id':native,
            'class':CLASS_NAMES[int(labels[i])],'center_xyz':centers[i].tolist(),'size_lwh':b[3:6].tolist(),
            'yaw':float(np.arctan2(heading[1],heading[0])),'velocity_xy':velocity[:2].tolist(),
            'score':float(scores[i]),'coordinate_frame':'world_rh','valid_mask':True})
    return {'sample_token':info['token'],'episode_id':info['scene_token'],'timestamp':info['timestamp']/1e6,'objects':rows}

if __name__=='__main__':
    R=Path(__file__).resolve().parents[1];split=json.loads((R/'manifests/split.json').read_text());ep=split['development'][0]
    checked=0
    for path in (R/'data/converted_v2').glob('*/*_infos.pkl'):
        with path.open('rb') as f:infos=pickle.load(f)['infos']
        check_sequence(infos);checked+=1
    folder=R/'data/converted_v2'/ep
    with (folder/'sparse4d_infos.pkl').open('rb') as f:sp=pickle.load(f)['infos']
    with (folder/'bevnext_infos.pkl').open('rb') as f:be=pickle.load(f)['infos']
    a=sparse_input(sp[8]);b=bev_input(be,8)
    assert all(np.isfinite(x).all() for x in list(a.values())+list(b.values()))
    out=R/'artifacts/validation';np.savez_compressed(out/'sparse4d_real_input.npz',**a);np.savez_compressed(out/'bevnext_real_input.npz',**b)
    report={'checked_episode_model_files':checked,'episode_id':ep,'frame_id':40,
       'sparse4d_shapes':{k:list(v.shape) for k,v in a.items()},'bevnext_shapes':{k:list(v.shape) for k,v in b.items()},
       'bevnext_history_timestamps_s':b['timestamps_s'].tolist(),'finite':True,'gt_free':True,
       'scope':'NumPy 입력 구성 검증. 공식 모델 forward 및 custom CUDA op 검증은 별도 필요'}
    (out/'model_input_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))

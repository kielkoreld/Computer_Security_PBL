"""Verified geometry, GT sidecars and GT-free native model metadata for saved CARLA episodes."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from geometry import *

R=Path(__file__).resolve().parents[1]
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-tag',default='converted_v2')
    parser.add_argument('--frame-step',type=int,choices=[1,5],default=5)
    parser.add_argument('--episode')
    parser.add_argument('--extracted-root',type=Path,default=R/'data/extracted')
    parser.add_argument('--audit',type=Path,default=R/'artifacts/validation/archive_audit.json')
    parser.add_argument('--split-manifest',type=Path,default=R/'protocol/split.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--records-root',type=Path,required=True)
    args=parser.parse_args()
    output_root=args.output_root
    if args.episode is None:raise ValueError('확장 변환은 --episode를 명시한다')
    if (output_root/args.episode).exists():raise FileExistsError('기존 episode 변환을 보존합니다')
    split=json.loads(args.split_manifest.read_text())
    audit=json.loads(args.audit.read_text())
    assert not audit['failures']
    reports=[];manifest=[]
    for epinfo in audit['episodes']:
        if args.episode and epinfo['episode_id']!=args.episode:continue
        ep=epinfo['episode_id'];root=args.extracted_root/ep
        annos=[]
        for p in sorted((root/'anno').glob('*.json.gz')):
            with gzip.open(p,'rt') as f:annos.append((int(p.name.split('.')[0]),json.load(f)))
        gt=[];sparse=[];bev=[];poses_rows=[];checks=defaultdict(list);counts=Counter();prev={};static=defaultdict(list)
        for frame,a in annos:
            G_E,E_L,G_L=poses(a);L_U=np.linalg.inv(G_L)@S
            for T in [G_E,E_L,G_L]:
                checks['rotation_orthogonality'].append(float(np.max(np.abs(T[:3,:3].T@T[:3,:3]-np.eye(3)))))
                checks['rotation_determinant'].append(abs(float(np.linalg.det(T[:3,:3]))-1))
                checks['matrix_inverse_roundtrip'].append(float(np.max(np.abs(np.linalg.inv(T)@T-np.eye(4)))))
            cams={}; mats={}
            for cam in SPARSE_CAMS:
                C_L,E_C,K,P=camera_matrices(a,cam);mats[cam]=(C_L,K,P)
                assert a['sensors'][cam]['image_size_x']==1600 and a['sensors'][cam]['image_size_y']==900
                L_C=np.linalg.inv(C_L);path=root/'camera'/cam.replace('CAM_','rgb_').lower()/f'{frame:05}.jpg'
                assert path.exists()
                cams[cam]={'data_path':str(path),'cam_intrinsic':K,'sensor2lidar_rotation':L_C[:3,:3],
                    'sensor2lidar_translation':L_C[:3,3],'sensor2ego_rotation':quaternion(E_C),
                    'sensor2ego_translation':E_C[:3,3].tolist(),'ego2global_rotation':quaternion(G_E),
                    'ego2global_translation':G_E[:3,3].tolist(),'timestamp':frame*100000}
                # Camera origin -> global location is independent cross-check of transform direction.
                camera_u=transform(np.zeros((1,3)),np.linalg.inv(A@np.asarray(a['sensors'][cam]['world2cam'])))[0]
                checks['camera_location_error_m'].append(float(np.linalg.norm(camera_u-a['sensors'][cam]['location'])))
            ids=[b['id'] for b in a['bounding_boxes']]
            assert len(ids)==len(set(ids)),(ep,frame,'중복 GT ID')
            objects=[];current={}
            for b in a['bounding_boxes']:
                cls=map_class(b)
                if b['class'] in ['traffic_sign','traffic_light']:
                    static[b['id']].append(np.asarray(b.get('center',b['location'])))
                if cls is None:
                    counts['unmapped/'+b['class']+'/'+str(b.get('base_type'))]+=1;continue
                if 'world_cord' not in b:
                    counts['missing_world_cord']+=1;continue
                center,size,heading,corners=box_geometry(b)
                checks['center_vs_corners_m'].append(float(np.linalg.norm(center-b['center'])))
                checks['extent_vs_corners_m'].append(float(np.max(np.abs(size-2*np.asarray(b['extent'])))))
                for cam,(C_L,K,P) in mats.items():
                    lpts=transform(corners,L_U)
                    uv,d=project(lpts,P)
                    K4=np.eye(4);K4[:3,:3]=K
                    uv2,d2=project(corners,K4@A@np.asarray(a['sensors'][cam]['world2cam']))
                    mask=(d>0.1)&(d2>0.1)
                    if mask.any():checks['projection_path_difference_px'].append(float(np.max(np.abs(uv[mask]-uv2[mask]))))
                    Q=augmentation_matrix();uv3,_=project(lpts,Q@P)
                    if mask.any():checks['augmentation_difference_px'].append(float(np.max(np.abs(uv3[mask]-(uv[mask]*.44-[0,140])))))
                cG=transform(center[None],S)[0];cE=transform(center[None],np.linalg.inv(G_E)@S)[0]
                cL=transform(center[None],L_U)[0];hG=S[:3,:3]@heading
                range_m=float(np.linalg.norm(cE[:2]));in_frustum=any(in_camera(transform(corners,L_U),C,K) for C,K,P in mats.values())
                max_range=40 if cls in ['pedestrian','bicycle','motorcycle'] else 50
                reasons=[]
                if b.get('state')=='static':reasons.append('static_geometry_unverified')
                if range_m>max_range:reasons.append('out_of_range')
                if not in_frustum:reasons.append('out_of_camera_frusta')
                if b.get('num_points',-1)<=0:reasons.append('not_lidar_observed')
                # Static meshes without base_type are already unmapped; unsupported labels remain documented.
                vel=None;vel_valid=False
                old=prev.get(b['id']);current[b['id']]=(frame,cG)
                bad_ids=set(map(str,a.get('teleport_ids',[])+a.get('disappear_ids',[])))
                if old is not None and old[0]==frame-1 and b['id'] not in bad_ids:
                    v=(cG-old[1])/.1
                    if np.linalg.norm(v)<80:vel=v[:2].tolist();vel_valid=True
                obj={'episode_id':ep,'timestamp':frame*.1,'track_id':ep+':'+b['id'],'class':cls,
                    'center_xyz':cG.tolist(),'center_lidar':cL.tolist(),'size_lwh':size.tolist(),
                    'yaw':float(np.arctan2(hG[1],hG[0])),'velocity_xy':vel,'velocity_valid':vel_valid,
                    'score':1.,'coordinate_frame':'world_rh','valid_mask':not reasons,'ignore_reasons':reasons,
                    'range_m':range_m,'num_lidar_points':b.get('num_points'),
                    'camera_frustum_overlap':in_frustum,'camera_occlusion':'unknown','source_type_id':b.get('type_id')}
                objects.append(obj);counts[cls+'/all']+=1
                if not reasons:counts[cls+'/valid']+=1
                for reason in reasons:counts['ignore/'+reason]+=1
            prev=current
            ego=[b for b in a['bounding_boxes'] if b['class']=='ego_vehicle'][0]
            poses_rows.append({'frame_id':frame,'timestamp':frame*.1,'ego_to_world':G_E.tolist(),
                               'lidar_to_world':G_L.tolist(),'lidar_to_ego':E_L.tolist(),'ego_forward_speed':ego['speed']})
            if frame%args.frame_step:continue
            token=ep+'/'+f'{frame:05}'
            gt.append({'sample_token':token,'episode_id':ep,'frame_id':frame,'timestamp':frame*.1,
                       'objects':objects,'weather':a['weather'],'ego_to_world':G_E.tolist(),
                       'history_ready':frame>=40,'split':epinfo['split']})
            info={'token':token,'scene_token':ep,'timestamp':frame*100000,'cams':cams,
                  'lidar_path':'virtual-reference/'+ep+'/'+f'{frame:05}', 'sweeps':[],
                  'lidar2ego_translation':E_L[:3,3].tolist(),'lidar2ego_rotation':quaternion(E_L),
                  'ego2global_translation':G_E[:3,3].tolist(),'ego2global_rotation':quaternion(G_E)}
            sparse.append(info)
            # BEVNeXt projects to right-handed ego coordinates, not Sparse4D reference LiDAR.
            binfo=dict(info);binfo['cams']={c:cams[c] for c in BEV_CAMS}
            bev.append(binfo)
            manifest.append({'sample_token':token,'split':epinfo['split'],'timestamp':frame*.1,'history_ready':frame>=40})
        # Check timing independently using true ego pose displacement vs recorded forward speed.
        for last,cur in zip(poses_rows,poses_rows[1:]):
            v=(last['ego_forward_speed']+cur['ego_forward_speed'])/2
            if abs(v)>2:
                distance=np.linalg.norm(np.asarray(cur['ego_to_world'])[:3,3]-np.asarray(last['ego_to_world'])[:3,3])
                checks['inferred_dt_from_ego_motion_s'].append(float(distance/abs(v)))
        for points in static.values():
            if len(points)>1:checks['static_world_span_m'].append(float(np.max(np.linalg.norm(np.asarray(points)-points[0],axis=1))))
        out=output_root/ep;out.mkdir(parents=True,exist_ok=True)
        for name,infos in [('sparse4d',sparse),('bevnext',bev)]:
            with (out/(name+'_infos.pkl')).open('wb') as f:pickle.dump({'infos':infos,'metadata':{'version':'carla-b2d-v0.0.4','ground_truth_free':True}},f)
        with (out/'gt_common.jsonl').open('w') as f:
            for x in gt:f.write(json.dumps(x,allow_nan=False)+'\n')
        dump(out/'poses_10hz.json',poses_rows)
        summary={k:{'count':len(v),'max':max(v),'median':float(np.median(v)),'p95':float(np.percentile(v,95))} for k,v in checks.items() if v}
        report={'episode_id':ep,'split':epinfo['split'],'frames_10hz':len(annos),
                'frames_2hz':sum(frame%5==0 for frame,_ in annos),'selected_frames':len(gt),'selected_hz':10/args.frame_step,
                'checks':summary,'counts_10hz':dict(counts)}
        dump(out/'conversion_report.json',report);reports.append(report)
        print(ep,len(annos),len(gt),flush=True)
    if not reports:raise ValueError('요청한 episode가 audit에 없습니다')
    dump(args.records_root/(args.episode+'_frames.json'),manifest)
    dump(args.records_root/(args.episode+'_conversion.json'),reports)

if __name__=='__main__':main()

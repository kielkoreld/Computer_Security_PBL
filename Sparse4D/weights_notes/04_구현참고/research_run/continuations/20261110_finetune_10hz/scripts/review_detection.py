"""저장 예측을 독립 경로로 읽어 범위·시간·개수·공식 AP 곡선과 경로별 AP 재검수."""
import argparse,collections,json,math,sys
import numpy as np
from pyquaternion import Quaternion
from common import *
sys.path.insert(0,str(R/'continuations/20261102_nuscenes44/scripts'))
import nu_metrics as nm

def convert(o,token,ego,truth):
 c=o['class']
 if c not in nm.CLASSES or truth and not o['valid_mask']:return None
 rel=np.asarray(o['center_xyz'],dtype=float)-ego
 if np.hypot(rel[0],rel[1])>=nm.RANGES[c]:return None
 l,w,h=o['size_lwh']
 return nm.make_box(dict(sample_token=token,translation=o['center_xyz'],size=[w,l,h],rotation=Quaternion(axis=[0,0,1],angle=o['yaw']).elements.tolist(),velocity=o['velocity_xy'] if not truth or o['velocity_valid'] else None,ego_translation=rel.tolist(),num_pts=o.get('num_lidar_points',-1),detection_name=c,detection_score=o['score']),truth)

def one(ep,out):
 source=P/'cache'/ep
 if not source.exists():source=OLD10/'cache'/ep
 G={};B={};GC={};BC={};all_objects=0;inside=collections.Counter();validGT=collections.Counter()
 with (source/'gt_common.jsonl').open() as f,(out/'predictions'/ep/'predictions.jsonl').open() as p:
  for i,gline in enumerate(f):
   line=p.readline();assert line,('예측 프레임 부족',ep,i)
   g=json.loads(gline);a=json.loads(line);tok=g['sample_token']
   assert tok==a['sample_token']==ep+f'/{i:05d}' and tok not in G
   assert abs(g['timestamp']-i*.1)<1e-8 and abs(a['timestamp']-g['timestamp'])<1e-8
   ego=np.asarray(g['ego_to_world'],dtype=float)[:3,3];G[tok]=[];B[tok]=[]
   assert len(a['objects'])<=500
   for truth,objects,target in [(True,g['objects'],G[tok]),(False,a['objects'],B[tok])]:
    for o in objects:
     if not truth:
      assert np.isfinite([*o['center_xyz'],*o['size_lwh'],o['yaw'],*o['velocity_xy'],o['score']]).all()
      assert all(x>0 for x in o['size_lwh']) and 0<=o['score']<=1
      all_objects+=1
     box=convert(o,tok,ego,truth)
     if box is not None:
      target.append(box);(validGT if truth else inside)[o['class']]+=1
   if i%5==0:GC[tok]=G[tok];BC[tok]=B[tok]
  assert not p.readline(),('예측 프레임 초과',ep)
 actual,curves=nm.official(G,B,curves=True);common,_=nm.official(GC,BC)
 save(out/'review/routes'/f'{ep}.json',dict(all=actual,common=common,all_saved_objects=all_objects,valid_GT=dict(validGT),prediction_inside_range=dict(inside)))
 return dict(episode=ep,all=actual,common=common,all_saved_objects=all_objects)

def curve_review(out,rows,common=False):
 folder=out/('metrics_common2hz' if common else 'metrics');pooled=read(folder/'pooled.json');which='common' if common else 'all';errors=[]
 assert pooled['frames']==sum(r[which]['frames'] for r in rows)
 for c,result in pooled['classes'].items():
  for count in ['gt','predictions','valid_velocity_GT']:assert result[count]==sum(r[which]['classes'][c][count] for r in rows),(out,c,count)
  if not result['gt']:
   assert result['mean_ap'] is None and all(v is None for v in result['threshold_ap'].values());continue
  curves=read(folder/f'curves_{c}.json') if (folder/f'curves_{c}.json').exists() else read(folder/'curves.json')
  for threshold,value in result['threshold_ap'].items():
   precision=np.asarray(curves[c+'_'+threshold]['precision']);assert precision.shape==(101,) and np.isfinite(precision).all()
   ap=float(np.maximum(precision[11:]-.1,0).mean()/.9);errors.append(abs(ap-value));assert abs(ap-value)<1e-12
  assert abs(result['mean_ap']-np.mean(list(result['threshold_ap'].values())))<1e-12
 vals=[r['mean_ap'] for r in pooled['classes'].values() if r['gt']]
 assert abs(pooled['mean_ap']-np.mean(vals))<1e-12
 return dict(frames=pooled['frames'],GT_classes=[c for c,x in pooled['classes'].items() if x['gt']],mean_ap=pooled['mean_ap'],maximum_AP_reconstruction_error=max(errors,default=0),same_pooled_GT_and_prediction_counts=True,official_precision_curve_AP_recomputed=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--tags',nargs='+',required=True);args=a.parse_args();summaries={};perroute={};eps=read(P/'protocol/split.json')['evaluation44']
 assert len(args.tags)>=2 and args.tags[0]=='baseline_44_10hz_dt01'
 assert 'pilot_v2_best_44_10hz' in args.tags or 'final_44_10hz' in args.tags,'기준모델만의 검수로 단계 통과 불가'
 for tag in args.tags:
  out=P/'evaluations'/tag;inf=read(out/'inference.json');assert inf['complete'] and inf['frames']==11645 and len(inf['runs'])==44
  assert {r['episode'] for r in inf['runs']}==set(eps)
  assert all(r['precision']=='fp32' and r['default_dt']==.1 and r['step']==1 and not r['no_history'] and not r['gt_used_by_inference'] for r in inf['runs'])
  rows=[]
  for ep in eps:
   path=out/'review/routes'/f'{ep}.json'
   if path.exists():rows.append(dict(episode=ep,**read(path)))
   else:rows.append(one(ep,out))
  summaries[tag]=dict(all=curve_review(out,rows),common=curve_review(out,rows,True),all_saved_objects=sum(r['all_saved_objects'] for r in rows));perroute[tag]={r['episode']:r['all']['mean_ap'] for r in rows}
  save(out/'review/summary.json',dict(pass_=True,results=summaries[tag],routes=perroute[tag],reviewed_utc=now()))
  print('저장 예측 전수 및 지표 검수',tag,summaries[tag]['all']['mean_ap'],flush=True)
 baseline=args.tags[0];comparisons={}
 for tag in args.tags[1:]:
  for folder in ['metrics','metrics_common2hz']:
   b=read(P/'evaluations'/baseline/folder/'pooled.json');v=read(P/'evaluations'/tag/folder/'pooled.json')
   assert b['frames']==v['frames'] and all(b['classes'][c]['gt']==v['classes'][c]['gt'] for c in b['classes'])
  delta=np.asarray([perroute[tag][e]-perroute[baseline][e] for e in eps if perroute[tag][e] is not None and perroute[baseline][e] is not None]);rng=np.random.default_rng(0)
  comparisons[tag]=dict(pooled_AP_change_pp=100*(summaries[tag]['all']['mean_ap']-summaries[baseline]['all']['mean_ap']),common2hz_timestamp_AP_change_pp=100*(summaries[tag]['common']['mean_ap']-summaries[baseline]['common']['mean_ap']),paired_route_macro_AP_change_pp=100*float(delta.mean()),paired_route_macro_bootstrap_95pct_pp=(100*np.percentile(rng.choice(delta,(2000,len(delta)),replace=True).mean(1),[2.5,97.5])).tolist(),bootstrap_seed=0,bootstrap_resamples=2000,note='경로 평균 AP의 구간이며 전체 pooled mAP의 신뢰구간이 아니다. 반복실행 변동도 반영하지 않는다.')
 label='final' if 'final_44_10hz' in args.tags else 'intermediate'
 save(P/'records'/f'{label}_detection_review.json',dict(pass_=True,results=summaries,comparison_to=baseline,comparisons=comparisons,source_files_reopened=True,scope='저장된 모든 예측의 시간·유한성·범위와 독립 변환 경로별 공식 AP, pooled 곡선 집계와 GT/예측 개수를 검수. pooled 매칭을 다른 알고리즘으로 재구현한 검사는 아님.',finished_utc=now()))
 if label=='intermediate':gate('05b_intermediate_test_review',dict(saved_outputs_valid=True,evaluation_scope_identical=True,all_AP_arithmetic_valid=True),dict(results=summaries,does_not_change_frozen_training_protocol=True))
 else:
  assert read(P/'records/main_supervision.json')['complete']
  expected=str((P/'training/best.pth').resolve());assert read(P/'evaluations/final_44_10hz/contract.json')['checkpoint']==expected
  gate('08_final_test44',dict(saved_outputs_valid=True,evaluation_scope_identical=True,selected_checkpoint_used=True,all_AP_arithmetic_valid=True),dict(results=summaries,comparisons=comparisons))
if __name__=='__main__':main()

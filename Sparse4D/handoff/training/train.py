"""인계용 2Hz 학습. 원래 학습 수식 보존; 호스트 운영 관문/비교 probe 제외. 경로 그룹 경계 체크포인트, 신호 보존, patience5."""
import argparse,hashlib,json,math,os,pickle,random,signal,sys,time
from pathlib import Path
import numpy as np
import torch
from paths import Q,R,S,ROOT,read,save,now,progress
from engine import seed,build,optimizer,groups,Group,Transfer,train_group
stop_requested=False
def on_signal(sig,frame):
 global stop_requested
 stop_requested=True
 print('중단 요청 수신: 현재 경로 그룹 종료 직후 가중치·optimizer·RNG를 저장합니다.',flush=True)
for s in [signal.SIGINT,signal.SIGTERM]:signal.signal(s,on_signal)

def atomic_model(p,d):
 t=p.with_suffix('.tmp');torch.save(d,t);t.replace(p)
def aggregate(rows):
 n=sum(x['frames'] for x in rows);terms={}
 for r in rows:
  for k,v in r['components'].items():terms[k]=terms.get(k,0.)+v*r['frames']
 return dict(frames=n,loss=sum(x['loss']*x['frames'] for x in rows)/n,components={k:v/n for k,v in terms.items()},elapsed_s=sum(x['elapsed_s'] for x in rows),mean_frame_s=sum(x['elapsed_s'] for x in rows)/n)

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after-groups',type=int,default=0);a=p.parse_args()
 batch=4;split=read(Q/'protocol/split.json');out=Q/('training_smoke' if a.smoke else 'training');out.mkdir(exist_ok=True)
 if a.smoke:
  split={'train':read(Q/'protocol/pilot.json')['episodes'],'validation':read(Q/'protocol/pilot.json')['episodes'][:1]}
 protocol=dict(gt='static_inclusive_v1',input='official Sparse4D test_pipeline 704x256 fixed augmentation, exact RGB LUT cache',batch_size_max=batch,effective_batch_target=8,tail_gradient='group-boundary remainder normalized by actual frame count',validation_batch=1,precision='FP32',seed=0,optimizer='AdamW',lr=2e-5,backbone_multiplier=.1,weight_decay=.001,clip=25,early_stopping_patience=5,min_delta=0,max_epochs=1 if a.smoke else 100,selection='minimum validation loss including pretrained epoch0',initial_checkpoint=str(ROOT/'checkpoints/sparse4dv3_r50.pth'),temporal='same-length independent routes, each stream chronological; reset at group boundary',caveat='batch changes BN and positive-count normalization; not numerically identical to prior batch1 optimization. camera occlusion unverified; other GT masks retained.')
 fingerprint={str(p.relative_to(Q)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Q/'training/train.py',Q/'training/engine.py',Q/'training/model_builder.py',Q/'training/prepare_gt.py',Q/'training/cache_inputs.py',Q/'protocol/split.json',Q/'protocol/loss_comparison.json']};fingerprint['protocol']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
 seed();torch.set_num_threads(4);model,cfg=build();optim=optimizer(model);transfer=Transfer();cfg.dump(str(out/'model_config.py'));save(out/'protocol.json',protocol)
 state=dict(complete=False,phase='validation',epoch=0,next_group=0,rows=[],epochs=[],best_val_loss=None,best_epoch=None,bad_epochs=0,started_utc=now(),training_frames_total=0,group_events_total=0,protocol=protocol,fingerprints=fingerprint,smoke_only=a.smoke)
 if (out/'last.pth').exists():
  c=torch.load(out/'last.pth',map_location='cpu',weights_only=False);assert c['state']['fingerprints']==fingerprint,'입력/코드/프로토콜 변경된 체크포인트 재개 금지'
  model.load_state_dict(c['state_dict'],strict=True);optim.load_state_dict(c['optimizer']);state=c['state'];torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng']);state['resumed_utc']=now();del c
 if state['complete']:print('이미 완료한 학습입니다.',flush=True);return
 last_saved=time.monotonic();started_groups=0
 def checkpoint(reason):
  nonlocal last_saved
  state.update(updated_utc=now(),checkpoint_reason=reason)
  atomic_model(out/'last.pth',dict(state_dict=model.state_dict(),optimizer=optim.state_dict(),state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
  save(out/'status.json',{k:v for k,v in state.items() if k!='rows'});last_saved=time.monotonic()
 def best(epoch,value):
  path=out/f'best_epoch_{epoch:03d}.pth';atomic_model(path,dict(state_dict=model.state_dict(),epoch=epoch,validation_loss=value,protocol=protocol))
  link=out/'best_next.pth'
  if link.is_symlink():link.unlink()
  link.symlink_to(path.name);link.replace(out/'best.pth')
 try:
  while not state['complete']:
   phase=state['phase'];epoch=state['epoch'];train=phase=='train';eps=split['train' if train else 'validation'];gs=groups(eps,batch if train else 1,epoch if train else 0)
   if train:
    ratio=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
    for g,mult in zip(optim.param_groups,[.1,1.]):g['lr']=2e-5*mult*ratio
   for index in range(state['next_group'],len(gs)):
    group=Group(gs[index]);row=train_group(model,group,optim if train else None,transfer)
    state['rows'].append(row);state['next_group']=index+1;state['group_events_total']+=1;started_groups+=1
    if train:state['training_frames_total']+=row['frames']
    state.update(current_groups=len(gs),current_frames=sum(r['frames'] for r in state['rows']),current_loss=sum(r['loss']*r['frames'] for r in state['rows'])/sum(r['frames'] for r in state['rows']),updated_utc=now())
    print(json.dumps(dict(time=now(),phase=phase,epoch=epoch,group=index+1,groups=len(gs),frames=state['current_frames'],loss=state['current_loss'],frame_s=row['frame_s']),ensure_ascii=False),flush=True)
    save(out/'status.json',{k:v for k,v in state.items() if k!='rows'})
    if stop_requested or (a.stop_after_groups and started_groups>=a.stop_after_groups):
     state['paused']=True;checkpoint('signal_or_smoke_stop_at_group_boundary');return
    if time.monotonic()-last_saved>=300:checkpoint('periodic_group_boundary')
   total=aggregate(state['rows']);assert total['frames']==sum(read(Q/'data'/e/'complete.json')['counts']['frames'] for e in eps)
   state.update(next_group=0,rows=[],paused=False)
   if train:
    state['current_train']=total;state['phase']='validation';checkpoint('train_epoch_complete')
   else:
    value=total['loss'];assert math.isfinite(value);improved=state['best_val_loss'] is None or value<state['best_val_loss']
    if improved:state.update(best_epoch=epoch,best_val_loss=value,bad_epochs=0);best(epoch,value)
    elif epoch>0:state['bad_epochs']+=1
    if epoch==0:state['initial_validation']=total
    else:
     result=dict(epoch=epoch,train=state.pop('current_train'),validation=total,improved=improved,patience_used=state['bad_epochs'],finished_utc=now());state['epochs'].append(result)
     with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    if state['bad_epochs']>=5:state.update(complete=True,stop_reason='val_loss_no_improvement_5_epochs')
    elif epoch>=protocol['max_epochs']:state.update(complete=True,stop_reason='max_epochs')
    else:state.update(phase='train',epoch=epoch+1)
    checkpoint('validation_epoch_complete')
  state['finished_utc']=now();checkpoint('training_complete')
 except BaseException as e:
  # 실패 시 마지막 일관된 그룹 경계 체크포인트 보존. 부분 step을 완료된 것으로 저장하지 않음.
  save(out/'failure.json',dict(time=now(),error=repr(e),epoch=state['epoch'],phase=state['phase'],next_group=state['next_group'],last_consistent_checkpoint=str(out/'last.pth')));raise
if __name__=='__main__':main()

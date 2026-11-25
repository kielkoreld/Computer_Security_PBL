"""10Hz 추가 학습. 시퀀스 그룹 경계 저장/복원, 매 단계 감사, patience5."""
import argparse,hashlib,math,random,signal,time,shutil
import numpy as np
import torch
from common import *
from engine import seed,build,optimizer,groups,Group,Transfer,run_group,reset
STOP=False
def interrupt(sig,frame):
 global STOP
 STOP=True
 print('현재 시퀀스 그룹 종료 후 가중치·optimizer·RNG·진행 상태를 보존합니다.',flush=True)
def atomic_ckpt(p,d):
 t=p.with_name(p.name+'.tmp');torch.save(d,t);t.replace(p)
def aggregate(rows):
 n=sum(x['frames'] for x in rows);assert n>0
 terms={};hist={}
 for r in rows:
  for k,v in r['components'].items():terms[k]=terms.get(k,0)+v*r['frames']
  for k,v in r['effective_batch_histogram'].items():hist[str(k)]=hist.get(str(k),0)+v
 return dict(frames=n,loss=sum(x['loss']*x['frames'] for x in rows)/n,components={k:v/n for k,v in terms.items()},elapsed_s=sum(x['elapsed_s'] for x in rows),updates=sum(x['updates'] for x in rows),effective_batch_histogram=hist)
def main():
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');a.add_argument('--one-pass',action='store_true');a.add_argument('--stop-after-groups',type=int,default=0);args=a.parse_args()
 assert read(P/'gates/04_training_ready.json')['pass_']
 if not args.pilot:assert all(read(P/'gates'/f'{name}.json')['pass_'] for name in ['05_pilot_review','05b_intermediate_test_review','06_full_inputs','07_full_batch_stress'])
 protocol=read(P/'protocol/training.json');precision=protocol['precision'];batch=protocol['batch'];lr=protocol['lr'];split=read(P/'protocol/pilot.json') if args.pilot else read(P/'protocol/split.json')
 out=PILOT if args.pilot else P/'training';out.mkdir(exist_ok=True)
 frozen={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'scripts/train.py',P/'scripts/engine.py',P/'scripts/cache_io.py',P/'scripts/common.py',P/'protocol/training.json',P/'protocol/split.json',P/'protocol/pilot.json']}
 seed(0,precision);torch.set_num_threads(4);m,cfg=build(precision=precision);opt=optimizer(m,lr);tr=Transfer();cfg.dump(str(out/'model_config.py'))
 state=dict(complete=False,phase='validation',epoch=0,next_group=0,rows=[],epochs=[],best_val_loss=None,best_epoch=None,bad_epochs=0,train_frames_total=0,updates_total=0,fingerprints=frozen,started_utc=now(),protocol=protocol,pilot=args.pilot)
 if (out/'last.pth').exists():
  ck=torch.load(out/'last.pth',map_location='cpu',weights_only=False);assert ck['state']['fingerprints']==frozen
  m.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);state=ck['state'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);state['resumed_utc']=now();del ck
 else:
  snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
  for name in ['train.py','engine.py','cache_io.py','common.py']:shutil.copy2(P/'scripts'/name,snapshot/name)
  for name in ['training.json','split.json','pilot.json']:shutil.copy2(P/'protocol'/name,snapshot/name)
 if state['complete']:print('이 분기의 학습은 이미 완료됨',flush=True);return
 saved_at=time.monotonic();started_groups=0
 def checkpoint(reason):
  nonlocal saved_at
  state.update(checkpoint_reason=reason,updated_utc=now())
  atomic_ckpt(out/'last.pth',dict(state_dict=m.state_dict(),optimizer=opt.state_dict(),state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
  save(out/'status.json',{k:v for k,v in state.items() if k!='rows'});saved_at=time.monotonic()
 def best(epoch,value):
  path=out/f'best_epoch_{epoch:03d}.pth';assert not path.exists()
  atomic_ckpt(path,dict(state_dict=m.state_dict(),epoch=epoch,validation_loss=value,protocol=protocol,training_hz=10 if epoch>0 else 2,additional_10hz_training_updates=state['updates_total'],evaluation_hz=10,parent_checkpoint=str(INITIAL),pilot_only=args.pilot))
  link=out/'best_next.pth'
  if link.is_symlink():link.unlink()
  link.symlink_to(path.name);link.replace(out/'best.pth')
 maxepochs=3 if args.pilot else protocol['max_epochs']
 try:
  while not state['complete']:
   epoch=state['epoch'];train=state['phase']=='train';eps=split['train' if train else 'validation'];gs=groups(eps,batch if train else 1,epoch if train else 0)
   # 본 학습/시험 분기 모두 같은 주기 정의. 검증 loss는 FP32로 별도 고정.
   if train:
    ratio=.01+.99*(1+math.cos(math.pi*(epoch-1)/protocol['max_epochs']))/2
    for pg,mult in zip(opt.param_groups,[.1,1]):pg['lr']=lr*mult*ratio
   for gi in range(state['next_group'],len(gs)):
    g=Group(gs[gi]);row=run_group(m,g,opt if train else None,tr);g.close();state['rows'].append(row);state['next_group']=gi+1;started_groups+=1
    if train:state['train_frames_total']+=row['frames'];state['updates_total']+=row['updates']
    state.update(current_groups=len(gs),current_frames=sum(r['frames'] for r in state['rows']),current_loss=sum(r['loss']*r['frames'] for r in state['rows'])/sum(r['frames'] for r in state['rows']),updated_utc=now())
    print(json.dumps(dict(time=now(),phase=state['phase'],epoch=epoch,group=gi+1,groups=len(gs),frames=state['current_frames'],loss=state['current_loss'],updates=state['updates_total'],frame_s=row['frame_s']),ensure_ascii=False),flush=True)
    save(out/'status.json',{k:v for k,v in state.items() if k!='rows'})
    if STOP or args.stop_after_groups and started_groups>=args.stop_after_groups:state['paused']=True;checkpoint('requested_consistent_group_boundary');return
    if time.monotonic()-saved_at>300:checkpoint('periodic_consistent_group_boundary')
   total=aggregate(state['rows']);assert total['frames']==sum(read(P/'cache'/e/'complete.json')['frames'] for e in eps)
   save(out/'passes'/f'epoch_{epoch:03d}_{state["phase"]}.json',dict(summary=total,groups=state['rows'],finished_utc=now()))
   state.update(next_group=0,rows=[],paused=False)
   if train:
    state['current_train']=total;state.update(phase='validation',current_frames=0,current_groups=len(split['validation']),current_loss=None);checkpoint('train_epoch_complete');continue
   value=total['loss'];assert math.isfinite(value);improved=state['best_val_loss'] is None or value<state['best_val_loss']
   if improved:state.update(best_epoch=epoch,best_val_loss=value,bad_epochs=0);best(epoch,value)
   elif epoch>0:state['bad_epochs']+=1
   if epoch==0:state['initial_validation']=total
   else:
    row=dict(epoch=epoch,train=state.pop('current_train'),validation=total,improved=improved,patience_used=state['bad_epochs'],finished_utc=now());state['epochs'].append(row)
    with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   # 가중치 갱신 및 선택은 위 val loss만 사용. 별도 검증 AP는 다음 감독 단계에서 계산.
   audit=dict(pass_=True,epoch=epoch,all_frames_counted=True,finite_loss=True,finite_weights=all(torch.isfinite(v).all().item() for v in m.state_dict().values() if v.is_floating_point()),selection_is_min_validation_loss=state['best_val_loss']<=value,patience=state['bad_epochs'],time=now())
   assert audit['finite_weights'] and audit['selection_is_min_validation_loss'];save(out/'audits'/f'epoch_{epoch:03d}.json',audit)
   if state['bad_epochs']>=5:state.update(complete=True,stop_reason='5_epochs_without_validation_loss_improvement')
   elif epoch>=maxepochs:state.update(complete=True,stop_reason='pilot_3_epochs' if args.pilot else 'max_epochs')
   else:state.update(phase='train',epoch=epoch+1,current_frames=0,current_groups=len(groups(split['train'],batch,epoch+1)),current_loss=None)
   checkpoint('validation_epoch_complete')
   progress('experiment_result','10Hz 시험 미세조정' if args.pilot else '10Hz 본 학습',f'{epoch} epoch의 검증 손실, 프레임 수, 가중치의 유한성을 확인했다.',f"검증 손실 {value:.6f}, 최저 {state['best_val_loss']:.6f}, 연속 미개선 {state['bad_epochs']}/5 epoch.",'고정된 분할과 선정 규칙으로 다음 단계 검증을 이어간다.')
   if args.one_pass and not state['complete']:return
  state['finished_utc']=now();checkpoint('training_complete')
 except BaseException as ex:
  save(out/'failure.json',dict(error=repr(ex),time=now(),last_consistent_checkpoint=str(out/'last.pth'),epoch=state['epoch'],phase=state['phase'],next_group=state['next_group']));raise
if __name__=='__main__':main()

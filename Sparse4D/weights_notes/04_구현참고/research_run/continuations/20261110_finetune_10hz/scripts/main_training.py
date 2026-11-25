"""첫 epoch 전체 검수 후에는 전체 검증 loss와 기본 오류 감시를 유지하는 foreground 감독."""
import os,signal,subprocess,time,fcntl
from common import *
from verify_runtime import verify
from artifact_context import CURRENT_RUN
child=None;stop=False
def interrupt(sig,frame):
 global stop
 stop=True
 if child is not None and child.poll() is None:child.send_signal(sig)
def run(script,args,log):
 global child
 if stop:raise InterruptedError('학습 감독 종료 요청: 다음 자식 작업을 시작하지 않음')
 cmd=[str(PY),str(P/'scripts'/script),*args];start=now()
 with (P/'logs'/log).open('a') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'});code=child.wait()
 child=None
 with (RUN/'main_commands.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(command=cmd,started_utc=start,finished_utc=now(),exit_code=code,log=str(P/'logs'/log)),ensure_ascii=False)+'\n')
 if stop:raise InterruptedError('학습 감독 종료 요청: 일관된 경계 체크포인트 보존')
 if code:raise RuntimeError((script,code))
def pending_epoch_review(state):
 if state is None or 'initial_validation' not in state:return None
 epoch=len(state['epochs'])
 if epoch==0 or (P/'training/audits'/f'epoch_{epoch:03d}_basic_899.json').exists():return None
 # 완료된 검증의 기록·저장 관문 전에 중단됐으면 다음 학습부터 시작하지 않는다.
 assert state['train_frames_total']==213580*epoch,'이전 epoch 검수 전에 다음 epoch 학습이 진행됨'
 assert state['checkpoint_reason'] in ['validation_epoch_complete','training_complete']
 return epoch
def monitor_checkpoint(state,epoch):
 import torch
 folder=P/'training/evaluation_checkpoints';folder.mkdir(exist_ok=True)
 source=P/'training/last.pth';fixed=folder/f'epoch_{epoch:03d}.pth'
 if not fixed.exists():
  # train.atomic_ckpt는 기존 inode에 쓰지 않고 새 파일로 replace한다.
  # hard link는 해당 순간의 파일을 유지하며 이후 last 교체를 따라가지 않는다.
  os.link(source,fixed)
  assert source.stat().st_ino==fixed.stat().st_ino
 checkpoint=torch.load(fixed,map_location='cpu',weights_only=False,mmap=True)
 saved=checkpoint['state'];expected_loss=state['initial_validation']['loss'] if epoch==0 else state['epochs'][-1]['validation']['loss']
 assert len(saved['epochs'])==epoch and saved['train_frames_total']==213580*epoch
 assert saved['checkpoint_reason'] in ['validation_epoch_complete','training_complete']
 assert saved['fingerprints']==state['fingerprints'] and saved['updates_total']==state['updates_total']
 actual_loss=saved['initial_validation']['loss'] if epoch==0 else saved['epochs'][-1]['validation']['loss']
 assert actual_loss==expected_loss
 save(folder/f'epoch_{epoch:03d}.json',dict(epoch=epoch,checkpoint=str(fixed),source=str(source),source_checkpoint_time=saved['updated_utc'],full_validation_loss=actual_loss,train_frames=saved['train_frames_total'],updates=saved['updates_total'],storage='고정된 checkpoint inode의 hard link. 이후 last.pth의 atomic replace와 별개로 보존.',includes_optimizer_and_rng=True))
 return fixed
def main():
 signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
 assert all(read(P/'gates'/f'{name}.json')['pass_'] for name in ['05_pilot_review','05b_intermediate_test_review','06_full_inputs','07_full_batch_stress'])
 policy=read(P/'protocol/supervision_policy_899.json')
 assert policy['priority']==1 and policy['patience']==5 and policy['batch']==4 and policy['hz']==10
 assert policy['full_validation_each_epoch'] and not policy['repeated_deep_reviews_after_first'] and not policy['validation8_AP_after_priority']
 progress('phase_start','전체 학습용 자료 10Hz 추가 미세조정',
  '저장된 모델·AdamW·난수·진행 상태에서 이어 진행한다. 첫 epoch의 미확인 전체 검증 및 모델 선정·저장만 추가 검수한다.',
  '학습835클립 213580시점, 검증221클립 53807시점. FP32 최대batch4, 전체 검증 loss 최저 모델 선정, 5epoch 연속 미개선 중단을 유지한다.',
  '첫 epoch 검수 후 반복 심층검수와 검증8클립 AP는 생략한다. 최종44클립·221클립 평가 및 HiVT 비교는 유지한다.')
 save(P/'records/main_supervision.json',dict(complete=False,phase='running',started_utc=now(),supervision_policy=str(P/'protocol/supervision_policy_899.json')))
 try:
  while not stop:
   state=read(P/'training/status.json') if (P/'training/status.json').exists() else None
   # 우선 지시 이전에 통과한 epoch0 AP는 재실행하지 않는다.
   # 중간 체크포인트에서 복원할 때는 train.py가 저장된 clip 경계를 사용한다.
   epoch=pending_epoch_review(state)
   if epoch is None:
    if state is not None and state['complete']:
     assert read(P/'records/first_epoch_end_to_end_review_899.json')['pass_'];break
    run('train.py',['--one-pass'],'train_main.log')
    state=read(P/'training/status.json')
    epoch=pending_epoch_review(state)
   assert epoch is not None and epoch>=1
   fixed=monitor_checkpoint(state,epoch)
   values=[state['initial_validation']['loss']]+[r['validation']['loss'] for r in state['epochs']]
   assert state['best_val_loss']==min(values) and state['best_epoch']==values.index(min(values))
   assert state['train_frames_total']==213580*epoch
   assert state['epochs'][-1]['validation']['frames']==53807
   assert read(P/'training/audits'/f'epoch_{epoch:03d}.json')['pass_']
   if not (P/'records/first_epoch_end_to_end_review_899.json').exists():
    assert epoch==1,'첫 epoch 미확인 부분을 검수하기 전에 다음 epoch가 진행됨'
    for phase in ['train','validation']:
     check=CURRENT_RUN/'epoch_reviews'/f'epoch_001_{phase}.json'
     if not check.exists():
      run(str(CURRENT_RUN/'audit_epoch_pass.py'),['--epoch','1','--phase',phase],f'first_epoch_{phase}_review_899.log')
     assert read(check)['pass_']
    import torch
    selected=(P/'training/best.pth').resolve();ck=torch.load(selected,map_location='cpu',weights_only=False,mmap=True)
    assert ck['epoch']==state['best_epoch'] and ck['validation_loss']==state['best_val_loss']
    review=dict(pass_=True,epoch=1,input_checks_reused=['06_full_inputs','07_full_batch_stress'],train_review=str(CURRENT_RUN/'epoch_reviews/epoch_001_train.json'),validation_review=str(CURRENT_RUN/'epoch_reviews/epoch_001_validation.json'),finite_weight_audit=str(P/'training/audits/epoch_001.json'),checkpoint=str(fixed),selected_checkpoint=str(selected),selected_epoch=state['best_epoch'],full_validation_loss=values[-1],previous_checks_not_repeated=True,validation8_AP_skipped=True,supervision_policy=str(P/'protocol/supervision_policy_899.json'),finished_utc=now())
    save(P/'records/first_epoch_end_to_end_review_899.json',review);save(CURRENT_RUN/'first_epoch_end_to_end_review.json',review);del ck
    progress('phase_complete','첫 epoch 전체 과정 검수','이미 통과한 전처리·입력·학습 검수를 재사용하고 전체 검증 및 선정 가중치 저장의 미확인 부분을 확인했다.',f"검증221클립 53807프레임 loss {values[-1]:.6f}, 선정 epoch {state['best_epoch']}. 첫 epoch 전체 과정 검수 통과.",'이후 반복 심층검수와 검증8클립 AP는 생략하며 전체 검증 loss·patience5·기본 오류 감시를 유지한다.')
   save(P/'training/audits'/f'epoch_{epoch:03d}_basic_899.json',dict(pass_=True,completed_training_epochs=epoch,full_validation_frames=53807,validation_loss=values[-1],monitor_AP=None,monitor_AP_skipped_by_user=True,preserved_checkpoint=str(fixed),selection_by_full_validation_loss=True,train_frames_counted=state['train_frames_total'],bad_epochs=state['bad_epochs'],next_epoch_allowed=not state['complete'],finished_utc=now()))
   progress('experiment_result','10Hz epoch 전체 검증 결과',f'{epoch}epoch의 전체 검증221클립 loss를 계산하고 최저 모델 선정 및 미개선 횟수를 기록했다.',f"전체 val loss {values[-1]:.6f}, 최저 {state['best_val_loss']:.6f}(epoch {state['best_epoch']}), 연속 미개선 {state['bad_epochs']}/5. 별도8클립 AP는 요청에 따라 생략했다.",'동일 학습 조건을 유지하며 종료 후 선정 모델의44클립·221클립 및 HiVT 평가를 수행한다.')
   if state['complete']:break
  if stop:raise InterruptedError('epoch 경계에서 감독 종료 요청을 확인함')
  assert state['complete'],'학습 종료 기준을 충족하지 않은 상태를 완료로 기록할 수 없음'
  save(P/'records/main_supervision.json',dict(complete=True,finished_utc=now(),best_epoch=state['best_epoch'],best_val_loss=state['best_val_loss'],epochs=len(state['epochs'])))
 except BaseException as ex:
  save(P/'records/main_supervision.json',dict(complete=False,error=repr(ex),stopped=stop,time=now()))
  if not stop:progress('blocked','본 학습 감독 검사','단계 실행 또는 검수에서 실패를 감지하여 다음 epoch 진행을 막았다.',f'오류 유형: {type(ex).__name__}. 마지막 일관된 체크포인트와 상세 로그를 보존했다.','원인을 해결하고 해당 관문을 재검수한 뒤 재개한다.')
  raise
if __name__=='__main__':
 (P/'training').mkdir(exist_ok=True)
 with (P/'training/supervision.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  main()

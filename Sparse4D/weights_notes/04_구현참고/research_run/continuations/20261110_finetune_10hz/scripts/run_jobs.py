"""분기별 foreground 감독. 자식 종료 신호 전달, 명령/종료 코드와 출력 보존."""
import argparse,os,signal,subprocess,time,fcntl,contextlib
from common import *
child=None;stop=False
def interrupt(sig,frame):
 global stop
 stop=True
 if child is not None and child.poll() is None:child.send_signal(sig)
def main():
 global child
 signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
 a=argparse.ArgumentParser();a.add_argument('jobs',type=Path);args=a.parse_args();rows=[]
 for job in read(args.jobs):
  if stop:break
  cmd=[str(job.get('python',PY)),str(P/'scripts'/job['script']),*job.get('args',[])];out=P/'logs'/job['log'];start=now()
  lockpath=Path(job['lock']) if job.get('lock') else None
  if lockpath:lockpath.parent.mkdir(parents=True,exist_ok=True)
  with lockpath.open('a') if lockpath else contextlib.nullcontext() as lock:
   if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
   if stop:break
   with out.open('a') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'});code=child.wait()
  child=None;rows.append(dict(command=cmd,started_utc=start,finished_utc=now(),exit_code=code,log=str(out)))
  save(RUN/(args.jobs.stem+'_execution.json'),dict(complete=len(rows)==len(read(args.jobs)) and not code and not stop,jobs=rows,stopped=stop))
  print(job['script'],job['args'],'exit',code,flush=True)
  if code:
   if not stop:progress('blocked','감독 중인 실험 단계','실행한 단계가 정상 종료하지 않아 후속 단계를 멈췄다.',f"실패 단계: {Path(job['script']).name}. 실행 기록과 단계 로그에 종료 결과를 보존했다.",'원인을 해결한 뒤 같은 설정과 저장된 상태로 해당 단계를 재검수한다.')
   raise SystemExit(code)
 if stop:raise SystemExit(130)
if __name__=='__main__':main()

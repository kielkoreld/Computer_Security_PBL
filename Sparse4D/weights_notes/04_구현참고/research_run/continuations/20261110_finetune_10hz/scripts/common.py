"""tg-563161895의 별도 경로, 원자적 기록과 한국어 진행 보고."""
import datetime,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];R=P.parents[1];ROOT=R.parent
Q=R/'continuations/20261105_resume_static_gt'
OLD10=R/'continuations/20261110_eval44_10hz_hivt'
S=R/'continuations/20261102_scheduled_finetune'
F=R/'continuations/20261102_full_dataset'
RUN=ROOT/'.agent_runs/tg-563161895';PY=R/'.venv-models/bin/python'
INITIAL=Q/'training/best_epoch_012.pth'
PILOT=P/'training_pilot_v2'
sys.path.insert(0,str(R/'scripts'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 t.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n');t.replace(p)
def progress(kind,phase,work,result,next_step,decision=None):
 row=dict(time=now(),kind=kind,phase=phase,work=work,result=result,next=next_step)
 if decision:row['decision']=decision
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'progress_reports.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def gate(name,checks,notes):
 assert all(checks.values()),(name,checks)
 save(P/'gates'/f'{name}.json',dict(pass_=True,checks=checks,notes=notes,checked_utc=now()))

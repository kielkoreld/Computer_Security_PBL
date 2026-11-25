"""Portable repository/workspace paths and atomic run records."""
import datetime,json,os
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
ROOT=(Q/'repo').resolve() if (Q/'repo').exists() else Q.parent
R=Q;S=Q
RUN=Q/'logs'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n');t.replace(p)
def source(ep):return Q/'converted'/ep
def progress(kind,phase,work,result,next_step):
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'progress_reports.jsonl').open('a') as f:f.write(json.dumps(dict(time=now(),kind=kind,phase=phase,work=work,result=result,next=next_step),ensure_ascii=False)+'\n')

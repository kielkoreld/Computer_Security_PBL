"""새 작업공간의 예측44클립을 기존 평가 함수로 계산한다."""
from pathlib import Path
import argparse,json,sys
C=Path(__file__).resolve().parents[2];R=C/'handoff'
sys.path.insert(0,str(R/'training'));sys.path.insert(0,str(R/'evaluation'))
import full_official as evaluator
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--gt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
eps=json.loads((R/'protocol/split.json').read_text())['evaluation44']
evaluator.source=lambda ep:a.gt/ep
a.output.mkdir(parents=True,exist_ok=False)
m=evaluator.pooled(eps,'sparse4d',a.predictions,a.output)
print(json.dumps(dict(clips=44,frames=m['frames'],mean_ap=m['mean_ap'],classes=m['classes']),ensure_ascii=False,indent=2))

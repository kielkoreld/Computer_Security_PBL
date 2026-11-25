"""Create a dated-path-free training/inference workspace from prepared clips."""
from pathlib import Path
import argparse,json,shutil,os
from conversion_validation import validate_conversion_report
ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'handoff'
def link(src,dst):
 assert src.exists(),str(src);dst.parent.mkdir(parents=True,exist_ok=True)
 dst.symlink_to(os.path.relpath(src.resolve(),dst.parent.resolve()),target_is_directory=src.is_dir())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['train','infer'],required=True)
 p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 split=json.loads((H/'protocol/split.json').read_text())
 eps=split['train']+split['validation']+split['evaluation44'] if a.mode=='train' else split['evaluation44']
 validations=[]
 for ep in eps:
  base=a.prepared_root/ep;c=json.loads((base/'prepared/cache'/ep/'complete.json').read_text())
  assert c['complete'] and c['bitwise_normalized_input_match']
  assert (base/'converted'/ep/'sparse4d_infos.pkl').is_file()
  validation=validate_conversion_report(json.loads((base/'converted'/ep/'conversion_report.json').read_text()),ep)
  assert validation['selected_frames']==c['frames'],'변환 보고서와 영상 캐시 프레임 수 불일치: '+ep
  validations.append(dict(episode=ep,frames=c['frames'],geometry_passed=True,motion_timing_verified=validation['motion_timing']['verified']))
  t=json.loads((base/'prepared/data'/ep/'complete.json').read_text());assert t['complete'] and t['counts']['frames']==c['frames']
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 (out/'input_validation.json').write_text(json.dumps(dict(passed=True,episodes=validations),ensure_ascii=False,indent=2)+'\n')
 for sub in ['training','protocol']:
  shutil.copytree(H/sub,out/sub,ignore=shutil.ignore_patterns('__pycache__'))
 link(ROOT,out/'repo');link(H/'core',out/'core');link(H/'artifacts',out/'artifacts')
 for ep in eps:
  base=(a.prepared_root/ep).resolve()
  link(base/'prepared/cache'/ep,out/'cache'/ep)
  link(base/'prepared/data'/ep,out/'data'/ep)
  link(base/'converted'/ep,out/'converted'/ep)
 link(H/'cache/normalization_lut.npy',out/'cache/normalization_lut.npy')
 (out/'logs').mkdir()
 if a.mode=='infer':
  import torch
  ck=ROOT/'checkpoints/best_epoch_012.pth';data=torch.load(ck,map_location='cpu',weights_only=False)
  assert data['epoch']==12 and abs(data['validation_loss']-8.487784253068527)<1e-12
  link(ck,out/'training/best.pth')
  (out/'training/status.json').write_text(json.dumps(dict(complete=True,best_epoch=12,source='이미 선정된 과거 모델; 새 학습 완료를 뜻하지 않음'),ensure_ascii=False))
 else:
  # The template retains the original model/loss/optimizer/loop and includes its own runtime file hash.
  text=(out/'training/train.py').read_text()
  assert 'Path(__file__).resolve()' in text
  (out/'training/train_portable.py').write_text(text)
 print(out)
if __name__=='__main__':main()

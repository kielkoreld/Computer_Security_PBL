"""외부 원본 한 클립을 당시 변환기→정적GT보정→공식 입력 캐시 순서로 새 경로에 생성."""
import argparse,json,sys,subprocess
from pathlib import Path
from conversion_validation import validate_conversion_report
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'handoff';Q=R
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extracted-root',type=Path,required=True);p.add_argument('--episode',required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 assert (a.extracted_root/a.episode/'anno').is_dir()
 split=json.loads((Q/'protocol/split.json').read_text());groups=[k for k in ['train','validation','evaluation44']if a.episode in split[k]];assert len(groups)==1
 # converter가 실제 각 주석·6카메라·투영·시간을 검사한다. 이 최소 audit는 archive CRC 전수검사 기록으로 주장하지 않는다.
 (out/'audit.json').write_text(json.dumps(dict(failures=[],episodes=[dict(episode_id=a.episode,split=groups[0])]),ensure_ascii=False))
 # Native 10Hz Sparse4D (frame-step=1). 2Hz densify path is retired.
 subprocess.run([sys.executable,str(R/'core/convert_dataset_extended.py'),'--episode',a.episode,'--frame-step','1',
  '--extracted-root',str(a.extracted_root.resolve()),'--audit',str(out/'audit.json'),'--split-manifest',str(Q/'protocol/split.json'),
  '--output-root',str(out/'converted'),'--records-root',str(out/'conversion_records')],check=True)
 report=json.loads((out/'converted'/a.episode/'conversion_report.json').read_text())
 try:
  validation=validate_conversion_report(report,a.episode)
 except ValueError as error:
  (out/'conversion_validation.json').write_text(json.dumps(dict(passed=False,episode=a.episode,error=str(error)),ensure_ascii=False,indent=2)+'\n')
  raise
 (out/'conversion_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
 sys.path.insert(0,str(Q/'training'));import prepare_gt
 prepare_gt.Q=out/'prepared';prepare_gt.source=lambda ep:out/'converted'/ep
 prepare_gt.prepare(a.episode)
 if not a.metadata_only:
  import cache_inputs,numpy as np
  cache_inputs.Q=out/'prepared';cache_inputs.source=lambda ep:out/'converted'/ep
  cache_inputs.cache(a.episode);np.save(out/'prepared/cache/normalization_lut.npy',cache_inputs.LUT)
 print(json.dumps(dict(complete=True,episode=a.episode,split=groups[0],output=str(out),official_image_cache=not a.metadata_only,geometry_validation_passed=True,motion_timing_verified=validation['motion_timing']['verified']),ensure_ascii=False))
if __name__=='__main__':main()

"""고정된 V0.0.4 목록의 다운로드/선택 추출. 기본값은 네트워크 없는 계획 출력."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['sample','evaluation44','train','validation','all'],default='sample')
 p.add_argument('--archives',type=Path,default=ROOT/'external_data/archives');p.add_argument('--extract-root',type=Path,default=ROOT/'external_data/extracted_work')
 p.add_argument('--download',action='store_true');p.add_argument('--extract',action='store_true');a=p.parse_args()
 split=json.loads((ROOT/'handoff/protocol/split.json').read_text())
 catalog=json.loads((ROOT/'handoff/settings/dataset_download.json').read_text())
 eps=[next(e for e in split['evaluation44']if e.startswith('DynamicObjectCrossing_'))] if a.split=='sample' else [e for k in ['train','validation','evaluation44']for e in split[k]] if a.split=='all' else split[a.split]
 files=[catalog['files'][ep+'.tar.gz']for ep in eps]
 print(json.dumps(dict(repo_id=catalog['repo_id'],revision=catalog['revision'],archives=len(files),compressed_bytes=sum(f['compressed_bytes']for f in files),download=a.download,extract=a.extract),ensure_ascii=False))
 if not(a.download or a.extract):return
 # 해시 목록은 배포 ZIP에 넣지 않고, 실행 시 고정 원격 버전의 LFS 정보를 읽는다.
 from huggingface_hub import HfApi
 remote=HfApi().dataset_info(catalog['repo_id'],revision=catalog['revision'],files_metadata=True)
 remote_hash={f.rfilename:f.lfs.sha256 for f in remote.siblings if f.lfs is not None}
 sys.path.insert(0,str(ROOT/'handoff/core'));from prepare_b2d_expansion import audit_episode,sha
 a.archives.mkdir(parents=True,exist_ok=True)
 if a.extract:a.extract_root.mkdir(parents=True,exist_ok=True)
 for ep,f in zip(eps,files):
  path=a.archives/f['path']
  if a.download and not path.exists():
   from huggingface_hub import hf_hub_download
   hf_hub_download(repo_id=catalog['repo_id'],repo_type='dataset',revision=catalog['revision'],filename=f['path'],local_dir=str(a.archives))
  assert path.is_file() and path.stat().st_size==f['compressed_bytes'],f['path']
  expected=remote_hash[f['path']]
  assert sha(path)==expected,'고정 원격 파일과 일치하지 않습니다: '+f['path']
  if a.extract:
   role=next(k for k in ['train','validation','evaluation44']if ep in split[k]);audit_episode(path,a.extract_root,expected,role)
 print('선택된 파일 확인 완료. 추출 데이터:',a.extract_root/'data/extracted')
if __name__=='__main__':main()

"""이미 받은 가중치를 원래 코드가 읽는 위치에 연결한다. 원본은 변경하지 않는다."""
from pathlib import Path
import argparse,shutil,json
ROOT=Path(__file__).resolve().parents[2];W=ROOT
DEST={'sparse_initial':'checkpoints/sparse4dv3_r50.pth',
      'sparse_epoch12':'checkpoints/best_epoch_012.pth'}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in DEST:p.add_argument('--'+key.replace('_','-'),type=Path)
 p.add_argument('--copy',action='store_true',help='기본 심볼릭 링크 대신 복사');a=p.parse_args();done=[]
 for key,rel in DEST.items():
  src=getattr(a,key)
  if src is None:continue
  src=src.expanduser().resolve(strict=True);dst=W/rel
  if dst.exists() or dst.is_symlink():
   if dst.resolve()==src:continue
   raise FileExistsError('기존 파일을 덮어쓰지 않습니다: '+str(dst))
  dst.parent.mkdir(parents=True,exist_ok=True)
  if a.copy:shutil.copy2(src,dst)
  else:dst.symlink_to(src)
  done.append(key)
 # 원래 Sparse4D config가 요구하는 anchor 초기값도 받은 가중치에서 생성한다.
 src=W/DEST['sparse_initial']
 if not src.exists():src=W/DEST['sparse_epoch12']
 if src.exists():
  import torch,numpy as np
  ck=torch.load(src,map_location='cpu',weights_only=False)
  if src.name=='best_epoch_012.pth':
   assert ck['epoch']==12 and abs(ck['validation_loss']-8.487784253068527)<1e-12
  anchor=ck['state_dict']['head.instance_bank.anchor'].detach().cpu().numpy()
  dst=W/'handoff/artifacts/nuscenes_kmeans900_from_checkpoint.npy';dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert np.array_equal(np.load(dst),anchor),'다른 anchor 파일이 이미 있습니다'
  else:np.save(dst,anchor)
 print(json.dumps(dict(configured=done,required_locations=DEST),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

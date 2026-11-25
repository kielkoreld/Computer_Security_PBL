"""기존 분할과 원본의 모든 프레임 식별자·6카메라를 확인한다."""
import collections,hashlib,json,os,pickle,platform,re,shutil,subprocess
from common import *
def main():
 for x in ['scripts','protocol','records','gates','logs','reports','data','cache','training','diagnostics','figures']:(P/x).mkdir(parents=True,exist_ok=True)
 split=read(Q/'protocol/split.json');save(P/'protocol/split.json',split)
 groups={k:{re.search(r'Town[^_]+_Route\d+',e).group(0) for e in split[k]} for k in ['train','validation','evaluation44']}
 assert not(groups['train']&groups['validation'] or groups['train']&groups['evaluation44'] or groups['validation']&groups['evaluation44'])
 rows=[]
 for k in groups:
  for e in split[k]:
   with (Q/'data'/e/'sparse4d_infos.pkl').open('rb') as f:infos=pickle.load(f)['infos']
   path=Path(next(iter(infos[0]['cams'].values()))['data_path']);raw=path.parents[2]
   assert raw.name==e
   frames=sorted(int(p.name.split('.')[0]) for p in (raw/'anno').glob('*.json.gz'))
   assert frames==list(range(len(frames))) and frames
   cameras={cam:sorted(int(p.stem) for p in d.glob('*.jpg')) for cam,d in [(c,Path(v['data_path']).parent) for c,v in infos[0]['cams'].items()]}
   assert len(cameras)==6 and all(v==frames for v in cameras.values())
   assert len(infos)==len(frames[::5]);assert read(Q/'cache'/e/'complete.json')['complete']
   rows.append(dict(episode=e,split=k,raw_path=str(raw),frames=len(frames),old_2hz_frames=len(infos),cameras=list(cameras)))
 save(P/'protocol/inventory.json',dict(complete=True,episodes=rows,finished_utc=now()))
 save(P/'protocol/raw_inventory_audit.json',dict(failures=[],episodes=[dict(episode_id=e['episode'],split=e['split']) for e in rows]))
 import torch,importlib.metadata as im
 env=dict(time=now(),os=platform.platform(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cpu_count=os.cpu_count(),cpu=subprocess.getoutput("lscpu | sed -n '/Model name/p'"),disk_free_bytes=shutil.disk_usage(P).free,gpu_status=subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv'),ram=subprocess.getoutput('free -b'),packages={n:im.version(n) for n in ['numpy','mmcv-full','mmdet','nuscenes-devkit']},checkpoint=str(INITIAL),checkpoint_sha256=hashlib.file_digest(INITIAL.open('rb'),'sha256').hexdigest())
 save(P/'records/environment.json',env);save(RUN/'environment.json',env)
 counts={k:dict(clips=len(split[k]),town_route_groups=len(groups[k]),frames=sum(e['frames'] for e in rows if e['split']==k),frames_2hz=sum(e['old_2hz_frames'] for e in rows if e['split']==k)) for k in groups}
 save(P/'records/inventory.json',dict(complete=True,counts=counts,camera_frame_sets_equal=True,contiguous_from_zero=True,group_overlap=False))
 gate('01_inventory',dict(split_unchanged=True,no_town_route_overlap=True,all_six_camera_frames_match=True,checkpoint_present=INITIAL.exists()),dict(counts=counts,scope='기존2Hz 최적 가중치부터10Hz 추가 학습. 평가44클립 점수로 학습 조건 선정 금지.'))
 text=f'\n\n## tg-563161895: 10Hz 추가 미세조정 ({now()})\n\n새 분기 `{P}`. 원본 프레임·6카메라·분할 검증 완료. 학습835/검증221/평가44 클립 유지. 입력·시간 처리와 batch4 실제 시험을 진행한다. 실행 계획 `{RUN}/PLAN_KO.md`.\n'
 with (R/'STATE.md').open('a') as f:f.write(text)
 (RUN/'STATE.md').write_text(text)
 progress('phase_complete','환경·분할·원본 입력 확인','기존 분할과 원본 프레임, 여섯 카메라 파일 목록을 전수 대조했다.',f"학습 {counts['train']['frames']}·검증 {counts['validation']['frames']}·평가 {counts['evaluation44']['frames']}개의10Hz 시점. 경로 그룹 중복 없음.",'시간 처리 진단과 무손실 캐시 및 batch4 실행을 검증한다.')
 print(json.dumps(counts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

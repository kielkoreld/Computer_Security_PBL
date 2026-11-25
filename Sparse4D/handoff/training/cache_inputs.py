"""공식 Sparse4D Dataset/Compose 출력의 무손실 RGB 캐시. GT는 포함하지 않음."""
import argparse,concurrent.futures,multiprocessing,os,pickle,shutil,sys,time
import numpy as np
from paths import Q,R,ROOT,source,save,read,now,progress
sys.path.insert(0,str(ROOT))
import torch,mmcv
from mmcv import Config
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.nuscenes_3d_det_track_dataset import NuScenes3DDetTrackDataset
CFG=Config.fromfile(str(ROOT/'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'))
MEAN=np.asarray(CFG.img_norm_cfg['mean'],np.float32);STD=np.asarray(CFG.img_norm_cfg['std'],np.float32)
# 256개 정수 화소의 정규화값을 실제 MMCV 함수로 계산. 반올림 오차 없이 동일 입력 복원.
LUT=mmcv.imnormalize(np.broadcast_to(np.arange(256,dtype=np.uint8)[:,None,None],(256,1,3)).copy(),MEAN,STD,True)[:,0,:].T.copy()

def reconstruct(rgb):return LUT[np.arange(3)[None,:,None,None],rgb]

def cache(ep):
 torch.set_num_threads(1)
 out=Q/'cache'/ep;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return read(out/'complete.json')
 start=time.perf_counter();ds=NuScenes3DDetTrackDataset(ann_file=str(source(ep)/'sparse4d_infos.pkl'),pipeline=CFG.test_pipeline,data_root='',classes=CFG.class_names,modality=CFG.input_modality,test_mode=True,data_aug_conf=CFG.data_aug_conf)
 temp=out/'rgb.tmp.npy';rgb=np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(len(ds),6,3,256,704));meta=[]
 for i in range(len(ds)):
  native=ds[i];im=native['img'].data.numpy();m=native['img_metas'].data
  raw=np.rint(im*STD[None,:,None,None]+MEAN[None,:,None,None]);assert raw.min()>=0 and raw.max()<=255
  raw=raw.astype(np.uint8);restored=reconstruct(raw)
  assert np.array_equal(restored,im),('공식 입력 캐시 복원 불일치',ep,i,float(np.max(abs(restored-im))))
  rgb[i]=raw
  meta.append(dict(token=ds.data_infos[i]['token'],projection_mat=np.asarray(native['projection_mat']).copy(),image_wh=np.asarray(native['image_wh']).copy(),timestamp=float(native['timestamp']),T_global=np.asarray(m['T_global']).copy(),T_global_inv=np.asarray(m['T_global_inv']).copy()))
 rgb.flush();del rgb;temp.replace(out/'rgb.npy')
 with (out/'meta.tmp').open('wb') as f:pickle.dump(meta,f,protocol=5)
 (out/'meta.tmp').replace(out/'meta.pkl')
 result=dict(complete=True,episode=ep,frames=len(ds),images=len(ds)*6,bitwise_normalized_input_match=True,metadata_origin='official NuScenes3DDetTrackDataset + config.test_pipeline',bytes=(out/'rgb.npy').stat().st_size,elapsed_s=time.perf_counter()-start,finished_utc=now())
 save(out/'complete.json',result);return result

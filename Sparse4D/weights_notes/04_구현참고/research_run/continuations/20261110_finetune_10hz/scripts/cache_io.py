"""기존 2Hz 무압축 캐시 + 새 네 시점의 프레임별 무손실 zlib."""
import json,zlib
import numpy as np
from common import P,Q
SHAPE=(6,3,256,704)
class RGB:
 def __init__(self,ep):
  self.old=np.load(Q/'cache'/ep/'rgb.npy',mmap_mode='r')
  self.index=json.loads((P/'cache'/ep/'rgb_index.json').read_text())
  self.fd=(P/'cache'/ep/'rgb.zlib').open('rb');self.n=len(self.index)
 def __len__(self):return self.n
 def __getitem__(self,i):
  if i%5==0:return np.asarray(self.old[i//5])
  pos,n=self.index[i];self.fd.seek(pos)
  return np.frombuffer(zlib.decompress(self.fd.read(n)),np.uint8).reshape(SHAPE)
 def close(self):self.fd.close()

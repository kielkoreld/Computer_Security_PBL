"""CARLA left-handed world -> right-handed ego / nuScenes-like reference LiDAR."""
import numpy as np
from scipy.spatial.transform import Rotation

SPARSE_CAMS = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
BEV_CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
S = np.diag([1.,-1.,1.,1.])
H = np.array([[0.,1,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,1.]])
A = np.array([[0.,1,0,0],[0,0,-1.,0],[1.,0,0,0],[0,0,0,1.]])
EDGES = [(i,i^bit) for i in range(8) for bit in [1,2,4] if i<(i^bit)]

def transform(points, T):
    p=np.asarray(points,dtype=np.float64)
    return p @ np.asarray(T)[:3,:3].T + np.asarray(T)[:3,3]

def quaternion(T):
    r=np.asarray(T)[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5:
        raise ValueError('반사 또는 잘못된 회전을 quaternion으로 변환할 수 없습니다')
    q=Rotation.from_matrix(r).as_quat()
    return q[[3,0,1,2]].tolist()

def poses(anno):
    ego=[b for b in anno['bounding_boxes'] if b['class']=='ego_vehicle']
    if len(ego)!=1: raise ValueError('ego pose가 유일하지 않습니다')
    U_Eu=np.linalg.inv(np.asarray(ego[0]['world2ego'],dtype=float))
    U_Lu=np.linalg.inv(np.asarray(anno['sensors']['LIDAR_TOP']['world2lidar'],dtype=float))
    G_E=S@U_Eu@S
    E_L=S@np.linalg.inv(U_Eu)@U_Lu@H
    G_L=G_E@E_L
    return G_E,E_L,G_L

def camera_matrices(anno, cam):
    G_E,E_L,G_L=poses(anno)
    C_U=A@np.asarray(anno['sensors'][cam]['world2cam'],dtype=float)
    C_L=C_U@S@G_L
    E_C=np.linalg.inv(G_E)@S@np.linalg.inv(C_U)
    K=np.asarray(anno['sensors'][cam]['intrinsic'],dtype=float)
    K4=np.eye(4);K4[:3,:3]=K
    return C_L,E_C,K,K4@C_L

def project(points, P):
    q=transform(points,P)
    uv=np.full((len(q),2),np.nan)
    valid=q[:,2]>1e-6
    uv[valid]=q[valid,:2]/q[valid,2,None]
    return uv,q[:,2]

def clipped_segments(corners, C, K, near=0.05):
    pts=transform(corners,C); result=[]
    for i,j in EDGES:
        p,q=pts[i].copy(),pts[j].copy()
        if max(p[2],q[2])<near: continue
        if p[2]<near: p+=(q-p)*(near-p[2])/(q[2]-p[2])
        if q[2]<near: q+=(p-q)*(near-q[2])/(p[2]-q[2])
        uv=np.stack([K@p,K@q]);uv=uv[:,:2]/uv[:,2,None]
        result.append(uv)
    return result

def in_camera(corners, C, K, width=1600,height=900):
    # Frustum overlap only; this does NOT assert absence of occlusion.
    segs=clipped_segments(corners,C,K)
    if not segs:return False
    uv=np.concatenate(segs)
    return bool(uv[:,0].max()>=0 and uv[:,0].min()<width and uv[:,1].max()>=0 and uv[:,1].min()<height)

def box_geometry(box):
    """Use recorded vertices to include stored local center and orientation, not actor origin."""
    c=np.asarray(box['world_cord'],dtype=float)
    if c.shape!=(8,3) or not np.isfinite(c).all(): raise ValueError('GT 꼭짓점 결손')
    length=np.mean(c[:4],axis=0)-np.mean(c[4:],axis=0)
    width=np.mean(c[[0,1,4,5]],axis=0)-np.mean(c[[2,3,6,7]],axis=0)
    height=np.mean(c[[0,2,4,6]],axis=0)-np.mean(c[[1,3,5,7]],axis=0)
    size=np.linalg.norm([length,width,height],axis=1)
    if (size<=1e-4).any(): raise ValueError('GT 박스 퇴화')
    return c.mean(axis=0),size,length/size[0],c

def map_class(box):
    if box['class']=='walker':return 'pedestrian'
    if box['class']!='vehicle':return None
    base=box.get('base_type')
    if base is None and box.get('state')=='static' and box.get('type_id','').startswith('/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/'):
        return 'car'
    return {'car':'car','truck':'truck','bus':'bus','bicycle':'bicycle','motorcycle':'motorcycle','van':'car'}.get(base)

def augmentation_matrix(resize=.44,crop=(0,140),flip=False,rotate=0.,size=(704,256)):
    M=np.array([[resize,0,-crop[0]],[0,resize,-crop[1]],[0,0,1.]])
    if flip:M=np.array([[-1.,0,size[0]],[0,1.,0],[0,0,1.]])@M
    t=np.deg2rad(rotate); Q=np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1.]])
    center=np.array(size)/2;Q[:2,2]=center-Q[:2,:2]@center
    out=np.eye(4);out[:3,:3]=Q@M
    return out

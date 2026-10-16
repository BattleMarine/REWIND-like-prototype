"""채택된 reflected-world 변환. 입력은raw cm, 출력camera translation은mm."""
import math


F = [[1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]
B = [[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]]


def mmul(a,b):return [[math.fsum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def transpose(a):return list(map(list,zip(*a)))
def mv(a,v):return [math.fsum(x*y for x,y in zip(row,v)) for row in a]
def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def scale(a,k):return [x*k for x in a]
def maximum_error(a,b):return max(abs(x-y) for x,y in zip(a,b))
def determinant(m):return math.fsum(m[0][i]*(m[1][(i+1)%3]*m[2][(i+2)%3]-m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))


def vector(value,dim):
    try:
        return isinstance(value,list) and len(value)==dim and all(type(x) in (int,float) and math.isfinite(x) for x in value)
    except OverflowError:
        return False


def axis(axis,angle):
    c,s=math.cos(angle),math.sin(angle)
    if axis=='x':return [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
    if axis=='y':return [[c,0.,s],[0.,1.,0.],[-s,0.,c]]
    return [[c,-s,0.],[s,c,0.],[0.,0.,1.]]


def camera_transform(camera,factor,tolerance):
    if not isinstance(camera,dict) or any(not vector(camera.get(k),3) for k in ('trans','rot','open3d_trans','open3d_rot','device_pts3d')):
        raise ValueError('camera_field_shape_finite')
    p,y,r=camera['rot']
    if maximum_error(camera['open3d_rot'],[r,-p,-y])>tolerance or maximum_error(camera['open3d_trans'],mv(F,camera['trans']))>tolerance:
        raise ValueError('camera_convention_mismatch')
    ox,oy,oz=map(math.radians,camera['open3d_rot'])
    q=mmul(mmul(axis('z',oz),axis('y',oy)),axis('x',ox))
    rotation=mmul(B,transpose(q))
    orthogonal=mmul(rotation,transpose(rotation))
    if abs(determinant(rotation)-1)>1e-10 or any(abs(orthogonal[i][j]-float(i==j))>1e-10 for i in range(3) for j in range(3)):
        raise ValueError('camera_proper_rotation_failed')
    t_raw=scale(mv(rotation,mv(F,camera['trans'])),-1.)
    return {'rotation_matrix':rotation,'translation':scale(t_raw,factor)},t_raw


def world_to_camera(world,rotation,t_raw):return add(mv(rotation,mv(F,world)),t_raw)
def camera_to_world(point,rotation,t_raw):return mv(F,mv(transpose(rotation),sub(point,t_raw)))


def project(point,calibration):
    x,y,z=point;r=math.hypot(x,y)
    if r==0:return calibration['image_center'][:]
    theta=math.atan2(-z,r)
    rho=math.fsum(c*theta**i for i,c in enumerate(calibration['polynomialW2C']))
    nx,ny=x/r*rho,y/r*rho;c,d,e=calibration['affine'];cx,cy=calibration['image_center']
    return [c*nx+d*ny+cx,e*nx+ny+cy]

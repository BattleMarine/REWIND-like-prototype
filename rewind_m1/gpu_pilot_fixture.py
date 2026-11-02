"""local_fixture에서 확장하는 합성 계산량 profile. 실제 관절/카메라 의미는 없다."""
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image

from rewind_archive.contracts import digest
from .local_fixture import create_fixture


def create_pilot_fixture(output, profile):
    output=Path(output)
    if profile not in ('cpu_smoke','gpu'): raise ValueError('pilot_profile')
    output.mkdir(exist_ok=False)
    # 기존 합성 생성기의 source/좌표 형식을 재사용한다. 전체 dataset을 내려받지 않는다.
    template_root=output/'template'; create_fixture(template_root)
    pose=json.loads((template_root/'features.json').read_bytes())
    rgb=json.loads((template_root/'manifest.json').read_bytes())
    size=[8,6] if profile=='cpu_smoke' else [1024,1024]
    names=['root','tip'] if profile=='cpu_smoke' else [f'synthetic_j{i:02d}' for i in range(62)]
    specs=[]
    for name,split,offset in (('train_a','train',0),('train_b','train',2),('validation','validation',0)):
        p,r=copy.deepcopy(pose),copy.deepcopy(rgb)
        p['contract'].update(joint_names=names,image_sizes=[size,size],preprocessing_version='pilot_synthetic_native_v1')
        p['records']=[row for row in p['records'] if row['split']==split]
        r['records']=[row for row in r['records'] if row['split']==split]
        r.update(id='pilot_'+name,image_size=size)
        for row in p['records']:
            original=copy.deepcopy(row['joints']); row['frame_index']+=offset; row['joints']={}
            for i,joint_name in enumerate(names):
                raw=copy.deepcopy(original['root' if i%2==0 else 'tip'])
                # Synthetic scalar coordinates. 실제62관절 정의/성능으로 해석하지 않는다.
                raw['device_pts3d']=[20.+i,30.+row['frame_index'],10.+i/2]
                for side in ('left','right'):
                    point=raw.get('camera_'+side+'_pts2d')
                    if point is not None:
                        raw['camera_'+side+'_pts2d']=[point[0]*(size[0]-1)/7,point[1]*(size[1]-1)/5]
                row['joints'][joint_name]=raw
        for row in r['records']:
            row['frame_index']+=offset
            for v,side in enumerate(('left','right')):
                view=row['views'][side]
                member=f"{row['sequence_id']}/fisheye_final_image/camera_{side}/final_{row['frame_index']}.png"
                target=output/'images'/member; target.parent.mkdir(parents=True,exist_ok=True)
                with Image.new('RGB',tuple(size),(40+(offset*20)+(60 if split=='validation' else 0),60+row['frame_index']*20,100+v*50)) as image:
                    image.save(target)
                raw=target.read_bytes()
                view.update(path=member,member=member,image_size=size,size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        r['content_sha256']=digest({k:v for k,v in r.items() if k!='content_sha256'})
        for file,doc in ((name+'_pose.json',p),(name+'_rgb.json',r)):
            (output/file).write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n',encoding='utf8')
        specs.append({'id':name,'pose':name+'_pose.json','rgb_manifest':name+'_rgb.json','rgb_root':'images'})
    return specs

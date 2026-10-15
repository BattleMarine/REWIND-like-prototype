"""원본 device 계약을 검증한 후 손목 상대 tensor를 내부에서만 만든다."""
import copy
from dataclasses import dataclass
import math

import torch

from rewind_data.audit import REQUIRED_FIELDS
from rewind_m1.data import DataContractError,content_hash
from rewind_m2.data import CAMERA_DEFINITION,load_document as load_m2

BODY = ['root','head','shoulder_l','shoulder_r','wrist_l','wrist_r']
UPPER = ['head','shoulder_l','shoulder_r','wrist_l','wrist_r']
FINGERS = ['index_l','tip_l','index_r','tip_r']


@dataclass
class HandSplit:
    features: torch.Tensor
    time_valid: torch.Tensor
    target_mm: torch.Tensor
    target_valid: torch.Tensor
    anchors: list
    global_target_mm: torch.Tensor
    global_target_valid: torch.Tensor
    body_target_mm: torch.Tensor
    body_target_valid: torch.Tensor
    coordinate_frame: str = 'wrist_relative_parallel_device'


def validate_roles(document, names):
    roles = document.get('m3_contract')
    if not isinstance(roles,dict) or set(roles)!={'version','body_joint_names','upper_joint_names','wrists','hands','hand_coordinate_frame','wrist_owner'}:
        raise DataContractError('명시적인 M3 body/upper/wrist/hand 계약 필요')
    if roles['version']!='m3_roles_v1' or roles['hand_coordinate_frame']!='wrist_relative_parallel_device' or roles['wrist_owner']!='body':
        raise DataContractError('M3 좌표/손목 소유 정책 불일치')
    body,upper = roles['body_joint_names'],roles['upper_joint_names']
    for group in (body,upper):
        if not isinstance(group,list) or not group or any(not isinstance(n,str) or n not in names for n in group) or len(set(group))!=len(group):
            raise DataContractError('M3 관절 목록 오류')
    if any(n not in body for n in upper):
        raise DataContractError('upper는 body subset이어야 함')
    if not isinstance(roles['wrists'],dict) or set(roles['wrists'])!={'left','right'} or not isinstance(roles['hands'],dict) or set(roles['hands'])!={'left','right'}:
        raise DataContractError('좌우 wrist/hand 명시 필요')
    fingers = []
    for side in ('left','right'):
        wrist,hand = roles['wrists'][side],roles['hands'][side]
        if wrist not in body or wrist not in upper or not isinstance(hand,list) or len(hand)<2 or hand[0]!=wrist or len(set(hand))!=len(hand):
            raise DataContractError('hand 첫 관절은 해당 body wrist 필요')
        if any(n not in names or n in body for n in hand[1:]):
            raise DataContractError('finger/body 소유 중복 또는 미정의 관절')
        fingers.extend(hand[1:])
    if roles['wrists']['left']==roles['wrists']['right'] or len(set(fingers))!=len(fingers) or set(names)!=set(body+fingers):
        raise DataContractError('좌우 중복 또는 body+finger coverage 오류')
    return copy.deepcopy(roles),fingers


def body_document(document):
    doc = copy.deepcopy(document)
    roles,_ = validate_roles(doc,doc['contract']['joint_names'])
    body = roles['body_joint_names']
    doc['contract']['joint_names'] = body
    doc['m2_contract']['skeleton']['joint_names'] = body
    doc['m2_contract']['skeleton']['edges'] = [edge for edge in doc['m2_contract']['skeleton']['edges'] if all(n in body for n in edge)]
    doc['m2_contract']['skeleton']['version'] += '_body'
    doc['manifest'] = {**doc['manifest'],'id':doc['manifest']['id']+'_body', 'm3_parent_sha256':content_hash(document)}
    del doc['m3_contract']
    for record in doc['records']:
        record['joints'] = {n:record['joints'][n] for n in body}
        # 예측 입력은 자동 생성/GT 대체하지 않고 명시된 전체 cache에서 순서대로 선택한다.
        if 'prediction' in record:
            old = record['prediction']['joint_names']
            indices = [old.index(n) for n in body]
            record['prediction']['joint_names'] = body
            for side in ('left','right'):
                record['prediction'][side] = {k:[v[i] for i in indices] for k,v in record['prediction'][side].items()}
    return doc


def prepare(document, window_size, scale_mm):
    contract,whole,cameras,manifest,all_edges = load_m2(document,window_size,scale_mm)
    roles,fingers = validate_roles(document,contract.joint_names)
    body = roles['body_joint_names']
    names = list(contract.joint_names)
    bi,fi = [names.index(n) for n in body],[names.index(n) for n in fingers]
    wrist_per_finger = [body.index(roles['wrists'][side]) for side in ('left','right') for _ in roles['hands'][side][1:]]
    hand_edges = [(fi.index(a),fi.index(b)) for a,b in all_edges if a in fi and b in fi]
    splits = {}
    for split,data in whole.items():
        bgt,bvalid = data.target_mm[:,:,bi],data.target_valid[:,:,bi]
        ggt,gvalid = data.target_mm[:,:,fi],data.target_valid[:,:,fi]
        valid = gvalid & bvalid[:,:,wrist_per_finger] & data.time_valid[...,None]
        local = torch.where(valid[...,None],ggt-bgt[:,:,wrist_per_finger],0.)
        x = data.features.reshape(*data.features.shape[:2],len(names),14)[:,:,fi].flatten(2)
        splits[split] = HandSplit(x,data.time_valid,local,valid,data.anchors,ggt,gvalid,bgt,bvalid)
    return contract,roles,fingers,splits,cameras,manifest,hand_edges,wrist_per_finger


def synthetic_document():
    names = BODY+FINGERS
    edges = [['root','head'],['head','shoulder_l'],['head','shoulder_r'],['shoulder_l','wrist_l'],['shoulder_r','wrist_r'],
             ['wrist_l','index_l'],['index_l','tip_l'],['wrist_r','index_r'],['index_r','tip_r']]
    records = []
    for split in ('train','validation'):
        for clip in range(2):
            for index in range(6):
                dx = 3.*index+clip
                coordinates = {'root':[dx,0.,80.], 'head':[dx,40.,80.+index],
                    'shoulder_l':[dx-18.,30.,80.], 'shoulder_r':[dx+18.,30.,80.],
                    'wrist_l':[dx-35.,15.+index,75.], 'wrist_r':[dx+35.,15.-index,85.]}
                for side in ('l','r'):
                    w = coordinates['wrist_'+side]
                    sign = -1 if side=='l' else 1
                    coordinates['index_'+side] = [w[0]+sign*5.,w[1]+5.+.3*index,w[2]+2.]
                    coordinates['tip_'+side] = [w[0]+sign*9.,w[1]+9.+.5*index,w[2]+3.]
                joints = {}
                for name,p in coordinates.items():
                    entry = {field:[0.]*dim for field,dim in REQUIRED_FIELDS.items()}
                    # 인위적 선형 대응: 어안 projection 또는 실제 학습 데이터가 아니다.
                    entry.update(device_pts3d=p,camera_left_pts2d=[32.+.3*p[0],12.+.3*p[1]],
                                 camera_right_pts2d=[34.+.3*p[0],12.+.3*p[1]])
                    joints[name] = entry
                angle = .01*index;c,s = math.cos(angle),math.sin(angle)
                poses = {side:{'rotation_matrix':[[c,-s,0.],[s,c,0.],[0.,0.,1.]],
                    'translation':[(-6. if side=='left' else 6.),float(index),0.]} for side in ('left','right')}
                records.append({'sequence_id':f'{split}_{clip}','motion_id':f'{split}_motion_{clip}',
                    'split':split,'frame_index':index,'joints':joints,'camera_pose':poses})
    return {'schema_version':'m1_features_v1','contract':{'joint_names':names,'distance_unit':'mm',
        'coordinate_frame':'device_relative','image_sizes':[[64,48],[64,48]],'input_source':'gt_2d','data_kind':'synthetic_fixture',
        'frontend_version':None,'preprocessing_version':'synthetic_m3_pixels_v1'},
        'manifest':{'id':'synthetic_m3_v1','dataset_version':'synthetic_m3_v1'},'records':records,
        'm2_contract':{'camera':copy.deepcopy(CAMERA_DEFINITION),'skeleton':{'version':'synthetic_named_whole_v1','joint_names':names,'edges':edges}},
        'm3_contract':{'version':'m3_roles_v1','body_joint_names':BODY,'upper_joint_names':UPPER,
            'wrists':{'left':'wrist_l','right':'wrist_r'},'hands':{'left':['wrist_l','index_l','tip_l'],'right':['wrist_r','index_r','tip_r']},
            'hand_coordinate_frame':'wrist_relative_parallel_device','wrist_owner':'body'}}

"""M1 관측/mask 계약을 유지하며 명시적인 camera/skeleton 계약을 더한다."""
import copy
import math

import torch

from rewind_m1.data import DataContractError, load_document as load_m1, synthetic_document as m1_fixture

CAMERA_DEFINITION = {
    'transform': 'world_to_camera_column_vector',
    'rotation_6d': 'matrix_columns_0_then_1',
    'translation_unit': 'mm',
    'translation_normalization': 'fixed_target_scale_mm',
    'views': ['left', 'right'],
}


def encode_camera(poses, scale_mm):
    """x_camera=R@x_world+t. UE raw Euler를 이 계약으로 자동 해석하지 않는다."""
    if not isinstance(poses, dict) or set(poses) != {'left', 'right'}:
        raise DataContractError('명시적 좌우 camera pose 필요')
    if type(scale_mm) not in (int,float) or not math.isfinite(scale_mm) or scale_mm <= 0:
        raise DataContractError('camera 고정 scale 필요')
    encoded = []
    for side in CAMERA_DEFINITION['views']:
        view = poses[side]
        if not isinstance(view,dict) or set(view) != {'rotation_matrix','translation'}:
            raise DataContractError('camera rotation_matrix/translation 계약 필요')
        rotation,translation = view['rotation_matrix'],view['translation']
        if not isinstance(rotation,list) or len(rotation)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in rotation) or not isinstance(translation,list) or len(translation)!=3:
            raise DataContractError('camera numeric list shape 필요')
        if any(type(v) not in (int,float) or not math.isfinite(v) for row in rotation for v in row) or any(type(v) not in (int,float) or not math.isfinite(v) for v in translation):
            raise DataContractError('camera bool 제외 유한한 수치 필요')
        try:
            r = torch.tensor(view['rotation_matrix'], dtype=torch.float64)
            t = torch.tensor(view['translation'], dtype=torch.float64)
        except (ValueError,TypeError,RuntimeError) as exc:
            raise DataContractError('camera 수치 배열 필요') from exc
        if r.shape != (3,3) or t.shape != (3,) or not torch.isfinite(r).all() or not torch.isfinite(t).all():
            raise DataContractError('camera shape/finite 불일치')
        if not torch.allclose(r.T@r, torch.eye(3,dtype=r.dtype), atol=1e-6,rtol=0) or abs(float(torch.linalg.det(r))-1) > 1e-6:
            raise DataContractError('camera proper rotation 필요')
        encoded.append(torch.cat((r[:,0], r[:,1], t/scale_mm)).float())
    result = torch.stack(encoded)
    if not torch.isfinite(result).all():
        raise DataContractError('camera float32 overflow')
    return result


def validate_extension(document, joint_names):
    extension = document.get('m2_contract')
    if not isinstance(extension,dict) or extension.get('camera') != CAMERA_DEFINITION:
        raise DataContractError('m2_contract camera 정의 불일치')
    skeleton = extension.get('skeleton')
    if not isinstance(skeleton,dict) or skeleton.get('joint_names') != list(joint_names) or not isinstance(skeleton.get('edges'),list):
        raise DataContractError('명시적 skeleton 이름/edge 필요')
    if not isinstance(skeleton.get('version'),str) or not skeleton['version'].strip():
        raise DataContractError('skeleton version 필요')
    edges, seen = [], set()
    for edge in skeleton['edges']:
        if not isinstance(edge,list) or len(edge)!=2 or any(name not in joint_names for name in edge) or edge[0]==edge[1]:
            raise DataContractError('skeleton edge 오류')
        pair = tuple(sorted(joint_names.index(name) for name in edge))
        if pair in seen:
            raise DataContractError('skeleton 중복 edge')
        seen.add(pair);edges.append(pair)
    if document['contract']['data_kind'] == 'dataset_sample':
        evidence = extension.get('verification_evidence',{})
        if not isinstance(evidence,dict) or any(not isinstance(evidence.get(k),str) or not evidence[k].strip() for k in ('camera_transform','skeleton')):
            raise DataContractError('실데이터 camera/skeleton 추가 검증 근거 필요')
    return edges


def load_document(document, window_size, scale_mm):
    # 기존 M1 guard는 sequence/motion split, 관측 source, 3D mask/단위를 모두 검사한다.
    try:
        contract, splits, manifest_hash = load_m1(document, window_size)
    except (ValueError,TypeError,KeyError) as exc:
        raise DataContractError(str(exc)) from exc
    edges = validate_extension(document, contract.joint_names)
    records = {(r['sequence_id'],r['frame_index']): r for r in document['records']}
    cameras = {key: encode_camera(r.get('camera_pose'),scale_mm) for key,r in records.items()}
    packed = {}
    for split,data in splits.items():
        tensor = torch.zeros(len(data.anchors),window_size,2,9)
        for row,(sequence,index) in enumerate(data.anchors):
            for slot in range(window_size):
                if data.time_valid[row,slot]:
                    # M1 window는 gap에서 reset하며 실제slot은 연속 suffix이다.
                    tensor[row,slot] = cameras[(sequence,index-window_size+1+slot)]
        packed[split] = tensor
    return contract,splits,packed,manifest_hash,edges


def synthetic_document():
    doc = copy.deepcopy(m1_fixture())
    doc['manifest'] = {'id':'synthetic_m2_v1','dataset_version':'synthetic_m2_v1'}
    doc['m2_contract'] = {'camera':copy.deepcopy(CAMERA_DEFINITION),
        'skeleton':{'version':'synthetic_two_joint_v1','joint_names':['root','tip'],'edges':[['root','tip']]}}
    for record in doc['records']:
        angle = .02*record['frame_index']
        c,s = math.cos(angle), math.sin(angle)
        rotation = [[c,-s,0.],[s,c,0.],[0.,0.,1.]]
        record['camera_pose'] = {side:{'rotation_matrix':rotation,
            'translation':[(-6. if side=='left' else 6.),float(record['frame_index']),0.]} for side in ('left','right')}
    return doc

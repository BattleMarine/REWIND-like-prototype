"""이미지/향후 video adapter가 공유할 선언 metadata. Decoder·temporal 승인은 포함하지 않는다."""
import copy
import math
import re

from rewind_archive.contracts import digest,safe_path,validate_manifest
from rewind_archive.matching import identify

SCHEMA='stereo_frame_input_v1'
PREPROCESS={'version':'rgb_0_1_v1','pixel_format':'RGB','value_range':[0.,1.],
            'coordinate_space':'native_pixel','spatial_transform':None}


def _fields(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ValueError('frame_input_fields')
def _text(value):return isinstance(value,str) and 0<len(value)<=512 and not any(ord(c)<32 for c in value)
def _sha(value):return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None
def _number(value):
    try:return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:return False
def _integer(value):return type(value) is int and 0<=value<10**12


def unknown_timing(reason):
    return {'status':'unknown','timestamp_seconds':None,'pts':None,'time_base':None,'clock_id':None,'reason':reason}


def validate_timing(value):
    _fields(value,('status','timestamp_seconds','pts','time_base','clock_id','reason'))
    timestamp,pts,base=value['timestamp_seconds'],value['pts'],value['time_base']
    if value['status']=='unknown':
        if any(value[k] is not None for k in ('timestamp_seconds','pts','time_base','clock_id')) or not _text(value['reason']):raise ValueError('frame_input_unknown_time')
        return None
    if value['status']!='provided' or value['reason'] is not None or not _text(value['clock_id']):raise ValueError('frame_input_time_status')
    if timestamp is not None and not _number(timestamp):raise ValueError('frame_input_timestamp')
    if (pts is None)!=(base is None):raise ValueError('frame_input_pts_timebase_pair')
    if pts is not None:
        _fields(base,('numerator','denominator'))
        if type(pts) is not int or any(type(n) is not int or not 1<=n<=10**12 for n in base.values()) or abs(pts)>=2**63:raise ValueError('frame_input_pts_timebase')
        seconds=pts*base['numerator']/base['denominator']
        if not math.isfinite(seconds) or timestamp is not None and not math.isclose(timestamp,seconds,rel_tol=0.,abs_tol=1e-9):raise ValueError('frame_input_time_disagreement')
        return seconds  # 검증 계산에만 사용한다. record에 새 시간을 만들지 않는다.
    if timestamp is None:raise ValueError('frame_input_time_missing')
    return timestamp


def _geometry(value):
    _fields(value,('calibration','camera_pose'))
    calibration=value['calibration'];_fields(calibration,('status','sha256','projection_model','reason'))
    if calibration['status']=='unknown':
        if calibration['sha256'] is not None or calibration['projection_model'] is not None or not _text(calibration['reason']):raise ValueError('frame_input_unknown_calibration')
    elif calibration['status']!='provided' or not _sha(calibration['sha256']) or not _text(calibration['projection_model']) or calibration['reason'] is not None:
        raise ValueError('frame_input_calibration')
    _fields(value['camera_pose'],('left','right'))
    for pose in value['camera_pose'].values():
        _fields(pose,('status','matrix','direction','translation_unit','coordinate_frame','timing','reason'))
        validate_timing(pose['timing'])
        if pose['status']=='unknown':
            if any(pose[k] is not None for k in ('matrix','direction','translation_unit','coordinate_frame')) or not _text(pose['reason']):raise ValueError('frame_input_unknown_pose')
        else:
            matrix=pose['matrix']
            if pose['status']!='provided' or pose['reason'] is not None or pose['direction'] not in ('world_to_camera','camera_to_world') or pose['translation_unit'] not in ('mm','cm','m') or not _text(pose['coordinate_frame']):raise ValueError('frame_input_pose_contract')
            if not isinstance(matrix,list) or len(matrix)!=4 or any(not isinstance(row,list) or len(row)!=4 or not all(_number(x) for x in row) for row in matrix) or matrix[3]!=[0,0,0,1]:raise ValueError('frame_input_pose_matrix')
            # Reflected-world convention을 허용한다. det=+1/SO(3)을 임의로 강제하지 않는다.


def validate_frame_input(record):
    _fields(record,('schema_version','mode','sequence_id','frame_index','dataset_version','data_kind','views','stereo_sync',
                    'geometry','preprocessing','validity','temporal_ready','train_ready','content_sha256'))
    if record['schema_version']!=SCHEMA or record['mode']!='independent_frame' or record['temporal_ready'] is not False or record['train_ready'] is not False:raise ValueError('frame_input_independent_only')
    if safe_path(record['sequence_id'])!=record['sequence_id'] or not _integer(record['frame_index']) or not _text(record['dataset_version']) or record['data_kind'] not in ('dataset_sample','synthetic_fixture'):raise ValueError('frame_input_identity')
    _fields(record['views'],('left','right'));times={}
    for side,view in record['views'].items():
        _fields(view,('side','image_size','source_identity','timing','declared_valid'))
        if view['side']!=side or view['declared_valid'] is not True or not isinstance(view['image_size'],list) or len(view['image_size'])!=2 or any(type(n) is not int or n<4 for n in view['image_size']) or math.prod(view['image_size'])>16*1024**2:raise ValueError('frame_input_view')
        identity=view['source_identity']
        if not isinstance(identity,dict):raise ValueError('frame_input_source_identity')
        if identity.get('kind')=='archive_image':
            _fields(identity,('kind','source_id','archive_manifest_sha256','extraction_manifest_sha256','member','payload_sha256','payload_size_bytes'))
            if not _text(identity['source_id']) or any(not _sha(identity[k]) for k in ('archive_manifest_sha256','extraction_manifest_sha256','payload_sha256')) or type(identity['payload_size_bytes']) is not int or not 1<=identity['payload_size_bytes']<=64*1024**2:raise ValueError('frame_input_image_identity')
            if safe_path(identity['member'])!=identity['member'] or identify(identity['member'],'rgb')!=(record['sequence_id'],record['frame_index'],side):raise ValueError('frame_input_member_key')
        elif identity.get('kind')=='video_stream':
            _fields(identity,('kind','asset_sha256','stream_index','source_frame_index','decoder_version'))
            if not _sha(identity['asset_sha256']) or type(identity['stream_index']) is not int or not 0<=identity['stream_index']<256 or not _integer(identity['source_frame_index']) or not _text(identity['decoder_version']):raise ValueError('frame_input_video_identity')
        else:raise ValueError('frame_input_source_kind')
        times[side]=validate_timing(view['timing'])
    if record['views']['left']['source_identity']==record['views']['right']['source_identity']:raise ValueError('frame_input_duplicate_view_source')
    sync=record['stereo_sync'];_fields(sync,('status','evidence','max_skew_seconds'))
    if not _text(sync['evidence']):raise ValueError('frame_input_sync_evidence')
    if sync['status']=='unverified':
        if sync['max_skew_seconds'] is not None:raise ValueError('frame_input_unverified_sync')
    elif sync['status']=='provided':
        clocks=[v['timing']['clock_id'] for v in record['views'].values()]
        if any(t is None for t in times.values()) or clocks[0]!=clocks[1] or not _number(sync['max_skew_seconds']) or sync['max_skew_seconds']<0 or abs(times['left']-times['right'])>sync['max_skew_seconds']:raise ValueError('frame_input_sync_mismatch')
    else:raise ValueError('frame_input_sync_status')
    _geometry(record['geometry'])
    if record['preprocessing']!=PREPROCESS:raise ValueError('frame_input_preprocessing_not_supported')
    if record['validity']!={'view_presence':'declared','annotation_mask':None,'visibility':None}:raise ValueError('frame_input_validity_declaration')
    if not _sha(record['content_sha256']) or digest({k:v for k,v in record.items() if k!='content_sha256'})!=record['content_sha256']:raise ValueError('frame_input_hash')
    return copy.deepcopy(record)


def from_image_metadata(metadata):
    """기존 image metadata를 공통 선언에 복사한다. 시간/geometry를 채워 넣지 않는다."""
    row=metadata['record'];views={}
    for side in ('left','right'):
        view=row['views'][side];source=metadata['sources'][view['source_id']]
        archive=validate_manifest(source['archive_manifest'])
        if archive['kind']!='rgb' or archive['dataset_version']!=metadata['dataset_version']:raise ValueError('frame_input_archive_identity')
        views[side]={'side':side,'image_size':list(view['image_size']),'declared_valid':True,
            'source_identity':{'kind':'archive_image','source_id':view['source_id'],'archive_manifest_sha256':digest(archive),
                'extraction_manifest_sha256':source['extraction_manifest_sha256'],'member':view['member'],
                'payload_sha256':view['sha256'],'payload_size_bytes':view['size_bytes']},
            'timing':unknown_timing('image_manifest_has_no_verified_timing')}
    unknown_pose={'status':'unknown','matrix':None,'direction':None,'translation_unit':None,'coordinate_frame':None,
                  'timing':unknown_timing('camera_pose_not_in_rgb_manifest'),'reason':'camera_pose_not_in_rgb_manifest'}
    body={'schema_version':SCHEMA,'mode':'independent_frame','sequence_id':row['sequence_id'],'frame_index':row['frame_index'],
        'dataset_version':metadata['dataset_version'],'data_kind':metadata['data_kind'],'views':views,
        'stereo_sync':{'status':'unverified','evidence':'matching_sequence_frame_does_not_verify_timing','max_skew_seconds':None},
        'geometry':{'calibration':{'status':'unknown','sha256':None,'projection_model':None,'reason':'calibration_not_in_rgb_manifest'},
                    'camera_pose':{'left':copy.deepcopy(unknown_pose),'right':copy.deepcopy(unknown_pose)}},
        'preprocessing':copy.deepcopy(PREPROCESS),'validity':{'view_presence':'declared','annotation_mask':None,'visibility':None},
        'temporal_ready':False,'train_ready':False}
    return validate_frame_input({**body,'content_sha256':digest(body)})

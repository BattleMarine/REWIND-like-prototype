"""Hash와 공식 split을 대조해 학습이 차단된 실제 pose candidate를 만든다."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import time

from rewind_archive.contracts import digest,safe_path
from rewind_data.audit import read_json,REQUIRED_FIELDS
from .coordinates import add,camera_transform,camera_to_world,maximum_error,mv,F,project,scale,sub,vector,world_to_camera
from .spec import validate_spec
from .provenance import revision_identity

ROOT=Path(__file__).resolve().parents[1]
FRAME=re.compile(r'frame_([0-9]+)\.json\Z')


def sha(data):return hashlib.sha256(data).hexdigest()


class Budget:
    def __init__(self,limits):self.limits=limits;self.total=0;self.started=time.monotonic()
    def check(self):
        if time.monotonic()-self.started>self.limits['max_elapsed_seconds']:raise ValueError('prepare_elapsed_limit')
    def read(self,path):
        self.check()
        if path.is_symlink() or not path.is_file():raise ValueError('source_not_regular_file')
        size=path.stat().st_size
        if size>self.limits['max_file_bytes'] or self.total+size>self.limits['max_total_input_bytes']:raise ValueError('prepare_input_byte_limit')
        with path.open('rb') as f:raw=f.read(self.limits['max_file_bytes']+1)
        self.total+=len(raw)
        if len(raw)!=size or len(raw)>self.limits['max_file_bytes'] or self.total>self.limits['max_total_input_bytes']:raise ValueError('prepare_source_size_changed')
        self.check();return raw


def official_splits(directory,spec,budget):
    owners={};sources={}
    for split in ('train','validation','test'):
        raw=budget.read(directory/(split+'.txt'))
        if sha(raw)!=spec['pinned_sources']['split_sha256'][split]:raise ValueError('official_split_hash_mismatch')
        rows=raw.decode('utf-8-sig').splitlines()
        if not rows:raise ValueError('official_split_empty')
        for seq in rows:
            if seq!=safe_path(seq) or len(seq.split('/'))!=6 or seq in owners:raise ValueError('official_split_duplicate_or_path')
            owners[seq]=split
        sources[split]={'sha256':sha(raw),'rows':len(rows)}
    return owners,sources


def motion_entries(doc,sequences,owners):
    if not isinstance(doc,dict) or set(doc)!={'schema_version','sequences'} or doc['schema_version']!='explicit_motion_map_v1' or not isinstance(doc['sequences'],dict) or set(doc['sequences'])!=set(sequences):
        raise ValueError('explicit_motion_map_sequence_coverage')
    motion_owners={}
    for seq,entry in doc['sequences'].items():
        if not isinstance(entry,dict) or set(entry)!={'motion_id','identity_status','evidence'}:raise ValueError('motion_entry_fields')
        if not isinstance(entry['motion_id'],str) or not re.fullmatch(r'[A-Za-z0-9_:/.-]{1,256}',entry['motion_id']):raise ValueError('explicit_motion_id_required')
        if entry['identity_status'] not in ('declared_unverified','original_verified') or not isinstance(entry['evidence'],str) or not entry['evidence'].strip() or len(entry['evidence'])>2048:raise ValueError('motion_identity_evidence')
        if motion_owners.get(entry['motion_id'],owners[seq])!=owners[seq]:raise ValueError('motion_split_overlap')
        motion_owners[entry['motion_id']]=owners[seq]
    return doc['sequences']


def candidate_contract(spec):
    return {'joint_names':spec['joint_names'],'distance_unit':'mm','coordinate_frame':'device_relative',
        'data_kind':'dataset_sample','input_source':'gt_2d','image_sizes':spec['coordinates']['image_sizes'],
        'preprocessing_version':spec['preprocessing_version'],'frontend_version':None,'frontend_checkpoint_sha256':None,
        'world_basis':'F_diag_1_minus1_1','camera_transform':'world_to_camera_column_vector',
        'camera_translation_unit':'mm','rotation_6d':'matrix_columns_0_then_1',
        'visibility_semantics':'unknown','confidence_semantics':'unknown','training_release':'not_implemented'}


def convert(input_root,extraction_manifest,split_dir,motion_map,spec_path,calibration):
    paths=[Path(p).resolve() for p in (input_root,extraction_manifest,split_dir,motion_map,spec_path,calibration)]
    input_root,extraction_manifest,split_dir,motion_map,spec_path,calibration=paths
    # Spec 자체도 무제한 read하지 않는다.
    if spec_path.stat().st_size>262144:raise ValueError('spec_file_limit')
    spec_raw=spec_path.read_bytes();spec=validate_spec(read_json(spec_raw));budget=Budget(spec['limits'])
    em_raw=budget.read(extraction_manifest);em=read_json(em_raw)
    if not isinstance(em,dict) or em.get('schema_version')!='archive_extraction_v1' or em.get('inspection_candidates_usable') is not True or em.get('failure_class') not in ('none','bounded_stop'):
        raise ValueError('extraction_not_usable')
    if em.get('content_sha256')!=digest({k:v for k,v in em.items() if k!='content_sha256'}):raise ValueError('extraction_manifest_content_hash')
    if em.get('archive_manifest',{}).get('kind')!='pose' or not isinstance(em.get('files'),list):raise ValueError('extraction_pose_manifest')
    expected={}
    for row in em['files']:
        if not isinstance(row,dict) or row.get('kind')!='pose' or row.get('tar_complete') is not True:raise ValueError('extraction_incomplete_payload')
        member=safe_path(row.get('member'))
        if member in expected:raise ValueError('extraction_duplicate_member')
        expected[member]=row
    owners,split_sources=official_splits(split_dir,spec,budget)
    calib_raw=budget.read(calibration)
    if sha(calib_raw)!=spec['pinned_sources']['scaramuzza_sha256']:raise ValueError('calibration_hash_mismatch')
    calib=read_json(calib_raw)
    if calib.get('size')!=spec['coordinates']['image_sizes'][0]:raise ValueError('calibration_size')
    map_raw=budget.read(motion_map);mapping=read_json(map_raw)
    files=[]
    for path in input_root.rglob('*.json'):
        budget.check()
        if path.is_symlink() or not FRAME.fullmatch(path.name):raise ValueError('pose_filename_or_symlink')
        files.append(path)
        if len(files)>spec['limits']['max_files']:raise ValueError('prepare_file_count_limit')
    if not files:raise ValueError('pose_input_empty')
    parsed=[];sequences=set();keys=set()
    for path in files:
        relative=path.relative_to(input_root).as_posix();member=safe_path(relative)
        if path.parent.name!='json':raise ValueError('pose_sequence_path')
        seq=path.parent.parent.relative_to(input_root).as_posix()
        if seq not in owners:raise ValueError('sequence_not_in_official_split')
        index=int(FRAME.fullmatch(path.name)[1])
        if (seq,index) in keys:raise ValueError('duplicate_sequence_frame')
        keys.add((seq,index));sequences.add(seq)
        if member not in expected:raise ValueError('payload_not_in_extraction_manifest')
        raw=budget.read(path);h=sha(raw);entry=expected[member]
        if h!=entry.get('sha256') or len(raw)!=entry.get('size_bytes'):raise ValueError('payload_hash_or_size_mismatch')
        doc=read_json(raw)
        if not isinstance(doc,dict) or type(doc.get('frame')) not in (int,str) or str(doc['frame'])!=str(index) or not isinstance(doc.get('joints'),dict):raise ValueError('pose_frame_or_joints')
        parsed.append((seq,index,relative,h,len(raw),doc))
    motions=motion_entries(mapping,sequences,owners)
    names=spec['joint_names'];factor=spec['coordinates']['mm_per_raw_unit'];tol=spec['coordinates']['raw_consistency_tolerance']
    records=[];stats=collections.Counter();per_sequence={};residuals=collections.defaultdict(list);source_files=[]
    for seq,index,relative,h,size,doc in sorted(parsed):
        budget.check()
        previous=per_sequence.get(seq,{}).get('last_frame')
        start=previous is None or index!=previous+1
        info=per_sequence.setdefault(seq,{'split':owners[seq],'frames':[],'contiguous_runs':[],'gap_count':0,'last_frame':None})
        info['frames'].append(index);info['last_frame']=index
        if start:
            if previous is not None:info['gap_count']+=1
            info['contiguous_runs'].append([index,index])
        else:info['contiguous_runs'][-1][1]=index
        timestamp=doc.get('timestamp')
        if timestamp is not None and not (isinstance(timestamp,str) and len(timestamp)<=256 or type(timestamp) in (int,float) and vector([timestamp],1)):
            raise ValueError('timestamp_raw_type_finite')
        camera={};raw_t={}
        for side in ('left','right'):
            camera[side],raw_t[side]=camera_transform(doc.get('camera_'+side),factor,tol)
        left,right=doc['camera_left'],doc['camera_right']
        baseline=sum((a-b)**2 for a,b in zip(left['trans'],right['trans']))**.5
        lo,hi=spec['coordinates']['baseline_raw_range']
        if not lo<=baseline<=hi:raise ValueError('metric_anchor_baseline_mismatch')
        if maximum_error([x for row in camera['left']['rotation_matrix'] for x in row],[x for row in camera['right']['rotation_matrix'] for x in row])>tol:raise ValueError('stereo_rotation_mismatch')
        midpoint=[(a+b)/2 for a,b in zip(left['trans'],right['trans'])]
        device=doc.get('device',{})
        if not vector(device.get('open3d_trans'),3) or not vector(device.get('open3d_rot'),3) or maximum_error(device['open3d_trans'],mv(F,midpoint))>tol:
            raise ValueError('device_midpoint_mismatch')
        if maximum_error(device['open3d_rot'],[math.radians(v) for v in left['open3d_rot']])>tol:raise ValueError('device_rotation_units_mismatch')
        for side in ('left','right'):
            meta=doc['camera_'+side];offset=mv(camera[side]['rotation_matrix'],mv(F,sub(meta['trans'],midpoint)))
            if maximum_error(offset,meta['device_pts3d'])>tol:raise ValueError('camera_device_offset_mismatch')
        fields={f:{'coordinates':[],'valid_mask':[]} for f in REQUIRED_FIELDS}
        features=[];valid_map={};selected_raw={}
        for name in names:
            joint=doc['joints'].get(name,{})
            if not isinstance(joint,dict):joint={}
            selected_raw[name]=joint
            valid_map[name]={}
            for field,dim in REQUIRED_FIELDS.items():
                value=joint.get(field);ok=vector(value,dim)
                valid_map[name][field]=ok;stats[field+('_valid' if ok else '_missing_or_invalid')]+=1
                converted=scale(value,factor) if ok and dim==3 else value[:] if ok else None
                if converted is not None and not vector(converted,dim):raise ValueError('millimeter_conversion_overflow')
                fields[field]['coordinates'].append(converted);fields[field]['valid_mask'].append(ok)
            for v,side in enumerate(('left','right')):
                p=joint.get('camera_'+side+'_pts2d');ok=valid_map[name]['camera_'+side+'_pts2d'];width,height=spec['coordinates']['image_sizes'][v]
                features.extend([2*p[0]/(width-1)-1,2*p[1]/(height-1)-1] if ok else [0.,0.])
                features.extend([float(ok),0.,0.,0.,0.])
                if ok and any(x<0 for x in p):stats['negative_2d_joint_views']+=1
        # 선택 밖25관절도 world/camera 수치 검사에 사용하고, 원본에서 삭제하지 않는다.
        for joint in doc['joints'].values():
            if not isinstance(joint,dict):continue
            for side in ('left','right'):
                c3=joint.get('camera_'+side+'_pts3d');c2=joint.get('camera_'+side+'_pts2d');d3=joint.get('device_pts3d')
                if vector(c3,3) and vector(joint.get('trans'),3):
                    prediction=world_to_camera(joint['trans'],camera[side]['rotation_matrix'],raw_t[side])
                    error=maximum_error(prediction,c3);residuals['world_camera_raw_max_abs'].append(error)
                    inv=camera_to_world(c3,camera[side]['rotation_matrix'],raw_t[side])
                    residuals['inverse_raw_max_abs'].append(maximum_error(inv,joint['trans']))
                    if error>tol:raise ValueError('world_camera_coordinate_mismatch')
                if vector(c3,3) and vector(d3,3):
                    error=maximum_error(add(c3,doc['camera_'+side]['device_pts3d']),d3)
                    residuals['camera_device_raw_max_abs'].append(error)
                    if error>tol:raise ValueError('camera_device_coordinate_mismatch')
                if vector(c3,3) and vector(c2,2):
                    error=sum((a-b)**2 for a,b in zip(project(c3,calib),c2))**.5
                    residuals['projection_px'].append(error)
                    if error>spec['coordinates']['projection_tolerance_px']:raise ValueError('annotation_projection_mismatch')
        local=[];local_valid=[]
        for side in ('left','right'):
            wrist=spec['wrists'][side]
            for name in spec['hands'][side][1:]:
                ok=valid_map[wrist]['device_pts3d'] and valid_map[name]['device_pts3d']
                local_valid.append(ok)
                local.append(scale(sub(selected_raw[name]['device_pts3d'],selected_raw[wrist]['device_pts3d']),factor) if ok else None)
        target_valid=fields['device_pts3d']['valid_mask']
        if not any(target_valid):stats['anchors_without_valid_target']+=1
        source={'member':relative,'sha256':h,'size_bytes':size,'extraction_manifest_sha256':sha(em_raw),'raw_joint_names':list(doc['joints'])}
        source_files.append(source)
        records.append({'sequence_id':seq,'frame_index':index,'split':owners[seq],'motion_id':motions[seq]['motion_id'],
            'motion_identity':motions[seq],'source':source,'timestamp_raw':timestamp,
            'time':{'history_reset':start,'gap_before':None if previous is None else index-previous-1,'fps_nominal':25,'timestamp_semantics':'unverified','time_seconds':None},
            'fields':fields,'observation_features':features,'camera_pose':camera,
            'visibility':{'left':[None]*62,'right':[None]*62},'confidence':{'left':[None]*62,'right':[None]*62},
            'hand_target':{'coordinate_frame':'wrist_relative_parallel_device','distance_unit':'mm','joint_names':[n for s in ('left','right') for n in spec['hands'][s][1:]],'coordinates':local,'valid_mask':local_valid,'wrist_owner':'body'}})
    for info in per_sequence.values():del info['last_frame']
    # 원본을 쓰지 않고 사후 hash를 다시 읽어 병행 변경도 감지한다.
    for seq,index,relative,h,size,doc in parsed:
        if sha(budget.read(input_root/relative))!=h:raise ValueError('raw_source_changed_during_preparation')
    blockers=['candidate_schema_not_training_release','rgb_pose_alignment_not_checked','independent_validation_not_verified']
    if not {'train','validation'}<=set(r['split'] for r in records):blockers.append('required_train_validation_splits_missing')
    if any(m['identity_status']!='original_verified' for m in motions.values()):blockers.append('original_motion_identity_unverified')
    if any(info['gap_count'] for info in per_sequence.values()):blockers.append('gapped_sequences_require_contiguous_selection')
    if stats['anchors_without_valid_target']:blockers.append('anchors_without_valid_target')
    source_hashes={'spec':sha(spec_raw),'extraction_manifest':sha(em_raw),'calibration':sha(calib_raw),'motion_map':sha(map_raw),'official_split_files':split_sources}
    source_code={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for package in ('rewind_prepare','rewind_data','rewind_archive') for p in sorted((ROOT/package).glob('*.py'))}
    source_revision=revision_identity(ROOT)
    candidate={'schema_version':'real_pose_candidate_v1','contract':candidate_contract(spec),'preparation_spec':spec,
        'manifest':{'id':spec['id'],'data_kind':'dataset_sample','source_hashes':source_hashes,'source_code':source_code,**source_revision,
                    'raw_files':source_files,'archive_id':em['archive_manifest'].get('archive_id'),'dataset_version':em['archive_manifest'].get('dataset_version'),
                    'whole_archive_integrity_verified':em.get('whole_archive_integrity_verified') is True},
        'records':records,'training_eligibility':{'train_ready':False,'blockers':blockers,'reason':'candidate_only_existing_training_guards_unchanged'}}
    candidate['content_sha256']=digest(candidate)
    report={'schema_version':'real_pose_preparation_report_v1','status':'candidate_blocked','train_ready':False,'blockers':blockers,
        'candidate_content_sha256':candidate['content_sha256'],'records':len(records),'selected_joint_count':62,'feature_width':868,'hand_target_count':40,
        'sequence_summary':per_sequence,'field_counts':dict(stats),'numeric_validation':{k:{'count':len(v),'max':max(v),'mean':sum(v)/len(v)} for k,v in residuals.items()},
        'source_hashes':source_hashes,'unit_status':spec['coordinates']['unit_evidence_status'],'graph_status':spec['status'],
        'rgb_alignment_checked':False,'independent_validation_checked':False,'training_run':False,
        'input_bytes_read':budget.total,'elapsed_seconds':time.monotonic()-budget.started,'limits':spec['limits']}
    validate_candidate(candidate)
    budget.check()
    return candidate,report


def validate_candidate(doc):
    if not isinstance(doc,dict) or doc.get('schema_version')!='real_pose_candidate_v1' or doc.get('content_sha256')!=digest({k:v for k,v in doc.items() if k!='content_sha256'}):raise ValueError('candidate_schema_hash')
    spec=validate_spec(doc['preparation_spec'])
    if doc.get('contract')!=candidate_contract(spec) or doc.get('training_eligibility',{}).get('train_ready') is not False:raise ValueError('candidate_contract_or_ready_flag')
    for record in doc['records']:
        if len(record['observation_features'])!=868 or not vector(record['observation_features'],868):raise ValueError('candidate_feature_shape')
        for field,dim in REQUIRED_FIELDS.items():
            f=record['fields'][field]
            if len(f['coordinates'])!=62 or len(f['valid_mask'])!=62 or any(type(m) is not bool or (not vector(p,dim) if m else p is not None) for p,m in zip(f['coordinates'],f['valid_mask'])):raise ValueError('candidate_field_mask_shape')
        hand=record['hand_target']
        if len(hand['coordinates'])!=40 or len(hand['valid_mask'])!=40 or hand['distance_unit']!='mm' or hand['coordinate_frame']!='wrist_relative_parallel_device':raise ValueError('candidate_hand_contract')
        if any(type(m) is not bool or (not vector(p,3) if m else p is not None) for p,m in zip(hand['coordinates'],hand['valid_mask'])):raise ValueError('candidate_hand_mask_shape')
    return doc

"""원본별 candidate 하나만 캐시하는 GT2D/GT3D/native RGB 진단 reader."""
import copy
from dataclasses import dataclass

from rewind_archive.contracts import digest
from rewind_archive.matching import identify
from rewind_data.audit import REQUIRED_FIELDS
from rewind_m1.local_rgb import LocalStereoRGB, RGBLimits
from rewind_prepare.converter import Budget, motion_entries, official_splits, validate_candidate, vector
from rewind_prepare.spec import validate_spec
from .integrity import Integrity, MIB, require, sha


@dataclass(frozen=True)
class ReaderLimits:
    max_candidate_bytes:int = 4*MIB
    max_records:int = 32
    max_batch_size:int = 2
    max_batch_tensor_bytes:int = 64*MIB

    def __post_init__(self):
        for name,ceiling in [('max_candidate_bytes',4*MIB),('max_records',32),('max_batch_size',2),('max_batch_tensor_bytes',64*MIB)]:
            require(type(getattr(self,name)) is int and 1 <= getattr(self,name) <= ceiling, 'reader_limit_range')


def key(row):
    require(isinstance(row.get('sequence_id'),str) and type(row.get('frame_index')) is int and row['frame_index'] >= 0, 'reader_key')
    return row['sequence_id'],row['frame_index']


def features_from_fields(fields):
    values = []
    for j in range(62):
        for side in ('left','right'):
            f = fields['camera_'+side+'_pts2d']; valid = f['valid_mask'][j]; p = f['coordinates'][j]
            values.extend([2*p[0]/1023-1,2*p[1]/1023-1] if valid else [0.,0.])
            values.extend([float(valid),0.,0.,0.,0.])
    return values


class PreparedBundleReader:
    def __init__(self,root,*,completion_sha256,split,diagnostic_only,limits=None):
        require(diagnostic_only is True, 'reader_explicit_diagnostic_only_required')
        require(split in ('train','validation','test'), 'reader_explicit_split_required')
        self.limits = ReaderLimits() if limits is None else limits
        require(isinstance(self.limits,ReaderLimits), 'reader_limits_type')
        self.split = split
        self._integrity = Integrity(root,completion_sha256)
        i = self._integrity
        self._cached_path, self._cached = None,None
        self._spec = validate_spec(i.doc('metadata/spec.json',262144))
        self._owners,_ = official_splits(i.root/'metadata/split',self._spec,Budget(self._spec['limits']))
        require(i.entries['metadata/calibration.json']['sha256'] == self._spec['pinned_sources']['scaramuzza_sha256'], 'reader_calibration_pin')
        self._motion_doc = i.doc('metadata/motion_map.json')
        self._provenance = i.doc('provenance.json')
        p = self._provenance
        require(p.get('schema_version') == 'prepared_bundle_provenance_v1' and p.get('train_ready') is False and
                p.get('content_sha256') == digest({k:v for k,v in p.items() if k != 'content_sha256'}) and
                p.get('rgb_root') == 'inputs' and p.get('rgb_manifest') == 'rgb/manifest.json', 'reader_provenance')
        rows = p['records']
        require(isinstance(rows,list) and 1 <= len(rows) <= self.limits.max_records, 'reader_record_limit')
        keys = [key(r) for r in rows]
        require(keys == sorted(set(keys)), 'reader_ordered_unique_keys')
        self._motions = motion_entries(self._motion_doc,{s for s,f in keys},self._owners)
        self._rgb_doc = i.doc('rgb/manifest.json',4*MIB)
        require(self._rgb_doc['content_sha256'] == p['rgb_content_sha256'], 'reader_rgb_pin')
        self._rgb = LocalStereoRGB(i.root/'rgb/manifest.json',i.root/'inputs',
                                   limits=RGBLimits(max_records=32,max_batch_size=self.limits.max_batch_size,
                                                    max_batch_tensor_bytes=self.limits.max_batch_tensor_bytes))
        require(len(self._rgb) == len(rows) and [key(r) for r in self._rgb_doc['records']] == keys and
                self._rgb_doc['image_size'] == [1024,1024], 'reader_rgb_key_size')
        plan = i.doc('metadata/plan.json')
        matching = i.doc('metadata/matching.json')
        require(plan['content_sha256'] == p['plan_sha256'] == digest({k:v for k,v in plan.items() if k != 'content_sha256'}) and
                matching['content_sha256'] == p['matching_sha256'] == digest({k:v for k,v in matching.items() if k != 'content_sha256'}) and
                matching['plan_sha256'] == p['plan_sha256'], 'reader_plan_matching_pin')
        self._jobs = {j['id']:j for j in plan['jobs']}
        require(len(self._jobs) == len(plan['jobs']) <= 16, 'reader_jobs')
        expected_views = {}
        for job_id,job in self._jobs.items():
            for member in job['targets']:
                seq,frame,side = identify(member,job['manifest']['kind'])
                sides = expected_views.setdefault((seq,frame),{})
                require(side not in sides, 'reader_plan_duplicate_view')
                sides[side] = (job_id,member)
        require(set(expected_views) == set(keys) and all(set(v) == {'left','right','pose'} for v in expected_views.values()), 'reader_plan_target_coverage')
        scopes = {j['manifest']['source_scope'] for j in self._jobs.values()}
        versions = {j['manifest']['dataset_version'] for j in self._jobs.values()}
        require(len(scopes) == len(versions) == 1 and self._rgb_doc['dataset_version'] in versions, 'reader_dataset_scope')
        self.source_scope = next(iter(scopes))
        require(self._rgb_doc['data_kind'] == ('synthetic_fixture' if self.source_scope == 'synthetic_fixture' else 'dataset_sample'), 'reader_rgb_kind')
        self._sources = {s['job_id']:s for s in p['sources']}
        require(set(self._sources) == set(self._jobs), 'reader_source_coverage')
        self._extractions = {}
        for job_id,source in self._sources.items():
            job = self._jobs[job_id]
            require(source['path'] == f'inputs/{job_id}/extraction_manifest.json' and source['source_sha256'] == job['source_sha256'], 'reader_source_identity')
            doc = i.doc(source['path'])
            require(doc['content_sha256'] == digest({k:v for k,v in doc.items() if k != 'content_sha256'}) and
                    digest(doc) == source['extraction_manifest_sha256'] and doc['content_sha256'] == source['producer_content_sha256'] and
                    doc['archive_manifest'] == job['manifest'] and doc['archive_manifest_sha256'] == digest(job['manifest']) and
                    doc['persistent_index']['source_sha256'] == job['source_sha256'] and doc['train_ready'] is False and
                    doc['targets_complete'] is True and doc['exact_targets'] == job['targets'] and
                    len(doc['files']) == len(job['targets']) and {r['member'] for r in doc['files']} == set(job['targets']), 'reader_producer_pin')
            for entry in doc['files']:
                require(entry['relative_path'] == 'files/'+entry['member'] and entry['kind'] == job['manifest']['kind'] and entry['tar_complete'] is True,
                        'reader_producer_file')
                actual_entry = i.entries.get(f"inputs/{job_id}/{entry['relative_path']}",{})
                require(actual_entry.get('sha256') == entry['sha256'] and actual_entry.get('size_bytes') == entry['size_bytes'], 'reader_producer_payload_pin')
            self._extractions[job_id] = {r['member']:r for r in doc['files']}
        require(matching['frames'] == [{'sequence_id':r['sequence_id'],'frame_index':r['frame_index'],'views':r['views']} for r in rows], 'reader_matching_rows')
        self._rows = []
        all_pose_indices = {}
        for index,row in enumerate(rows):
            rgb = self._rgb_doc['records'][index]
            require(row['rgb_record_index'] == index and row['split'] == rgb['split'] == self._owners[row['sequence_id']], 'reader_split')
            pose = row['pose']; job_id = pose['job_id']
            require(pose['candidate'] == f'pose/{job_id}/candidate.json' and job_id == row['views']['pose']['job_id'] and
                    self._jobs[job_id]['manifest']['kind'] == 'pose' and type(pose['record_index']) is int and pose['record_index'] >= 0, 'reader_pose_index')
            for side in ('left','right','pose'):
                view = row['views'][side]; entry = self._extractions[view['job_id']].get(view['member'])
                require(entry is not None and entry['sha256'] == view['sha256'] and identify(view['member'],entry['kind']) == (*key(row),side), 'reader_view_key_pin')
                require((view['job_id'],view['member']) == expected_views[key(row)][side], 'reader_plan_view_pin')
                if side != 'pose':
                    v = rgb['views'][side]; source = self._rgb_doc['sources'][v['source_id']]
                    job = self._jobs[view['job_id']]
                    require(v['path'] == f"{view['job_id']}/{entry['relative_path']}" and v['member'] == view['member'] and
                            v['sha256'] == entry['sha256'] and v['size_bytes'] == entry['size_bytes'] and
                            source['archive_manifest'] == job['manifest'] and source['extraction_manifest_sha256'] == self._sources[view['job_id']]['producer_content_sha256'], 'reader_rgb_producer_pin')
            all_pose_indices.setdefault(pose['candidate'],[]).append((pose['record_index'],key(row)))
            self._rows.append(copy.deepcopy(row))
        self._candidate_indices = all_pose_indices
        self._blockers = set()
        self._contract = None
        for path in sorted(all_pose_indices):
            self._candidate(path)  # 한 번에 한 job 수치/출처 감사. 모든 candidate를 보관하지 않는다.
        self._cached_path,self._cached = None,None
        self._selected = [r for r in self._rows if r['split'] == split]
        require(bool(self._selected), 'reader_empty_split')
        self._reference_paths = [n for n in i.entries if not n.startswith('pose/') or not n.endswith('candidate.json')]
        self._reference_paths = [n for n in self._reference_paths if not '/files/' in n]
        i.check_marker()

    def _candidate(self,path):
        i = self._integrity
        if path == self._cached_path:
            i.verify_file(path)
            job_id = path.split('/')[1]
            for entry in self._extractions[job_id].values():
                i.verify_file(f"inputs/{job_id}/{entry['relative_path']}")
            return self._cached
        # 이전 cache의 참조를 먼저 해제하고 새 JSON 하나만 읽는다.
        self._cached_path,self._cached = None,None
        candidate = validate_candidate(i.doc(path,self.limits.max_candidate_bytes))
        require(len(candidate['records']) <= self.limits.max_records, 'reader_candidate_record_limit')
        require(candidate['preparation_spec'] == self._spec and candidate['contract']['data_kind'] == 'dataset_sample' and
                candidate['contract']['input_source'] == 'gt_2d' and candidate['contract']['training_release'] == 'not_implemented' and
                candidate['training_eligibility']['blockers'], 'reader_candidate_gate')
        if self._contract is None:
            self._contract = copy.deepcopy(candidate['contract'])
        require(self._contract == candidate['contract'], 'reader_candidate_contract_mismatch')
        job_id = path.split('/')[1]; job = self._jobs[job_id]
        manifest = candidate['manifest']; sources = manifest['source_hashes']
        em_path = f'inputs/{job_id}/extraction_manifest.json'
        subset_path = f'pose/{job_id}/motion_map.json'
        subset = i.doc(subset_path)
        seqs = {r['sequence_id'] for r in candidate['records']}
        require(subset == {'schema_version':'explicit_motion_map_v1','sequences':{s:self._motions[s] for s in seqs}}, 'reader_job_motion_subset')
        require(manifest['archive_id'] == job['manifest']['archive_id'] and manifest['dataset_version'] == job['manifest']['dataset_version'] and
                sources['spec'] == i.entries['metadata/spec.json']['sha256'] and
                sources['calibration'] == i.entries['metadata/calibration.json']['sha256'] and
                sources['motion_map'] == i.entries[subset_path]['sha256'] and sources['extraction_manifest'] == i.entries[em_path]['sha256'], 'reader_candidate_source_pin')
        for split in ('train','validation','test'):
            require(sources['official_split_files'][split]['sha256'] == i.entries[f'metadata/split/{split}.txt']['sha256'] == self._spec['pinned_sources']['split_sha256'][split], 'reader_candidate_split_pin')
        require([key(r) for r in candidate['records']] == [k for _,k in sorted(self._candidate_indices[path])] and
                [n for n,_ in sorted(self._candidate_indices[path])] == list(range(len(candidate['records']))), 'reader_candidate_keys')
        require(manifest['raw_files'] == [r['source'] for r in candidate['records']], 'reader_candidate_raw_sources')
        require({r['source']['member'] for r in candidate['records']} == set(self._extractions[job_id]), 'reader_candidate_source_coverage')
        for row in candidate['records']:
            entry = self._extractions[job_id].get(row['source']['member'])
            require(entry is not None and identify(entry['member'],'pose') == (*key(row),'pose') and
                    row['source']['sha256'] == entry['sha256'] and row['source']['size_bytes'] == entry['size_bytes'] and
                    row['source']['extraction_manifest_sha256'] == sources['extraction_manifest'], 'reader_candidate_raw_pin')
            require(row['split'] == self._owners[row['sequence_id']] and row['motion_identity'] == self._motions[row['sequence_id']] and
                    row['motion_id'] == self._motions[row['sequence_id']]['motion_id'], 'reader_candidate_split_motion')
            raw = i.doc(f"inputs/{job_id}/{entry['relative_path']}",4*MIB)
            require(str(raw.get('frame')) == str(row['frame_index']) and isinstance(raw.get('joints'),dict), 'reader_raw_frame')
            for field,dimension in REQUIRED_FIELDS.items():
                coords,mask = [],[]
                for name in self._contract['joint_names']:
                    joint = raw['joints'].get(name,{})
                    value = joint.get(field) if isinstance(joint,dict) else None
                    valid = vector(value,dimension)
                    coords.append([x*self._spec['coordinates']['mm_per_raw_unit'] for x in value] if valid and dimension == 3 else value[:] if valid else None)
                    mask.append(valid)
                require(row['fields'][field] == {'coordinates':coords,'valid_mask':mask}, 'reader_raw_field_or_unit_mismatch')
            require(row['observation_features'] == features_from_fields(row['fields']), 'reader_feature_cache_mismatch')
            require(row['visibility'] == {'left':[None]*62,'right':[None]*62} and row['confidence'] == {'left':[None]*62,'right':[None]*62} and
                    row['time']['time_seconds'] is None and row['time']['timestamp_semantics'] == 'unverified', 'reader_unknown_visibility_time')
        self._blockers.update(candidate['training_eligibility']['blockers'])
        self._cached_path,self._cached = path,candidate
        return candidate

    def __len__(self):
        return len(self._selected)

    def _index(self,index):
        require(type(index) is int and 0 <= index < len(self), 'reader_index')
        return self._selected[index]

    @property
    def identity(self):
        return {'schema_version':'prepared_bundle_t1_data_v1','completion_sha256':self._integrity.marker_sha256,
                'data_kind':'dataset_sample','source_scope':self.source_scope,'diagnostic_only':True,'train_ready':False,
                'input_source':'gt_2d','training_release':'not_implemented','split':self.split,
                'joint_names':copy.deepcopy(self._contract['joint_names']), 'distance_unit':'mm','coordinate_frame':'device_relative',
                'temporal_ready':False,'time_steps':1,'visibility_semantics':'unknown_separate_from_annotation_valid',
                'blockers':sorted(self._blockers),'sample_keys':[list(key(r)) for r in self._selected]}

    def metadata(self,index):
        row = self._index(index)
        return {**copy.deepcopy(row),'motion_identity':copy.deepcopy(self._motions[row['sequence_id']]),
                'data_kind':'dataset_sample','source_scope':self.source_scope,'input_source':'gt_2d',
                'diagnostic_only':True,'train_ready':False,'time_steps':1,'time_seconds':None,'temporal_ready':False}

    def load_batch(self,indices,*,include_rgb=True):
        require(type(include_rgb) is bool and isinstance(indices,(list,tuple)) and 1 <= len(indices) <= self.limits.max_batch_size, 'reader_batch_size')
        rows = [self._index(n) for n in indices]
        # T=1 GT2D/mask/GT3D/특징/시간mask/RGB의 반환 tensor byte를 합산한다.
        per_sample = 2*62*(2*4+1)+62*(3*4+1)+868*4+1+(2*3*1024**2*4 if include_rgb else 0)
        require(len(rows)*per_sample <= self.limits.max_batch_tensor_bytes, 'reader_batch_tensor_limit')
        i = self._integrity; i.begin()
        for path in self._reference_paths:
            i.verify_file(path)
        import torch
        count = len(rows)
        xy = torch.zeros((count,2,62,2),dtype=torch.float32,device='cpu')
        annotation = torch.zeros((count,2,62),dtype=torch.bool,device='cpu')
        target = torch.zeros((count,1,62,3),dtype=torch.float32,device='cpu')
        target_valid = torch.zeros((count,1,62),dtype=torch.bool,device='cpu')
        features = torch.zeros((count,1,868),dtype=torch.float32,device='cpu')
        for b,row in enumerate(rows):
            candidate = self._candidate(row['pose']['candidate'])
            record = candidate['records'][row['pose']['record_index']]
            for v,side in enumerate(('left','right')):
                field = record['fields']['camera_'+side+'_pts2d']
                for j,valid in enumerate(field['valid_mask']):
                    if valid:
                        xy[b,v,j] = torch.tensor(field['coordinates'][j],dtype=torch.float32,device='cpu'); annotation[b,v,j] = True
            field = record['fields']['device_pts3d']
            for j,valid in enumerate(field['valid_mask']):
                if valid:
                    target[b,0,j] = torch.tensor(field['coordinates'][j],dtype=torch.float32,device='cpu'); target_valid[b,0,j] = True
            features[b,0] = torch.tensor(record['observation_features'],dtype=torch.float32,device='cpu')
            # 다음 job으로 전환하기 전에 현재 candidate의 지역 참조도 해제한다.
            del record,candidate,field
            i.check()
        require(torch.isfinite(xy).all().item() and torch.isfinite(target).all().item() and torch.isfinite(features).all().item(), 'reader_float32_nonfinite')
        result = {'gt_xy':xy,'annotation_valid':annotation,'target_mm':target,'target_valid':target_valid,
                  'features':features,'time_valid':torch.ones((count,1),dtype=torch.bool,device='cpu'),
                  'metadata':[self.metadata(n) for n in indices], 'identity':self.identity}
        if include_rgb:
            # 기존 loader가 같은 split, native pixels, PNG SHA/CRC를 검사하며 캐시하지 않는다.
            result['images'] = self._rgb.load_batch([r['rgb_record_index'] for r in rows])['images']
        i.check_marker(); i.check()
        return result

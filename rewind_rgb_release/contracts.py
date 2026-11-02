"""로컬 단일 운영자 신뢰 모델. 자동 검증과 외부 인간 검토 권한을 분리한다."""
from pathlib import Path
import re

from rewind_archive.contracts import digest
from rewind_bundle_data import PreparedBundleReader
from rewind_bundle_data.integrity import local, require, sha, MIB
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _regular_read

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'native_rgb2d_t1_train_validation_v1'
CATEGORIES = ('source_provenance','motion_identity','split_independence','rgb_pose_alignment','joint_definition','units_geometry')
ALLOWED_BLOCKERS = {'candidate_schema_not_training_release','rgb_pose_alignment_not_checked',
    'independent_validation_not_verified','required_train_validation_splits_missing',
    'original_motion_identity_unverified','gapped_sequences_require_contiguous_selection'}
ACTUAL_SCOPES = {'official_server','official_prefix_local','user_drive','local_unverified'}


def code_identity():
    # 재사용 모델/loss/resize/checkpoint와 reader·검증·변환 코드를 함께 고정한다.
    return {p.relative_to(ROOT).as_posix():sha(p.read_bytes())
        for package in ('rewind_rgb_release','rewind_bundle_data','rewind_m1','rewind_prepare','rewind_data','rewind_archive')
        for p in sorted((ROOT/package).glob('*.py'))}


def read(path,maximum=MIB):
    path = local(path)
    raw = _regular_read(path,maximum)
    return read_json(raw),sha(raw)


def seal(doc):
    return {**doc,'content_sha256':digest({k:v for k,v in doc.items() if k != 'content_sha256'})}


def facts(root,pin,check_budget=lambda:None):
    readers = {}
    for split in ('train','validation'):
        check_budget()
        readers[split] = PreparedBundleReader(root,completion_sha256=pin,split=split,diagnostic_only=True)
        check_budget()
    require(readers['train']._integrity.marker_file_sha256 == readers['validation']._integrity.marker_file_sha256,
            'release_preparation_snapshot_changed')
    train = readers['train']; rows = sorted(train._rows,key=lambda r:(r['sequence_id'],r['frame_index']))
    require(len(rows) <= 32, 'release_record_limit')
    require(all(r['split'] in ('train','validation','test') for r in rows), 'release_split')
    return readers, {'completion_sha256':pin,'completion_file_sha256':train._integrity.marker_file_sha256,
        'source_scope':train.source_scope,'candidate_contract':train._contract,
        'key_coverage':[[r['sequence_id'],r['frame_index'],r['split']] for r in rows],
        'input_sha256':read(train._integrity.root/'COMPLETE.json')[0]['input_sha256'],
        'candidate_blockers':{s:readers[s].identity['blockers'] for s in readers},
        'original_motion_map':train._motion_doc,'counts':{s:len(r) for s,r in readers.items()},
        'scope':SCOPE,'temporal_ready':False,'whole_archive_coverage':'not_measured'}


def validate_review(path, data):
    review,identity = read(path)
    root = local(path).parent
    coverage = digest(data['key_coverage'])
    require(review.get('schema_version') == 'rgb2d_human_review_v1' and review.get('scope') == SCOPE and
            review.get('completion_sha256') == data['completion_sha256'] and review.get('key_coverage_sha256') == coverage,
            'release_review_scope_coverage')
    require(set(review.get('evidence',{})) == set(CATEGORIES), 'release_review_categories')
    evidence_pins = {}
    reviewers = set()
    for category,row in review['evidence'].items():
        require(set(row) == {'artifact','artifact_sha256','reviewer','date','conclusion','limitations','key_coverage_sha256'}, 'release_evidence_fields')
        require(isinstance(row['reviewer'],str) and 0 < len(row['reviewer'].strip()) <= 128 and
                isinstance(row['date'],str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',row['date']) and
                isinstance(row['limitations'],list) and all(isinstance(x,str) and 0 < len(x) <= 2048 for x in row['limitations']) and
                row['key_coverage_sha256'] == coverage and row['conclusion'] in
                (('adopted_assumptions_for_rgb2d_t1','verified_facts_for_rgb2d_t1') if category == 'units_geometry' else ('reviewed_for_rgb2d_t1',)),
                'release_evidence_review_required')
        from rewind_archive.contracts import safe_path
        relative = safe_path(row['artifact'])
        artifact = local(root/relative)
        require(artifact.is_relative_to(root), 'release_evidence_path')
        raw = _regular_read(artifact,MIB)
        require(sha(raw) == row['artifact_sha256'], 'release_evidence_hash')
        evidence_pins[category] = {'sha256':sha(raw),**row}
        reviewers.add(row['reviewer'])
    motions = review.get('reviewed_motion_map')
    original = data['original_motion_map']['sequences']
    owners = {seq:split for seq,frame,split in data['key_coverage']}
    require(isinstance(motions,dict) and set(motions) == set(owners), 'release_motion_coverage')
    motion_owners = {}
    for seq,row in motions.items():
        require(set(row) == {'original_motion_id','candidate_motion_id','evidence_note'} and
                isinstance(row['original_motion_id'],str) and re.fullmatch('[A-Za-z0-9_:/.-]{1,256}',row['original_motion_id']) and
                row['candidate_motion_id'] == original[seq]['motion_id'] and
                isinstance(row['evidence_note'],str) and 0 < len(row['evidence_note'].strip()) <= 2048, 'release_motion_identity')
        require(motion_owners.get(row['original_motion_id'],owners[seq]) == owners[seq], 'release_motion_split_leak')
        motion_owners[row['original_motion_id']] = owners[seq]
    require(set(review.get('geometry_statement',{})) == {'verified_facts','adopted_assumptions'} and
            all(isinstance(v,list) and v and all(isinstance(x,str) and 0 < len(x) <= 2048 for x in v)
                for v in review['geometry_statement'].values()), 'release_geometry_fact_assumption')
    return {'review_file_sha256':identity,'evidence':evidence_pins,'reviewers':sorted(reviewers),
            'reviewed_motion_map':motions,'geometry_statement':review['geometry_statement']}


def proposal_from_data(data,config,review_path):
    require(data['source_scope'] in ACTUAL_SCOPES, 'release_synthetic_not_real')
    review = validate_review(review_path,data)
    require(all(set(b) <= ALLOWED_BLOCKERS for b in data['candidate_blockers'].values()), 'release_unhandled_candidate_blocker')
    return seal({'schema_version':'reviewed_rgb2d_release_v1','scope':SCOPE,'data':data,
                 'review':review,'config_sha256':digest(config),'code_sha256':code_identity()})


def draft(root,pin,config,review_path):
    _,data = facts(root,pin)
    return proposal_from_data(data,config,review_path)


def validate_policy(path,expected_sha256,proposal,proposal_sha256):
    policy,policy_sha = read(path)
    require(isinstance(expected_sha256,str) and policy_sha == expected_sha256, 'release_external_policy_pin')
    require(set(policy) == {'schema_version','trust_model','scope','allowed_releases'} and
            policy['schema_version'] == 'local_rgb2d_review_policy_v1' and
            policy['trust_model'] == 'external_operator_pinned_allowlist' and policy['scope'] == SCOPE,
            'release_policy_schema')
    allow = {'release_file_sha256':proposal_sha256,'release_content_sha256':proposal['content_sha256'],
             'review_file_sha256':proposal['review']['review_file_sha256'],
             'code_sha256':digest(proposal['code_sha256']),'reviewers':proposal['review']['reviewers']}
    require(isinstance(policy['allowed_releases'],list) and 1 <= len(policy['allowed_releases']) <= 32 and
            allow in policy['allowed_releases'], 'release_not_in_external_allowlist')
    return policy_sha


def admit(request,config,check_budget=lambda:None):
    require(isinstance(request,dict) and request.get('mode') in ('reviewed_real','development_fixture'), 'release_explicit_mode')
    require(set(request) == ({'mode','bundle','completion_sha256'} if request['mode'] == 'development_fixture' else
            {'mode','bundle','completion_sha256','release','review','policy','expected_policy_sha256'}), 'release_request_fields')
    readers,data = facts(request['bundle'],request['completion_sha256'],check_budget)
    if request['mode'] == 'development_fixture':
        require(data['source_scope'] == 'synthetic_fixture', 'development_fixture_source_required')
        binding = {'mode':'development_fixture','scope':SCOPE,'data':data,'config_sha256':digest(config),
                   'code_sha256':code_identity(),'review':None,'policy_sha256':None,'release_file_sha256':None}
    else:
        require(data['source_scope'] in ACTUAL_SCOPES, 'release_synthetic_not_real')
        policy_path,release_path = local(request['policy']),local(request['release'])
        require(not policy_path.is_relative_to(local(request['bundle'])) and
                not policy_path.is_relative_to(release_path.parent), 'release_external_policy_required')
        proposal,proposal_sha = read(release_path)
        expected = proposal_from_data(data,config,request['review'])
        require(proposal == expected, 'release_payload_bundle_code_review_changed')
        policy_sha = validate_policy(policy_path,request['expected_policy_sha256'],proposal,proposal_sha)
        binding = {**proposal,'mode':'reviewed_real','policy_sha256':policy_sha,'release_file_sha256':proposal_sha}
    binding = seal(binding)
    return readers,binding


def check_authority(request,binding):
    """허용 이후에도 외부 policy/release/review/evidence의 변화를 거부한다."""
    if binding['mode'] != 'reviewed_real':
        return
    require(read(request['release'])[1] == binding['release_file_sha256'], 'release_authority_payload_changed')
    require(read(request['policy'])[1] == binding['policy_sha256'] == request['expected_policy_sha256'], 'release_authority_policy_changed')
    require(validate_review(request['review'],binding['data']) == binding['review'], 'release_authority_review_changed')


def audit_candidate(path):
    """기존 비-bundle 실제 자료도 원본 그대로 차단 이유를 보고한다. 허용 발급 없음."""
    from rewind_prepare.converter import validate_candidate
    doc,identity = read(path,4*MIB);validate_candidate(doc)
    splits = sorted({r['split'] for r in doc['records']})
    return {'schema_version':'real_rgb2d_candidate_audit_v1','candidate_file_sha256':identity,'train_ready':False,
        'release_issued':False,'original_blockers':doc['training_eligibility']['blockers'],
        'counts':{s:sum(r['split'] == s for r in doc['records']) for s in splits},
        'key_coverage':[[r['sequence_id'],r['frame_index'],r['split']] for r in doc['records']],
        'motion_declarations':{r['sequence_id']:r['motion_identity'] for r in doc['records']},
        'unit_status':doc['preparation_spec']['coordinates']['unit_evidence_status'],
        'admission_blockers':['completed_bundle_required','nonempty_independent_validation_required',
            'reviewed_original_motion_identity_required','external_review_evidence_and_policy_required'],
        'minimum_followup':['독립 원본 motion의 validation stereo+pose와 완료 bundle',
            '전체 sequence의 원본 motion 식별/중복배제 근거','좌우/RGB-pose 내용 정합·관절·단위/좌표 채택 검토 자료',
            '운영자가 별도 검토한 외부 policy의 고정 SHA']}

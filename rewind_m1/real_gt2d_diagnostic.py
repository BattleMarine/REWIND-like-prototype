"""학습 차단 실제 candidate의 GT2D→3D 단일 frame CPU 진단. 일반 loader를 해제하지 않는다."""
import argparse
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import torch

from rewind_archive.contracts import canonical,digest
from rewind_archive.matching import verify_extraction,identify
from rewind_data.audit import read_json,REQUIRED_FIELDS
from rewind_data.windows import CausalWindow,FrameSnapshot,FieldSnapshot
from rewind_prepare.converter import validate_candidate,vector
from .data import pack_window,TensorSplit
from .local_rgb import LocalStereoRGB,_regular_read
from .local_diagnostic import _preserve_state
from .metrics import masked_mse,metric_sums
from .models import make_model
from .training import Trainer,validate_config,require_torch_api,seed_run,rng_state,restore_rng,source_identity,save_checkpoint

ROOT=Path(__file__).resolve().parents[1]
ROLE='diagnostic_only_gt2d_single_frame'
FORMAT='real_gt2d_diagnostic_checkpoint_v1'
CONFIG={'model':'single_frame','window_size':1,'hidden':32,'dropout':0.1,'batch_size':2,
        'learning_rate':0.01,'seed':37,'target_scale_mm':1000.,'amp':False,'scheduler_step':50,'scheduler_gamma':0.8}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def tensor_sha(tensor):return sha(canonical({'dtype':str(tensor.dtype),'shape':list(tensor.shape)})+bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
def read(path,limit=2*1024**2):return _regular_read(Path(path).absolute(),limit)


def diagnostic_source_identity():
    identity=source_identity()
    for package in ('rewind_prepare','rewind_archive'):
        identity['code_sha256'].update({p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted((ROOT/package).glob('*.py'))})
    return identity


@dataclass
class DiagnosticData:
    tensors:TensorSplit
    identity:dict
    contract:dict
    blockers:list


def load_diagnostic(prepared,*,diagnostic_only,candidate_sha256,rgb_manifest_sha256):
    if diagnostic_only is not True:raise ValueError('explicit_diagnostic_only_required')
    root=Path(prepared).absolute()
    raw=read(root/'pose/candidate.json');rgb_raw=read(root/'rgb/manifest.json')
    if sha(raw)!=candidate_sha256 or sha(rgb_raw)!=rgb_manifest_sha256:raise ValueError('diagnostic_input_pin_mismatch')
    candidate=validate_candidate(read_json(raw));contract=candidate['contract'];spec=candidate['preparation_spec']
    if contract['data_kind']!='dataset_sample' or contract['input_source']!='gt_2d' or contract['distance_unit']!='mm':raise ValueError('diagnostic_real_gt_mm_required')
    if contract.get('training_release')!='not_implemented' or not candidate['training_eligibility']['blockers']:raise ValueError('diagnostic_candidate_blockers_required')
    spec_raw=read(ROOT/'configs/real_archvis62_v1.json')
    sources=candidate['manifest']['source_hashes']
    if read_json(spec_raw)!=spec or sha(spec_raw)!=sources['spec']:raise ValueError('diagnostic_spec_mismatch')
    metadata={}
    for key,relative in (('calibration','metadata/scaramuzza.json'),('motion_map','metadata/motion_map.json')):
        data=read(root/relative)
        if sha(data)!=sources[key]:raise ValueError('diagnostic_metadata_hash')
        metadata[relative]=sha(data)
    if metadata['metadata/scaramuzza.json']!=spec['pinned_sources']['scaramuzza_sha256']:raise ValueError('diagnostic_calibration_pin')
    owners={}
    for split in ('train','validation','test'):
        data=read(root/f'metadata/split/{split}.txt')
        if sha(data)!=sources['official_split_files'][split]['sha256'] or sha(data)!=spec['pinned_sources']['split_sha256'][split]:raise ValueError('diagnostic_split_pin')
        for sequence in data.decode('utf-8-sig').splitlines():
            if sequence in owners:raise ValueError('diagnostic_split_duplicate')
            owners[sequence]=split
        metadata['split/'+split]=sha(data)
    verified={k:verify_extraction(root/'inputs'/k,allow_code_change=True) for k in ('left','right','pose')}
    if any(v['archive_manifest']['dataset_version']!=candidate['manifest']['dataset_version'] for v in verified.values()):raise ValueError('diagnostic_extraction_dataset')
    pose_report=read(root/'inputs/pose/extraction_manifest.json')
    if sha(pose_report)!=sources['extraction_manifest']:raise ValueError('diagnostic_pose_report_hash')
    records=candidate['records'];keys=[(r['sequence_id'],r['frame_index']) for r in records]
    if len(records)!=2 or len(set(keys))!=2 or any(type(f) is not int or f<0 for s,f in keys):raise ValueError('diagnostic_two_unique_frames_required')
    if any(r['split']!='train' or owners.get(r['sequence_id'])!='train' for r in records):raise ValueError('diagnostic_train_only')
    loader=LocalStereoRGB(root/'rgb/manifest.json',root/'inputs')
    rgb_document=read_json(rgb_raw)
    if rgb_document['data_kind']!='dataset_sample' or rgb_document['dataset_version']!=candidate['manifest']['dataset_version']:raise ValueError('diagnostic_rgb_dataset')
    rgb_keys=[(r['sequence_id'],r['frame_index']) for r in rgb_document['records']]
    if len(rgb_keys)!=2 or set(rgb_keys)!=set(keys):raise ValueError('diagnostic_rgb_pose_keys')
    pose_files={r['member']:r for r in verified['pose']['files']}
    if set(pose_files)!={r['source']['member'] for r in records}:raise ValueError('diagnostic_pose_selection')
    motion=read_json(read(root/'metadata/motion_map.json'))['sequences'];packed=[];outside=[]
    names=tuple(contract['joint_names']);factor=spec['coordinates']['mm_per_raw_unit']
    if len(names)!=62:raise ValueError('diagnostic_fixed_62_joints')
    for row in records:
        source=row['source'];entry=pose_files.get(source['member'])
        if source['extraction_manifest_sha256']!=sha(pose_report) or source['sha256']!=entry['sha256'] or source['size_bytes']!=entry['size_bytes']:raise ValueError('diagnostic_pose_source')
        if identify(source['member'],'pose')!=(row['sequence_id'],row['frame_index'],'pose'):raise ValueError('diagnostic_pose_source_key')
        if row['motion_identity']!=motion[row['sequence_id']] or row['motion_id']!=motion[row['sequence_id']]['motion_id']:raise ValueError('diagnostic_motion_provenance')
        original=read_json(read(root/'inputs/pose'/entry['relative_path'],4*1024**2))
        snapshots=[]
        for field,dimension in REQUIRED_FIELDS.items():
            coordinates=[];mask=[]
            for name in names:
                value=original['joints'].get(name,{}).get(field);valid=vector(value,dimension)
                coordinates.append([x*factor for x in value] if valid and dimension==3 else value[:] if valid else None);mask.append(valid)
            if row['fields'][field]!={'coordinates':coordinates,'valid_mask':mask}:raise ValueError('diagnostic_raw_field_or_unit_mismatch')
            snapshots.append(FieldSnapshot(field,tuple(None if p is None else tuple(p) for p in coordinates),tuple(mask)))
        frame=FrameSnapshot(row['sequence_id'],row['frame_index'],source['member'],row.get('timestamp_raw'),tuple(snapshots),())
        window=CausalWindow(names,tuple(REQUIRED_FIELDS),(frame,),'diagnostic_independent_frame')
        # pack_window만 재사용한다. InputContract 검증 gate/일반 load_document는 통과시키지 않는다.
        values=pack_window(window,SimpleNamespace(**{**contract,'joint_names':names}))
        if not torch.equal(values[0][0],torch.tensor(row['observation_features'],dtype=torch.float32)):raise ValueError('diagnostic_feature_cache_mismatch')
        packed.append(values)
        outside.append({side:sum(valid and not (0<=point[0]<size[0] and 0<=point[1]<size[1]) for point,valid in zip(row['fields']['camera_'+side+'_pts2d']['coordinates'],row['fields']['camera_'+side+'_pts2d']['valid_mask']))
                        for side,size in zip(('left','right'),contract['image_sizes'])})
    for index,rgb_row in enumerate(rgb_document['records']):
        if rgb_row['split']!='train':raise ValueError('diagnostic_rgb_train_only')
        for side in ('left','right'):
            view=rgb_row['views'][side];entry=next((f for f in verified[side]['files'] if f['member']==view['member']),None)
            if entry is None or entry['sha256']!=view['sha256'] or entry['size_bytes']!=view['size_bytes']:raise ValueError('diagnostic_rgb_source')
            source=rgb_document['sources'][view['source_id']]
            if source['archive_manifest']!=verified[side]['archive_manifest'] or source['extraction_manifest_sha256']!=verified[side]['content_sha256']:
                raise ValueError('diagnostic_rgb_provenance')
            if identify(view['member'],'rgb')!=(rgb_row['sequence_id'],rgb_row['frame_index'],side):raise ValueError('diagnostic_rgb_source_key')
            if view['image_size']!=contract['image_sizes'][('left','right').index(side)]:raise ValueError('diagnostic_image_size')
        # PNG hash/CRC/실제 해상도까지 검사한다. 영상은 모델 입력에 넣지 않는다.
        del_image=loader.load_batch([index]);del del_image
    tensors=TensorSplit(*(torch.stack([p[i] for p in packed]) for i in range(4)),keys)
    if tensors.features.shape!=(2,1,868) or tensors.target_mm.shape!=(2,1,62,3) or not tensors.target_valid[:,-1].any(1).all():raise ValueError('diagnostic_tensor_shape_or_mask')
    identity={'schema':'real_gt2d_diagnostic_data_v1','data_kind':'dataset_sample','input_source':'gt_2d','target_unit':'mm',
        'candidate_file_sha256':sha(raw),'rgb_manifest_file_sha256':sha(rgb_raw),'spec_file_sha256':sha(spec_raw),
        'metadata_sha256':metadata,'raw_pose':copy.deepcopy(candidate['manifest']['raw_files']),
        'extractions':{k:{'content_sha256':v['content_sha256'],'producer_code_sha256':v['code_sha256']} for k,v in verified.items()},
        'sample_keys':keys,'joint_names':list(names),'features_sha256':tensor_sha(tensors.features),'target_sha256':tensor_sha(tensors.target_mm),
        'mask_sha256':tensor_sha(tensors.target_valid),'outside_image_counts':outside,'visibility_semantics':'unknown; separate from valid_mask',
        'candidate_contract':contract,'unit_status':spec['coordinates']['unit_evidence_status'],'diagnostic_only':True,'train_ready':False}
    return DiagnosticData(tensors,identity,contract,list(candidate['training_eligibility']['blockers']))


class DiagnosticTrainer:
    """일반 Trainer 생성자/evaluate를 사용하지 않는 전용 실행 상태."""
    def __init__(self,data,config):
        require_torch_api();self.config=validate_config(config)
        if config['model']!='single_frame' or config['window_size']!=1 or config['amp'] or config['batch_size']!=2 or config['hidden']>64:raise ValueError('diagnostic_single_frame_cpu_limits')
        seed_run(config['seed']);self.data=data;self.splits={'train':data.tensors};self.device=torch.device('cpu')
        self.model=make_model(config,868,62);self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,config['scheduler_step'],config['scheduler_gamma'])
        self.scaler=torch.amp.GradScaler('cuda',enabled=False);self.step=0;self.source=diagnostic_source_identity()
        self.environment={'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':torch.get_num_threads(),'amp':False}

    def train_step(self):
        loss=Trainer.train_step(self)
        norm=math.sqrt(sum(float(p.grad.square().sum()) for p in self.model.parameters() if p.grad is not None))
        if not math.isfinite(norm) or not math.isfinite(loss):raise ValueError('diagnostic_nonfinite_update')
        return {'step':self.step,'normalized_batch_mse':loss,'gradient_norm_l2':norm}

    def fit(self):
        before=self.model.training;self.model.eval();data=self.data.tensors
        with torch.no_grad():
            prediction=self.model(data.features,data.time_valid)[:,-1]*self.config['target_scale_mm']
            sums=metric_sums(prediction,data.target_mm[:,-1],data.target_valid[:,-1]);count=sums['valid_joints']
            loss=masked_mse(prediction,data.target_mm[:,-1],data.target_valid[:,-1],self.config['target_scale_mm'])
        self.model.train(before)
        return {'evaluation_kind':'train_fit_diagnostic','normalized_mse':float(loss),'mpjpe_mm':sums['distance_sum_mm']/count,'valid_joints':count}

    def checkpoint(self):
        return {'format_version':FORMAT,'role':ROLE,'diagnostic_only':True,'train_ready':False,'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),
            'rng':rng_state(),'step':self.step,'samples_seen':self.step*2,'config':self.config,'data_identity':self.data.identity,
            'source':self.source,'environment':self.environment,'sampler_policy':'random_with_replacement_torch_global_rng'}

    def resume(self,checkpoint):
        if checkpoint.get('format_version')!=FORMAT or checkpoint.get('role')!=ROLE or checkpoint.get('diagnostic_only') is not True or checkpoint.get('train_ready') is not False:raise ValueError('diagnostic_checkpoint_role')
        for key,expected in (('config',self.config),('data_identity',self.data.identity),('source',self.source),('environment',self.environment)):
            if checkpoint[key]!=expected:raise ValueError('diagnostic_resume_identity')
        step=checkpoint['step']
        if type(step) is not int or not 0<=step<=100 or checkpoint['samples_seen']!=step*2:raise ValueError('diagnostic_checkpoint_position')
        for name in ('model','optimizer','scheduler','scaler'):getattr(self,name).load_state_dict(checkpoint[name])
        restore_rng(checkpoint['rng']);self.step=step


def equal_state(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if type(left)!=type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def independence(trainer):
    model=trainer.model;before=model.training;model.eval();data=trainer.data.tensors
    with torch.no_grad():
        batch=model(data.features,data.time_valid)
        alone=torch.cat([model(data.features[i:i+1],data.time_valid[i:i+1]) for i in range(2)])
        reordered=model(data.features.flip(0),data.time_valid.flip(0)).flip(0)
        perturbed=data.features.clone();perturbed[1]+=17
        first=model(perturbed,data.time_valid)[0]
    model.train(before)
    return {'single_frame_T':1,'alone_equal':torch.allclose(batch,alone,rtol=1e-6,atol=1e-6),
            'reorder_equal':torch.allclose(batch,reordered,rtol=1e-6,atol=1e-6),
            'other_frame_perturbation_equal':torch.allclose(batch[0],first,rtol=1e-6,atol=1e-6),
            'alone_max_abs_mm':float((batch-alone).abs().max())*trainer.config['target_scale_mm']}


def run(prepared,output,*,diagnostic_only,candidate_sha256,rgb_manifest_sha256,config=None):
    if diagnostic_only is not True:raise ValueError('explicit_diagnostic_only_required')
    output=Path(output).absolute();prepared=Path(prepared).absolute()
    if output.exists() or output.is_relative_to(prepared) or prepared.is_relative_to(output):raise ValueError('diagnostic_new_separate_output')
    output.mkdir(parents=True);started=time.monotonic();total=0
    report={'schema_version':'real_gt2d_single_frame_diagnostic_v1','status':'started','diagnostic_only':True,
        'data_kind':'dataset_sample','input_source':'gt_2d','evaluation_kind':'train_fit_diagnostic','train_ready':False,
        'rgb_prediction_performance':False,'temporal_experiment':False,'validation_or_test_run':False,
        'limits':{'optimizer_updates':200,'max_seconds':300,'device':'cpu','samples':2,'hidden_max':64},'trace':[]}
    def check():
        if time.monotonic()-started>300 or total>200:raise ValueError('diagnostic_budget_exceeded')
    try:
        with _preserve_state():
            data=load_diagnostic(prepared,diagnostic_only=True,candidate_sha256=candidate_sha256,rgb_manifest_sha256=rgb_manifest_sha256)
            report.update(data_identity=data.identity,blockers=data.blockers,feature_shape=list(data.tensors.features.shape),target_shape=list(data.tensors.target_mm.shape))
            trainer=DiagnosticTrainer(data,config or CONFIG);report.update(config=trainer.config,source=trainer.source,environment=trainer.environment,initial_fit=trainer.fit())
            for step in range(100):
                check();report['trace'].append(trainer.train_step());total+=1
                if trainer.step==50:save_checkpoint(trainer.checkpoint(),output/'step_050.pt')
            report['final_fit']=trainer.fit();report['independence']=independence(trainer)
            final=trainer.checkpoint();save_checkpoint(final,output/'uninterrupted_100.pt')
            restored=DiagnosticTrainer(data,config or CONFIG)
            checkpoint=torch.load(io.BytesIO(read(output/'step_050.pt',16*1024**2)),map_location='cpu',weights_only=True)
            restored.resume(checkpoint);report['resume_trace']=[]
            for step in range(50):check();report['resume_trace'].append(restored.train_step());total+=1
            resumed=restored.checkpoint();save_checkpoint(resumed,output/'resumed_100.pt')
            report['resume_full_state_equal']=equal_state(final,resumed)
            report['resume_fit']=restored.fit();report['loss_decreased']=report['final_fit']['normalized_mse']<report['initial_fit']['normalized_mse']
            report['finite_gradients']=all(math.isfinite(r['gradient_norm_l2']) for r in report['trace']+report['resume_trace'])
            check();report['status']='completed_diagnostic'
            if not report['resume_full_state_equal'] or not all(v for k,v in report['independence'].items() if k.endswith('_equal')):raise ValueError('diagnostic_consistency_failed')
    except BaseException as exc:
        report.update(status='failed' if isinstance(exc,Exception) else 'interrupted',error_type=type(exc).__name__)
        raise
    finally:
        report.update(optimizer_updates_executed=total,elapsed_seconds=time.monotonic()-started)
        (output/'report.json').write_bytes(canonical(report)+b'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate-sha256','rgb-manifest-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args(argv)
    result=run(args.prepared,args.output,diagnostic_only=args.diagnostic_only,candidate_sha256=args.candidate_sha256,rgb_manifest_sha256=args.rgb_manifest_sha256)
    print(json.dumps({k:result[k] for k in ('status','initial_fit','final_fit','resume_full_state_equal','independence','optimizer_updates_executed','elapsed_seconds','train_ready')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

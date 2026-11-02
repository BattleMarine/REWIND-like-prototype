"""고정 합성 RGB→2D→native cache→T1→3D pilot. GPU 실제 실행은 테스트 세션 전용."""
import argparse
from collections import deque
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time

import torch
from PIL import __version__ as pillow_version

from .__main__ import save_report
from .artifacts import load_weights as load_m1_weights
from .data import load_document
from .gpu_pilot_fixture import create_pilot_fixture
from .local_rgb import _no_link
from .models import make_model
from .prediction_join import join_predictions
from .rgb_frontend_artifacts import build_cache, load_weights as load_rgb_weights, preprocessing_version
from .rgb_shards import build_catalog as build_rgb_catalog, open_rgb
from .rgb_training import RGBTrainer
from .rgb_training_cli import export_checkpoint as export_rgb, read_checkpoint as read_rgb
from .shard_cli import export_checkpoint as export_m1, read_checkpoint as read_m1
from .shard_training import ShardTrainer
from .shards import ShardDataset, build_t1_catalog
from .training import ROOT, save_checkpoint, source_identity

MIB=1024**2
GIB=1024**3
LIMITS={'attempts':96,'seconds':600.,'host_rss_bytes':4*GIB,'cuda_allocated_bytes':2*GIB,
        'cuda_reserved_bytes':2*GIB,'worker_artifact_bytes':120*MIB,'total_output_bytes':256*MIB,
        'recovery_zip_bytes':128*MIB,'log_events':64}


class PilotBudgetStop(RuntimeError): pass


def same_state(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(same_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(same_state(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def _disk(output):
    total=0
    for i,path in enumerate(output.rglob('*')):
        if i>=4096: raise PilotBudgetStop('pilot_output_file_count')
        _no_link(path)
        if path.is_file(): total+=path.stat().st_size
    return total


class Budget:
    def __init__(self, output, device, profile):
        import psutil
        self.process=psutil.Process(); self.output=output; self.device=device
        self.started=time.monotonic(); self.attempts=self.successful_updates=self.amp_skips=0
        self.peak_rss=self.peak_allocated=self.peak_reserved=self.peak_disk=0
        self.last={}; self.phase_peaks=[]
        self.max_attempts=96 if profile=='gpu' else 48
        self.max_seconds=600. if profile=='gpu' else 120.

    def sync(self):
        if self.device=='cuda': torch.cuda.synchronize()

    def check(self):
        import psutil
        rss=self.process.memory_info().rss; disk=_disk(self.output)
        allocated=reserved=peak_allocated=peak_reserved=0
        if self.device=='cuda':
            allocated=torch.cuda.memory_allocated(); reserved=torch.cuda.memory_reserved()
            peak_allocated=torch.cuda.max_memory_allocated(); peak_reserved=torch.cuda.max_memory_reserved()
        self.peak_rss=max(self.peak_rss,rss); self.peak_disk=max(self.peak_disk,disk)
        self.peak_allocated=max(self.peak_allocated,peak_allocated,allocated)
        self.peak_reserved=max(self.peak_reserved,peak_reserved,reserved)
        self.last={'host_rss_bytes':rss,'host_available_bytes':psutil.virtual_memory().available,
            'worker_artifact_bytes':disk,'elapsed_seconds':time.monotonic()-self.started,
            'cuda_allocated_bytes':allocated,'cuda_reserved_bytes':reserved,
            'cuda_peak_allocated_bytes':peak_allocated,'cuda_peak_reserved_bytes':peak_reserved}
        for key,actual in (('host_rss_bytes',rss),('worker_artifact_bytes',disk),
                           ('cuda_allocated_bytes',self.peak_allocated),('cuda_reserved_bytes',self.peak_reserved)):
            if actual>LIMITS[key]: raise PilotBudgetStop(key)
        if self.last['elapsed_seconds']>=self.max_seconds: raise PilotBudgetStop('pilot_elapsed_limit')

    def before_attempt(self):
        self.check()
        if self.attempts>=self.max_attempts: raise PilotBudgetStop('pilot_global_attempt_limit')
        self.attempts+=1

    def report(self):
        return {'attempts':self.attempts,'successful_updates':self.successful_updates,'amp_skips':self.amp_skips,
            'observed_peak_host_rss_bytes':self.peak_rss,'cuda_peak_allocated_bytes':self.peak_allocated,
            'cuda_peak_reserved_bytes':self.peak_reserved,'peak_worker_artifact_bytes':self.peak_disk,
            'last_observation':self.last,'host_peak_scope':'cooperative_RSS_samples_only',
            'cuda_peak_scope':'torch_allocator_high_water_not_total_driver_usage'}


def _configs(profile, amp):
    rgb={'seed':101+(1 if amp else 0),'batch_size':2,'learning_rate':.005,'scheduler_step':2,'scheduler_gamma':.8,
         'amp':amp,'training_image_size':[256,256] if profile=='gpu' else [4,4],
         'frontend_version':'synthetic_gpu_pilot_small_cnn_v1','max_native_tensor_bytes':32*MIB,'max_heatmap_elements':32*MIB}
    m1={'model':'single_frame','window_size':1,'hidden':32 if profile=='gpu' else 8,'dropout':.1,'batch_size':2,
        'learning_rate':.005,'seed':201+(1 if amp else 0),'target_scale_mm':100.,'amp':amp,
        'scheduler_step':2,'scheduler_gamma':.8}
    return rgb,m1


def verify_m1_export(state, identity, weights_path, catalog, features, time_valid):
    """AMP는 훈련 옵션이다. CPU 추론 비교에 trainer/AMP scaler를 생성하지 않는다."""
    config=state['config']
    model,_=load_m1_weights(weights_path,expected_contract=catalog['contract'],expected_catalog_sha256=catalog['content_sha256'],
        expected_checkpoint_sha256=identity,expected_source={k:state['source'][k] for k in ('revision','code_sha256')},
        expected_model_config={k:config[k] for k in ('model','window_size','hidden','dropout','target_scale_mm')})
    with torch.random.fork_rng(devices=[]):
        reference=make_model(config,14*len(catalog['contract']['joint_names']),len(catalog['contract']['joint_names']))
    reference.load_state_dict(state['model']); reference.eval()
    with torch.no_grad():
        if not torch.equal(model(features,time_valid),reference(features,time_valid)):
            raise ValueError('pilot_export_prediction_disagreement')
    return True


def run_pilot(output, *, profile, job_id, expected_revision, source_zip_sha256=None):
    run_started=time.monotonic()
    if profile not in ('gpu','cpu_smoke') or not re.fullmatch('[A-Za-z0-9_-]{1,64}',job_id): raise ValueError('pilot_profile_job_id')
    if not re.fullmatch('[0-9a-f]{40}',expected_revision): raise ValueError('pilot_expected_revision_40_required')
    output=Path(output).absolute()
    for path in output.parents: _no_link(path)
    output.mkdir(exist_ok=False)
    effective_limits={**LIMITS,'attempts':96 if profile=='gpu' else 48,'seconds':600. if profile=='gpu' else 120.}
    report={'schema_version':'synthetic_rgb_t1_gpu_pilot_v1','status':'started','job_id':job_id,'profile':profile,
        'device':'cuda' if profile=='gpu' else 'cpu','limits':effective_limits,'phases':[],
        'last_durable':{},'results':{},'full_training_ready':False,'real_data_training_updates':0,
        'gpu_available':False,'gpu_training_attempted':False,'gpu_successful_updates':0,
        'remote_requests':0,'dependency_installation':False,'source_zip_sha256':source_zip_sha256,
        'source_zip_identity_scope':'notebook_verified_bytes_pin_not_runner_reread',
        'unresolved':['real_dataset_release','independent_validation_original_motion_units_time_evidence',
                      'drive_queue_new_vm_persistence','whole_archive_coverage','real_model_accuracy']}
    save_report({'job_id':job_id,'profile':profile,'expected_revision':expected_revision,'source_zip_sha256':source_zip_sha256},output/'started.json')
    modes=[('fp32',False),('amp',True)] if profile=='gpu' else [('fp32',False)]
    save_report({label:{'rgb':_configs(profile,amp)[0],'m1':_configs(profile,amp)[1]} for label,amp in modes},output/'configs.json')
    budget=None; events=deque(maxlen=64)
    def persist():
        if budget is not None: report['resources']=budget.report()
        report['runner_wall_seconds']=time.monotonic()-run_started
        report['resource_clock_scope']='after_source_environment_preflight; supervisor_includes_worker_imports'
        report['events']=list(events)
        save_report(report,output/'report.json')
    def safe_persist():
        try: persist()
        except BaseException as error:
            report['report_write_errors']=(report.get('report_write_errors',[])+[type(error).__name__])[-8:]
    @contextmanager
    def phase(name):
        report['current_phase']=name; report['phases'].append({'name':name,'status':'started'}); persist()
        started=time.monotonic()
        failed=False
        try:
            budget.sync(); budget.check()
            if budget.device=='cuda': torch.cuda.reset_peak_memory_stats()
            yield
            budget.sync(); budget.check()
            report['phases'][-1].update(status='passed',seconds_synchronized=time.monotonic()-started,
                resources=dict(budget.last),timing_scope='phase_including_io_not_model_only')
        except BaseException:
            failed=True
            report['phases'][-1]['status']='failed'
            raise
        finally:
            if failed: safe_persist()
            else: persist()
    def save(trainer,path):
        budget.check(); save_checkpoint(trainer.checkpoint(),path)
        report['last_durable'][str(path.relative_to(output))]={'step':trainer.step,'samples_seen':trainer.samples_seen,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        persist(); budget.check()
    def train_until(trainer,target,path):
        save(trainer,path)
        while trainer.step<target:
            budget.before_attempt()
            if profile=='gpu' and not report['gpu_training_attempted']:
                report['gpu_training_attempted']=True; persist()
            event=trainer.train_step()
            if profile=='gpu': report['gpu_successful_updates']+=int(event['updated'])
            budget.successful_updates+=int(event['updated']); budget.amp_skips+=int(not event['updated'])
            events.append({'phase':report['current_phase'],'step':trainer.step,'updated':event['updated'],
                           'samples_seen':trainer.samples_seen,'batch_samples':event['batch_samples']})
            budget.sync(); budget.check(); save(trainer,path)
    def exercise(kind, config, catalog, base, folder):
        cls,reader=(RGBTrainer,read_rgb) if kind=='rgb' else (ShardTrainer,read_m1)
        folder.mkdir()
        def create(): return cls(config,catalog,base,report['device'],check_budget=budget.check)
        with phase(str(folder.relative_to(output))+'/full'):
            trainer=create(); train_until(trainer,4,folder/'full.pt'); del trainer
            if budget.device=='cuda': torch.cuda.empty_cache()
        with phase(str(folder.relative_to(output))+'/prefix'):
            trainer=create(); train_until(trainer,1,folder/'prefix.pt'); del trainer
            if budget.device=='cuda': torch.cuda.empty_cache()
        with phase(str(folder.relative_to(output))+'/resume'):
            trainer=create(); trainer.resume(reader(folder/'prefix.pt')[0])
            train_until(trainer,4,folder/'resumed.pt')
            full=reader(folder/'full.pt')[0]; resumed=reader(folder/'resumed.pt')[0]
            if not same_state(full,resumed): raise ValueError('pilot_full_checkpoint_state_disagreement')
            del full,resumed
            metrics=trainer.evaluate()
            save_report(metrics,folder/'validation.json')
            del trainer
            if budget.device=='cuda': torch.cuda.empty_cache()
        return {'full_state_equal':True,'successful_updates_full_plus_split':8,'validation':metrics,
                'checkpoint':str((folder/'resumed.pt').relative_to(output))}
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        report['source']=source_identity(); save_report(report['source'],output/'source.json')
        report['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'pillow':pillow_version,
            'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,'device':report['device'],
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
        report['gpu_available']=report['environment']['cuda_available']
        persist(); save_report(report['environment'],output/'environment.json')
        if report['source']['revision']!=expected_revision: raise ValueError('pilot_source_revision_mismatch')
        if profile=='gpu':
            if not re.fullmatch('[0-9a-f]{64}',source_zip_sha256 or '') or not (ROOT/'SOURCE_REVISION.txt').is_file():
                raise ValueError('pilot_packaged_source_pins_required')
            if not torch.cuda.is_available(): raise ValueError('pilot_cuda_required_no_cpu_fallback')
            if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'): raise ValueError('pilot_cuda_workspace')
            props=torch.cuda.get_device_properties(0); free,total=torch.cuda.mem_get_info()
            report['environment'].update(gpu_name=props.name,gpu_total_bytes=props.total_memory,
                gpu_capability=list(torch.cuda.get_device_capability()),driver_free_bytes=free,driver_total_bytes=total)
            if free<2*GIB: raise PilotBudgetStop('pilot_initial_gpu_free_below_2GiB')
        import psutil
        report['environment'].update(psutil=psutil.__version__,host_total_bytes=psutil.virtual_memory().total,
                                     host_available_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(output).free)
        save_report(report['environment'],output/'environment.json')
        if profile=='gpu' and psutil.virtual_memory().available<4*GIB: raise PilotBudgetStop('pilot_initial_host_available_below_4GiB')
        if shutil.disk_usage(output).free<256*MIB: raise PilotBudgetStop('pilot_initial_disk_free_below_256MiB')
        budget=Budget(output,report['device'],profile); persist(); budget.check()
        with phase('fixture'):
            base=output/'fixture'; specs=create_pilot_fixture(base,profile)
            rgb_catalog=build_rgb_catalog(specs,base)
            save_report(rgb_catalog,base/'catalog.json')
        for label,amp in modes:
            mode=output/label; mode.mkdir(); rgb_config,m1_config=_configs(profile,amp)
            report['results'][label]={}
            report['results'][label]['rgb']=exercise('rgb',rgb_config,rgb_catalog,base,mode/'rgb')
            with phase(label+'/rgb_export_cache_join'):
                state,identity=read_rgb(mode/'rgb/resumed.pt')
                export_rgb(state,identity,mode/'rgb/weights.pt')
                model,artifact=load_rgb_weights(mode/'rgb/weights.pt',expected_catalog_sha256=rgb_catalog['content_sha256'],
                    expected_checkpoint_sha256=identity,expected_source={k:state['source'][k] for k in ('revision','code_sha256')},
                    expected_contract=rgb_catalog['contract'],expected_preprocessing_version=preprocessing_version(rgb_config['training_image_size']))
                feature_base=mode/'features'; feature_base.mkdir(); paths=[]
                for i,spec in enumerate(specs):
                    budget.check(); loader=open_rgb(rgb_catalog,base,i)
                    cache=build_cache(loader,model,artifact,check_budget=budget.check)
                    save_report(cache,feature_base/f'cache_{i}.json')
                    pose=json.loads((base/spec['pose']).read_bytes())
                    joined=join_predictions(pose,cache,rgb_catalog['shards'][i]['rgb']['content_sha256'])
                    original=load_document(pose,1)[1]; converted=load_document(joined,1)[1]
                    for split in original:
                        if not torch.equal(original[split].target_mm,converted[split].target_mm) or not torch.equal(original[split].target_valid,converted[split].target_valid):
                            raise ValueError('pilot_join_gt_or_mask_changed')
                    name=f'joined_{i}.json'; save_report(joined,feature_base/name); paths.append(name)
                    del cache,pose,joined,original,converted,loader
                del model,artifact,state
                t1=build_t1_catalog(paths,feature_base); save_report(t1,feature_base/'catalog.json')
                report['results'][label]['native_cache_join']={'shards':3,'gt3d_mask_preserved':True,
                    'inference_precision':'cpu_float32','t1_catalog_sha256':t1['content_sha256']}
            report['results'][label]['m1']=exercise('m1',m1_config,t1,feature_base,mode/'m1')
            with phase(label+'/m1_export'):
                state,identity=read_m1(mode/'m1/resumed.pt')
                export_m1(state,identity,mode/'m1/weights.pt')
                data=ShardDataset(t1,feature_base).load(0)[1]['train']
                verify_m1_export(state,identity,mode/'m1/weights.pt',t1,data.features,data.time_valid)
                del data,state
                report['results'][label]['m1']['export_prediction_equal']=True
        report['status']='passed'; report['gpu_executed']=report['gpu_successful_updates']>0; persist()
    except BaseException as exc:
        report['status']='bounded_stop' if isinstance(exc,PilotBudgetStop) else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        report['failure']={'type':type(exc).__name__,'reason':str(exc)[:512],
                           'cuda_oom':report['device']=='cuda' and isinstance(exc,torch.cuda.OutOfMemoryError)}
        report['gpu_executed']=report['gpu_successful_updates']>0
        report['failure_observation_may_predate_failure']=True
        safe_persist()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--profile',choices=('gpu','cpu_smoke'),required=True)
    parser.add_argument('--job-id',required=True); parser.add_argument('--expected-revision',required=True)
    parser.add_argument('--source-zip-sha256')
    args=parser.parse_args(argv)
    result=run_pilot(args.output,profile=args.profile,job_id=args.job_id,expected_revision=args.expected_revision,
                     source_zip_sha256=args.source_zip_sha256)
    print(json.dumps({k:result.get(k) for k in ('status','job_id','profile','resources','failure')},ensure_ascii=False,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())

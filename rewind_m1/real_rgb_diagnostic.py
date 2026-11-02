"""실제 정적2sample RGB→예측2D cache→3D CPU 진단. 일반 학습 gate는 유지한다."""
import argparse
import copy
import io
import json
import math
import platform
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from PIL import __version__ as pillow_version

from rewind_archive.contracts import canonical,digest
from rewind_data.windows import CausalWindow,FrameSnapshot,FieldSnapshot
from .data import pack_window,TensorSplit,content_hash
from .frontend import StereoHeatmapNet,heatmap_loss,prediction_cache,read_prediction_cache,save_frontend_weights,weights_hash
from .local_rgb import LocalStereoRGB
from .local_diagnostic import _preserve_state
from .real_gt2d_diagnostic import (DiagnosticData,DiagnosticTrainer,CONFIG,load_diagnostic,diagnostic_source_identity,
    read,sha,tensor_sha,equal_state,independence)
from .training import seed_run,rng_state,restore_rng,save_checkpoint

VERSION='small_cnn_real_static_64_v1'
PREPROCESS='rgb_0_1_resize64_bilinear_align_corners_v1'
RESIZE={'mode':'bilinear','align_corners':True,'antialias':False,'network_size':[64,64],
        'native_size':[1024,1024],'gt_to_network_scale':[63/1023,63/1023],
        'prediction_to_native_scale':[1023/63,1023/63]}
FRONT_CONFIG={'seed':71,'steps':100,'batch_size':2,'learning_rate':0.01,'scheduler_step':50,'scheduler_gamma':0.8}
FORMATS={'frontend':'real_rgb_frontend_diagnostic_checkpoint_v1','regression':'real_predicted2d_diagnostic_checkpoint_v1'}


def load_images(prepared,data):
    root=Path(prepared);loader=LocalStereoRGB(root/'rgb/manifest.json',root/'inputs')
    indices={(loader.metadata(i)['record']['sequence_id'],loader.metadata(i)['record']['frame_index']):i for i in range(len(loader))}
    keys=data.tensors.anchors;batch=loader.load_batch([indices[key] for key in keys]);native=batch['images']
    if native.shape!=(2,2,3,1024,1024):raise ValueError('rgb_diagnostic_native_shape')
    images=F.interpolate(native.flatten(0,1),size=(64,64),mode='bilinear',align_corners=True).reshape(2,2,3,64,64)
    raw=read(root/'pose/candidate.json')
    if sha(raw)!=data.identity['candidate_file_sha256']:raise ValueError('rgb_diagnostic_candidate_changed')
    rows={(r['sequence_id'],r['frame_index']):r for r in json.loads(raw)['records']}
    gt=torch.zeros(2,2,62,2);valid=torch.zeros(2,2,62,dtype=torch.bool)
    source_views=[]
    for b,key in enumerate(keys):
        views=batch['metadata'][b]['record']['views']
        source_views.append({'sequence_id':key[0],'frame_index':key[1],
            'views':{side:{k:views[side][k] for k in ('member','sha256','size_bytes','image_size','source_id')} for side in ('left','right')}})
        for v,side in enumerate(('left','right')):
            field=rows[key]['fields']['camera_'+side+'_pts2d']
            valid[b,v]=torch.tensor(field['valid_mask'])
            gt[b,v]=torch.tensor([point if ok else [0.,0.] for point,ok in zip(field['coordinates'],field['valid_mask'])])
    manifest={'schema_version':'real_rgb_diagnostic_input_v1','diagnostic_only':True,'train_ready':False,'data_kind':'dataset_sample',
        'native_rgb_manifest_file_sha256':data.identity['rgb_manifest_file_sha256'],
        'native_rgb_manifest_content_sha256':loader.manifest_sha256,'source_views':source_views,
        'frame_inputs':[item['frame_input'] for item in batch['metadata']],
        'resize':RESIZE,'image_tensor_sha256':tensor_sha(images),'gt2d_tensor_sha256':tensor_sha(gt),
        'annotation_mask_sha256':tensor_sha(valid),'joint_names':data.identity['joint_names']}
    return images,gt,valid,manifest


def checkpoint(trainer,stage):
    return {'format_version':FORMATS[stage],'role':'diagnostic_only_'+stage,'diagnostic_only':True,'train_ready':False,
        'model':trainer.model.state_dict(),'optimizer':trainer.optimizer.state_dict(),'scheduler':trainer.scheduler.state_dict(),
        'scaler':trainer.scaler.state_dict(),'rng':rng_state(),'step':trainer.step,'samples_seen':trainer.step*2,
        'config':trainer.config,'data_identity':trainer.identity,'source':trainer.source,'environment':trainer.environment}


def resume(trainer,state,stage):
    if state.get('format_version')!=FORMATS[stage] or state.get('role')!='diagnostic_only_'+stage or state.get('diagnostic_only') is not True or state.get('train_ready') is not False:
        raise ValueError('rgb_diagnostic_checkpoint_role')
    for key,expected in (('config',trainer.config),('data_identity',trainer.identity),('source',trainer.source),('environment',trainer.environment)):
        if state[key]!=expected:raise ValueError('rgb_diagnostic_resume_identity')
    if type(state['step']) is not int or not 0<=state['step']<=100 or state['samples_seen']!=state['step']*2:raise ValueError('rgb_diagnostic_position')
    for key in ('model','optimizer','scheduler','scaler'):getattr(trainer,key).load_state_dict(state[key])
    trainer.step=state['step'];restore_rng(state['rng'])


class FrontendTrainer:
    def __init__(self,images,gt,valid,identity):
        seed_run(FRONT_CONFIG['seed']);self.config=copy.deepcopy(FRONT_CONFIG);self.identity=identity
        self.images=images;self.gt=gt*torch.tensor(RESIZE['gt_to_network_scale']);self.valid=valid;self.step=0
        self.model=StereoHeatmapNet(62);self.optimizer=torch.optim.Adam(self.model.parameters(),lr=FRONT_CONFIG['learning_rate'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,50,0.8);self.scaler=torch.amp.GradScaler('cuda',enabled=False)
        self.source=diagnostic_source_identity();self.environment={'device':'cpu','python':platform.python_version(),'torch':str(torch.__version__),
            'pillow':pillow_version,'threads':torch.get_num_threads(),'amp':False,'precision':'float32'}

    def fit(self):
        before=self.model.training;self.model.eval()
        with torch.no_grad():loss,inside=heatmap_loss(self.model(self.images)[0],self.gt,self.valid,(64,64))
        self.model.train(before)
        return {'heatmap_mse':float(loss),'inside_joint_views':int(inside.sum()),'annotation_valid_joint_views':int(self.valid.sum()),'evaluation_kind':'train_fit_diagnostic'}

    def train_step(self):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss,inside=heatmap_loss(self.model(self.images)[0],self.gt,self.valid,(64,64));loss.backward()
        gradients=[p.grad for p in self.model.parameters()]
        if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() for g in gradients):raise ValueError('rgb_diagnostic_frontend_nonfinite')
        norm=math.sqrt(sum(float(g.square().sum()) for g in gradients));self.optimizer.step();self.scheduler.step();self.step+=1
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('rgb_diagnostic_nonfinite_weights')
        return {'step':self.step,'loss':float(loss.detach()),'gradient_norm_l2':norm,'inside_joint_views':int(inside.sum())}


class PredictedTrainer(DiagnosticTrainer):
    def __init__(self,data):
        if data.contract['input_source']!='predicted_2d' or data.identity['input_source']!='predicted_2d':raise ValueError('rgb_diagnostic_predicted_only')
        super().__init__(data,CONFIG);self.identity=data.identity

    def checkpoint(self):return checkpoint(self,'regression')
    def resume(self,state):resume(self,state,'regression')


def build_cache(model,images,data,manifest,weights_path):
    small=prediction_cache(model,images,data.tensors.anchors,data.identity['joint_names'],VERSION,PREPROCESS,digest(manifest),'dataset_sample')
    save_frontend_weights(model,small,weights_path)
    native=copy.deepcopy(small);native['image_sizes']=[[1024,1024],[1024,1024]]
    for row in native['records']:
        for side in ('left','right'):
            row['prediction'][side]['coordinates']=[[p[0]*1023/63,p[1]*1023/63] for p in row['prediction'][side]['coordinates']]
    native['content_sha256']=content_hash({k:v for k,v in native.items() if k!='content_sha256'})
    body={'schema_version':'real_rgb_diagnostic_cache_v1','diagnostic_only':True,'train_ready':False,'input_manifest':manifest,
        'resize':RESIZE,'weights_file_sha256':sha(read(weights_path)),'weights_state_sha256':weights_hash(model),
        'network_cache':small,'native_cache':native}
    return {**body,'content_sha256':digest(body)}


def cache_predictions(cache,data,manifest,weights_path,expected_weights_hash):
    body={k:v for k,v in cache.items() if k!='content_sha256'}
    if cache.get('schema_version')!='real_rgb_diagnostic_cache_v1' or cache.get('diagnostic_only') is not True or cache.get('train_ready') is not False or digest(body)!=cache.get('content_sha256'):
        raise ValueError('rgb_diagnostic_cache_schema_hash')
    if cache['input_manifest']!=manifest or cache['resize']!=RESIZE or cache['weights_state_sha256']!=expected_weights_hash or cache['weights_file_sha256']!=sha(read(weights_path)):
        raise ValueError('rgb_diagnostic_cache_provenance')
    weights=torch.load(io.BytesIO(read(weights_path)),map_location='cpu',weights_only=True)
    expected={'format_version':'m1_frontend_weights_v1','weights_only':True,'preprocessing_version':PREPROCESS,'image_sizes':[[64,64],[64,64]],
        'joint_names':data.identity['joint_names'],'data_kind':'dataset_sample','manifest_hash':digest(manifest),
        'frontend_version':VERSION,'frontend_checkpoint_sha256':expected_weights_hash,
        'model_config':{'architecture':'small_cnn_v1','joints':62,'hidden_channels':16,'stride':2}}
    if any(weights.get(k)!=v for k,v in expected.items()):raise ValueError('rgb_diagnostic_weights_contract')
    restored=StereoHeatmapNet(62);restored.load_state_dict(weights['model'])
    if weights_hash(restored)!=expected_weights_hash:raise ValueError('rgb_diagnostic_weights_hash')
    contract=dict(data.contract,input_source='predicted_2d',frontend_version=VERSION,
        frontend_checkpoint_sha256=expected_weights_hash,preprocessing_version=PREPROCESS)
    maps=[]
    for name,size in (('network_cache',64),('native_cache',1024)):
        maps.append(read_prediction_cache(cache[name],SimpleNamespace(**{**contract,'joint_names':tuple(contract['joint_names']),'image_sizes':((size,size),(size,size))}),digest(manifest)))
    if any(set(mapping)!=set(data.tensors.anchors) for mapping in maps):raise ValueError('rgb_diagnostic_cache_keys')
    for key in data.tensors.anchors:
        for side in ('left','right'):
            small=maps[0][key][side];native=maps[1][key][side]
            if small['valid_mask']!=native['valid_mask'] or small['confidence']!=native['confidence'] or small['visibility']!=native['visibility']:raise ValueError('rgb_diagnostic_cache_view_fields')
            for point,original,valid in zip(small['coordinates'],native['coordinates'],small['valid_mask']):
                expected=[point[0]*1023/63,point[1]*1023/63] if valid else None
                if original!=expected:raise ValueError('rgb_diagnostic_cache_transform')
    return maps[1]


def predicted_data(predictions,data,cache):
    if set(predictions)!=set(data.tensors.anchors):raise ValueError('rgb_diagnostic_predictions_required')
    contract={**data.contract,'input_source':'predicted_2d','frontend_version':VERSION,
        'frontend_checkpoint_sha256':cache['weights_state_sha256'],'preprocessing_version':PREPROCESS}
    packed=[];names=tuple(data.identity['joint_names'])
    for i,(sequence,frame) in enumerate(data.tensors.anchors):
        # GT2D field나 candidate feature cache 자체가 없다. Target은 3D/mask만 전달한다.
        field=FieldSnapshot('device_pts3d',tuple(tuple(p.tolist()) if ok else None for p,ok in zip(data.tensors.target_mm[i,0],data.tensors.target_valid[i,0])),tuple(data.tensors.target_valid[i,0].tolist()))
        snapshot=FrameSnapshot(sequence,frame,'diagnostic_3d_target_only',None,(field,),())
        window=CausalWindow(names,('device_pts3d',),(snapshot,),'diagnostic_independent_frame')
        packed.append(pack_window(window,SimpleNamespace(**{**contract,'joint_names':names}),predictions))
    tensors=TensorSplit(*(torch.stack([row[k] for row in packed]) for k in range(4)),list(data.tensors.anchors))
    if not torch.equal(tensors.target_mm,data.tensors.target_mm) or not torch.equal(tensors.target_valid,data.tensors.target_valid):raise ValueError('rgb_diagnostic_target_mask_changed')
    identity={'schema':'real_rgb_predicted2d_diagnostic_data_v1','data_kind':'dataset_sample','input_source':'predicted_2d',
        'target_unit':'mm','diagnostic_only':True,'train_ready':False,'cache_content_sha256':cache['content_sha256'],
        'weights_state_sha256':cache['weights_state_sha256'],'weights_file_sha256':cache['weights_file_sha256'],'resize':RESIZE,
        'prepared_data_identity_sha256':digest(data.identity),'joint_names':list(names),'sample_keys':data.tensors.anchors,
        'features_sha256':tensor_sha(tensors.features),'target_sha256':tensor_sha(tensors.target_mm),'mask_sha256':tensor_sha(tensors.target_valid)}
    return DiagnosticData(tensors,identity,contract,data.blockers)


def errors_2d(predictions,data,gt,valid):
    xy=torch.tensor([[predictions[key][side]['coordinates'] for side in ('left','right')] for key in data.tensors.anchors])
    inside=valid&(gt[...,0]>=0)&(gt[...,0]<=1023)&(gt[...,1]>=0)&(gt[...,1]<=1023)
    errors=torch.linalg.vector_norm(xy-gt,dim=-1)
    def mean(mask):return float(errors[mask].mean()) if mask.any() else None
    return {'evaluation_kind':'train_fit_diagnostic','coordinate_unit':'native_pixel',
        'all_valid_count':int(valid.sum()),'all_valid_mean_px':mean(valid),
        'inside_count':int(inside.sum()),'inside_mean_px':mean(inside),
        'outside_count':int((valid&~inside).sum()),'outside_mean_px':mean(valid&~inside)}


def run(prepared,output,*,diagnostic_only,candidate_sha256,rgb_manifest_sha256):
    if diagnostic_only is not True:raise ValueError('explicit_diagnostic_only_required')
    output=Path(output).absolute();prepared=Path(prepared).absolute()
    if output.exists() or output.is_relative_to(prepared) or prepared.is_relative_to(output):raise ValueError('rgb_diagnostic_new_separate_output')
    output.mkdir(parents=True);started=time.monotonic();updates={'frontend':0,'regression':0}
    report={'schema_version':'real_rgb_predicted2d_diagnostic_v1','status':'started','diagnostic_only':True,'train_ready':False,
        'data_kind':'dataset_sample','input_source':'predicted_2d','evaluation_kind':'train_fit_diagnostic',
        'validation_or_test_run':False,'temporal_experiment':False,'limits':{'frontend_updates':100,'regression_updates':100,'seconds':300},
        'frontend_trace':[],'regression_trace':[],'uninterrupted_resume_equivalence_tested':False}
    def check():
        if time.monotonic()-started>300 or any(n>100 for n in updates.values()):raise ValueError('rgb_diagnostic_budget')
    def save_json(name,doc):(output/name).write_bytes(canonical(doc)+b'\n')
    try:
        with _preserve_state():
            data=load_diagnostic(prepared,diagnostic_only=True,candidate_sha256=candidate_sha256,rgb_manifest_sha256=rgb_manifest_sha256)
            images,gt,valid,manifest=load_images(prepared,data);save_json('input_manifest.json',manifest)
            report.update(prepared_data_identity=data.identity,resize=RESIZE,input_manifest_sha256=digest(manifest),blockers=data.blockers)
            front=FrontendTrainer(images,gt,valid,manifest);report.update(source=front.source,environment=front.environment,frontend_config=front.config,initial_frontend_fit=front.fit())
            for i in range(100):
                check();report['frontend_trace'].append(front.train_step());updates['frontend']+=1
                if front.step==50:
                    state=checkpoint(front,'frontend');save_checkpoint(state,output/'frontend_050.pt')
                    restored=FrontendTrainer(images,gt,valid,manifest);resume(restored,torch.load(io.BytesIO(read(output/'frontend_050.pt')),map_location='cpu',weights_only=True),'frontend')
                    report['frontend_resume_loaded_state_equal']=equal_state(state,checkpoint(restored,'frontend'));front=restored
            report['final_frontend_fit']=front.fit();save_checkpoint(checkpoint(front,'frontend'),output/'frontend_100.pt')
            cache=build_cache(front.model,images,data,manifest,output/'frontend_weights.pt');save_json('prediction_cache.json',cache)
            predictions=cache_predictions(json.loads(read(output/'prediction_cache.json')),data,manifest,output/'frontend_weights.pt',weights_hash(front.model))
            pred_data=predicted_data(predictions,data,cache)
            report['cache_reload_features_equal']=torch.equal(pred_data.tensors.features,predicted_data(cache_predictions(cache,data,manifest,output/'frontend_weights.pt',weights_hash(front.model)),data,cache).tensors.features)
            changed=copy.deepcopy(data);changed.tensors.features+=999
            report['gt2d_feature_change_has_no_effect']=torch.equal(pred_data.tensors.features,predicted_data(predictions,changed,cache).tensors.features)
            report['train_fit_2d']=errors_2d(predictions,data,gt,valid);report['predicted_data_identity']=pred_data.identity
            regression=PredictedTrainer(pred_data);report.update(regression_config=regression.config,initial_regression_fit=regression.fit())
            for i in range(100):
                check();report['regression_trace'].append(regression.train_step());updates['regression']+=1
                if regression.step==50:
                    state=regression.checkpoint();save_checkpoint(state,output/'regression_050.pt')
                    restored=PredictedTrainer(pred_data);restored.resume(torch.load(io.BytesIO(read(output/'regression_050.pt')),map_location='cpu',weights_only=True))
                    report['regression_resume_loaded_state_equal']=equal_state(state,restored.checkpoint());regression=restored
            report['final_regression_fit']=regression.fit();report['independence']=independence(regression)
            save_checkpoint(regression.checkpoint(),output/'regression_100.pt')
            report['frontend_loss_decreased']=report['final_frontend_fit']['heatmap_mse']<report['initial_frontend_fit']['heatmap_mse']
            report['regression_loss_decreased']=report['final_regression_fit']['normalized_mse']<report['initial_regression_fit']['normalized_mse']
            report['finite_gradients']=all(math.isfinite(row['gradient_norm_l2']) for row in report['frontend_trace']+report['regression_trace'])
            if not all(report[k] for k in ('cache_reload_features_equal','gt2d_feature_change_has_no_effect','frontend_resume_loaded_state_equal','regression_resume_loaded_state_equal')):raise ValueError('rgb_diagnostic_consistency')
            if not all(v for k,v in report['independence'].items() if k.endswith('_equal')):raise ValueError('rgb_diagnostic_independence')
            check();report['status']='completed_diagnostic'
    except BaseException as exc:
        report.update(status='failed' if isinstance(exc,Exception) else 'interrupted',error_type=type(exc).__name__)
        raise
    finally:
        report.update(updates_executed=updates,elapsed_seconds=time.monotonic()-started);save_json('report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate-sha256','rgb-manifest-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--diagnostic-only',action='store_true');args=parser.parse_args(argv)
    result=run(args.prepared,args.output,diagnostic_only=args.diagnostic_only,candidate_sha256=args.candidate_sha256,rgb_manifest_sha256=args.rgb_manifest_sha256)
    print(json.dumps({k:result[k] for k in ('status','initial_frontend_fit','final_frontend_fit','train_fit_2d','initial_regression_fit','final_regression_fit','updates_executed','elapsed_seconds','train_ready')},indent=2))


if __name__=='__main__':main()

"""Same six regression clips, same fixed frontend, arm-local camera-shift diagnostics."""

import io
import json
from pathlib import Path

import torch

from rewind_archive.contracts import digest
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal,check_pin
from rewind_m1.local_rgb import _regular_read
from . import camera_training as train,camera_training_data as data,camera_gauge as gauge,cross_data as cross,cross_inference as infer
from .frontend import predicted_features
from .data import Clip


def prepare_eval(inputs,references,model,rgb_root,pose_root,output,device):
    output=Path(output);output.mkdir(exist_ok=True);budget=cross.Budget(gauge.LIMITS['prepare_seconds'])
    plan=gauge.fixed_plan(inputs,references,rgb_root,pose_root,output/'selection')
    prepared=cross.prepare(inputs,output/'selection/selection.json',plan['content_sha256'],output/'prepared',context=gauge.context(),budget=budget)
    if (output/'eval_features.json').exists():
        existing=json.loads((output/'eval_features.json').read_bytes());check_pin(existing,'v15_eval_features_v1',existing['content_sha256'])
        if existing['source_sha256']!=digest(train.source_identity()):raise ValueError('v15_restored_eval_feature_source')
        for old,new in zip(existing['clips'],prepared['clips'],strict=True):
            if old['entry']['sequence']!=new['sequence'] or old['entry']['frames']!=new['frames']:
                raise ValueError('v15_restored_eval_raw_mismatch')
        return existing
    _,spec,_,_=cross.assets(inputs)
    models,pins,_,_=infer.view._models(model,spec,False,training_sha256=cross.MODEL_SHA,run_id=infer.view.FULL_RUN_ID)
    frontend=models['frontend'].to(device);records=[]
    for number,(entry,ref) in enumerate(zip(prepared['clips'],references['clips'],strict=True)):
        budget.check();root=output/'prepared'/f'clip-{number:02d}';p=entry['tensor'];raw=_regular_read(root/'clip.pt',32*cross.MIB,p['bytes'])
        if sha(raw)!=p['sha256']:raise ValueError('v15_eval_prepared_pin')
        tensors=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        with torch.inference_mode():features=torch.cat([predicted_features(frontend,v.to(device)).cpu() for v in tensors.pop('images').split(2)])
        tensors['features']=features
        meta={'sequence':entry['sequence'],'start':entry['frames'][0]['frame'],'frames':[r['frame'] for r in entry['frames']],
            'rotation':[r['rotation_matrix'] for r in ref['camera']['frames']],
            'translation_mm':[r['world_to_camera_translation_mm'] for r in ref['camera']['frames']]}
        absolute,_=train.CameraOrigin('absolute').apply(meta['rotation'],meta['translation_mm'],meta['sequence'],meta['start'],meta['frames'])
        if not torch.equal(absolute,tensors['camera']):raise ValueError('v15_eval_original_camera_mismatch')
        pin=data._save_tensor(output/f'eval-{number:02d}.pt',tensors,4*cross.MIB)
        records.append({'metadata':meta,'entry':entry,'tensor':pin})
    doc=seal({'schema':'v15_eval_features_v1','run_id':train.RUN_ID,'source_sha256':digest(train.source_identity()),
        'frontend':pins['frontend'],'clips':records,'source_bytes_read':budget.read_bytes,'spec':spec,
        'scope':'Six fixed regression/development clips, not independent generalization'})
    cross.write_report(output/'eval_features.json',doc);return doc


def centroid_metrics(body,tensors,spec):
    idx=[spec['joint_names'].index(n) for n in spec['body_joint_names']]
    gt=tensors['target'][:,idx];valid=tensors['target_valid'][:,idx];bias=[];centered_sum=0.;count=0
    for p,g,m in zip(body,gt,valid,strict=True):
        if not m.any():continue
        pc,gc=p[m].double().mean(0),g[m].double().mean(0);bias.append(float((pc-gc).norm()))
        centered_sum+=float(((p[m].double()-pc)-(g[m].double()-gc)).norm(dim=-1).sum());count+=int(m.sum())
    return {'body_centroid_bias_mean_mm':sum(bias)/len(bias) if bias else None,'centroid_valid_frames':len(bias),
        'centered_body_error_sum_mm':centered_sum,'centered_body_valid_joints':count,
        'centered_body_error_mean_mm':centered_sum/count if count else None}


def train_fit(experiment):
    key=experiment.keys[0];clip=experiment.clips[key];metadata=experiment.metadata[key];result={}
    indices=[experiment.spec['joint_names'].index(n) for n in experiment.spec['body_joint_names']]
    fingers,wrists=infer._hand_indices(experiment.spec['joint_names'],experiment.spec)
    targets={'body':clip.target_mm[0,:,indices],'hand_global':clip.target_mm[0,:,fingers],
             'hand_local':clip.target_mm[0,:,fingers]-clip.target_mm[0,:,wrists]}
    valid={'body':clip.target_valid[0,:,indices],'hand_global':clip.target_valid[0,:,fingers],
           'hand_local':clip.target_valid[0,:,fingers]&clip.target_valid[0,:,wrists]}
    for arm in train.ARMS:
        camera,_=train.camera_clip(clip,metadata,arm);models={r:experiment.stages[arm][r].model.eval().requires_grad_(False) for r in ('body','hand')};models['spec']=experiment.spec
        prediction=infer.sample_features(models,clip.features[0],camera.camera[0],[(key[0],f) for f in metadata['frames']],experiment.cfg,experiment.device)
        metrics={r:{'sum_mm':float((prediction[r]-targets[r]).norm(dim=-1)[valid[r]].double().sum()),'count':int(valid[r].sum()),
            'mean_mm':float((prediction[r]-targets[r]).norm(dim=-1)[valid[r]].double().mean())} for r in targets}
        shape_clip=clip.select_joints(indices)
        edges=[(experiment.spec['body_joint_names'].index(a),experiment.spec['body_joint_names'].index(b)) for a,b in experiment.spec['edges'] if a in experiment.spec['body_joint_names'] and b in experiment.spec['body_joint_names']]
        metrics['body_shape_sums']=infer.CascadeStages._shape(prediction['body'][None],shape_clip,edges)
        fingers_names=[n for side in ('left','right') for n in experiment.spec['hands'][side][1:]]
        hand_edges=[(fingers_names.index(a),fingers_names.index(b)) for a,b in experiment.spec['edges'] if a in fingers_names and b in fingers_names]
        from .data import hand_clip
        metrics['hand_local_shape_sums']=infer.CascadeStages._shape(prediction['hand_local'][None],hand_clip(clip,experiment.spec),hand_edges)
        result[arm]=metrics
    return {'key':list(key),'scope':'first fixed train clip final-fit diagnostic only; not merged into regression and never used for best selection','arms':result}


def evaluate(experiment,eval_root,output,*,expected_pin):
    if experiment.position['role']!='complete':raise ValueError('v15_requires_final_fixed_models')
    eval_root,output=Path(eval_root),Path(output);output.mkdir();budget=cross.Budget(train.POLICY['limits']['eval_seconds'])
    doc=json.loads(_regular_read(eval_root/'eval_features.json',16*cross.MIB));check_pin(doc,'v15_eval_features_v1',expected_pin)
    if doc['source_sha256']!=digest(train.source_identity()) or len(doc['clips'])!=6:raise ValueError('v15_eval_binding')
    summaries=[]
    for number,row in enumerate(doc['clips']):
        budget.check();p=row['tensor'];raw=_regular_read(eval_root/p['name'],4*cross.MIB,p['bytes'])
        if sha(raw)!=p['sha256']:raise ValueError('v15_eval_tensor_pin')
        tensors=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu');features=tensors.pop('features');metadata=row['metadata']
        clip=Clip(features[None],tensors['camera'][None],tensors['target'][None],tensors['target_valid'][None],
            torch.ones(1,50,dtype=torch.bool),torch.tensor(metadata['frames'],dtype=torch.float64)[None]/25,
            'predicted_2d',split='validation',input_sha256='0'*64)
        keys=[(metadata['sequence'],f) for f in metadata['frames']];predictions={};results={}
        folder=output/f'clip-{number:02d}';folder.mkdir()
        for arm in train.ARMS:
            models={r:experiment.stages[arm][r].model.eval().requires_grad_(False) for r in ('body','hand')};models['spec']=experiment.spec
            base,audit=train.camera_clip(clip,metadata,arm)
            # First chunk establishes its anchor; continuation restores that exact state.
            origin=train.CameraOrigin(arm);camera0,_=origin.apply(metadata['rotation'][:25],metadata['translation_mm'][:25],metadata['sequence'],metadata['start'],metadata['frames'][:25])
            continuation=train.CameraOrigin(arm,origin.state)
            camera1,_=continuation.apply(metadata['rotation'][25:],metadata['translation_mm'][25:],metadata['sequence'],metadata['start'],metadata['frames'][25:])
            if not torch.equal(torch.cat((camera0,camera1)),base.camera[0]):raise ValueError('v15_streaming_camera_mismatch')
            prediction=infer.sample_features(models,features,base.camera[0],keys,experiment.cfg,experiment.device)
            repeat=infer.sample_features(models,features,base.camera[0],keys,experiment.cfg,experiment.device)
            changed,changed_audit=train.camera_clip(clip,train.shifted_metadata(metadata,[100000,0,0]),arm)
            shifted=infer.sample_features(models,features,changed.camera[0],keys,experiment.cfg,experiment.device)
            prefix=infer.sample_features(models,features[:25],camera0,keys[:25],experiment.cfg,experiment.device)
            if any(not torch.allclose(prediction[k][:25],prefix[k],atol=.01,rtol=1e-5) for k in prediction):raise ValueError('v15_prediction_prefix_mismatch')
            stability=gauge.repeat_stability(prediction,repeat);paired=gauge.paired_metrics(shifted,prediction,repeat)
            shift_status=('inconclusive_repeat_unstable' if stability['status']!='stable_within_fixed_tolerance' else
                'exceeds_repeat_aware_tolerance' if any(v['coordinates_above_repeat_aware_tolerance'] for v in paired.values()) else 'within_repeat_aware_tolerance')
            variants={}
            for name,pred in (('baseline',prediction),('zero_repeat',repeat),('world_x_plus_100m',shifted)):
                metrics,xy,_=infer.metric_sums(pred,features,tensors,experiment.spec)
                variants[name]={'prediction':{k:v.tolist() for k,v in pred.items()},'aggregate':infer.aggregate(metrics),'frame_metrics':metrics,
                    **centroid_metrics(pred['body'],tensors,experiment.spec)}
            payload=seal({'schema':'v15_camera_eval_arm_v1','run_id':train.RUN_ID,'source_sha256':digest(train.source_identity()),
                'arm':arm,'keys':keys,'variants':variants,'paired_shift':paired,'repeat_stability':stability,
                'paired_shift_status':shift_status,
                'camera_input_shift_max_abs':float((base.camera-changed.camera).abs().max()),
                'normalized_input_within_tolerance':bool(torch.allclose(base.camera,changed.camera,atol=1e-5,rtol=1e-6)),
                'anchor':audit,'shifted_recomputed_anchor':changed_audit,'stream_camera_exact':True,'prefix25_within_tolerance':True,
                'gt_target_device_mm':tensors['target'].tolist(),'gt_target_valid':tensors['target_valid'].tolist(),
                'features_sha256':sha(features.numpy().tobytes()),'scope':'development regression; fixed final1000 updates, not best selection'})
            cross.write_report(folder/(arm+'.json'),payload);results[arm]=payload
            predictions[arm]=prediction;predictions[arm+'_shift']=shifted
        render_predictions={'baseline':predictions['absolute'],'first_midpoint':predictions['first_midpoint'],
                            'world_x_plus_100m':predictions['first_midpoint_shift']}
        used=sum(p.stat().st_size for p in experiment.root.iterdir() if p.is_file())+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        available=train.POLICY['limits']['output_bytes']-used-2*train.POLICY['limits']['feature_bytes']-16*cross.MIB
        if available<=0:raise ValueError('v15_eval_output_budget')
        artifacts=gauge.render_comparison(folder,eval_root/'prepared'/f'clip-{number:02d}',row['entry'],render_predictions,xy,tensors,
            experiment.spec,budget,available,
            labels=('GT','absolute','first_midpoint','first_midpoint +100m'),title='v1.5 camera training')
        summary=seal({'schema':'v15_camera_eval_clip_v1','run_id':train.RUN_ID,'number':number,'sequence':metadata['sequence'],
            'arms':{a:{'content_sha256':r['content_sha256'],'paired_shift':r['paired_shift'],'repeat_stability':r['repeat_stability'],
                'aggregate':r['variants']['baseline']['aggregate'],'centroid':{k:r['variants']['baseline'][k] for k in ('body_centroid_bias_mean_mm','centered_body_error_mean_mm')},
                'paired_shift_status':r['paired_shift_status'],'normalized_input_within_tolerance':r['normalized_input_within_tolerance']} for a,r in results.items()},
            'artifacts':artifacts,'render_bounds_device_mm':gauge.render_bounds(tensors,render_predictions)})
        cross.write_report(folder/'report.json',summary);summaries.append(summary)
    report=seal({'schema':'v15_camera_eval_report_v1','run_id':train.RUN_ID,'source_sha256':digest(train.source_identity()),
            'checkpoint_sha256':experiment.last['sha256'],'feature_receipt_sha256':doc['content_sha256'],'status':'complete','clips':summaries,
        'train_fit':train_fit(experiment),
        'scope':'fixed six regression/development clips; F9 pretrained20888, B/H1000, within-experiment arms only'})
    cross.write_report(output/'report.json',report);return report

"""Fixed world-origin sensitivity experiment; no training or GT-based gauge selection."""

from collections import defaultdict
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read,_no_link
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha,official_splits
from rewind_prepare.extracted import _local,_payload_path,_Budget,ExtractedLimits
from rewind_prepare.temporal_selection import seal,check_pin
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from . import camera_control as control,cross_data as cross,cross_inference as inference


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/v12_camera_gauge_001.json'
POLICY=json.loads(CONFIG.read_bytes());LIMITS=POLICY['limits'];RUN_ID=POLICY['run_id']
VARIANTS=tuple(POLICY['variants']);TOL=POLICY['comparison_tolerance']


def source_identity():
    names=('configs/v12_camera_gauge_001.json','scripts/build_v12_camera_gauge.py')
    return {**control.source_identity(),**{n:sha((ROOT/n).read_bytes()) for n in names}}


def context():return cross.EvaluationContext(RUN_ID,POLICY['environments'],source_identity)


def _json(root,name,schema=None):
    doc=json.loads(_regular_read(root/name,32*cross.MIB))
    if schema:check_pin(doc,schema,doc.get('content_sha256'))
    return doc


def _receipt(root,run_id):
    receipt=_json(root,'recovery_receipt.json')
    if receipt['schema']!='v12_cross_result_recovery_v1' or receipt['run_id']!=run_id:
        raise ValueError('gauge_reference_receipt')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(receipt['files'])|{'recovery_receipt.json'}:raise ValueError('gauge_reference_members')
    for name,pin in receipt['files'].items():
        p=_local(root/name)
        if not p.is_relative_to(root):raise ValueError('gauge_reference_path')
        raw=_regular_read(p,LIMITS['reference_unpacked_bytes'],pin['bytes'])
        if sha(raw)!=pin['sha256']:raise ValueError('gauge_reference_hash')
        if name.endswith('.json'):
            d=json.loads(raw)
            if 'content_sha256' in d:check_pin(d,d['schema'],d['content_sha256'])
    return receipt


def checked_references(root):
    """Both ZIPs must first be pinned by load_references; stage receipt pins prevent substitution."""
    root=_local(root,directory=True);city=control.checked_reference(root/'city');a=root/'control'
    _receipt(a,control.RUN_ID)
    plan=_json(a,'selection/selection.json','v12_cross_selection_v1')
    prepared=_json(a,'prepared/prepared.json','v12_cross_prepared_v1')
    report=_json(a,'inference/report.json','v12_cross_report_v1')
    cameras=_json(a,'cameras/cameras.json','v12_camera_records_v1')
    comparison=_json(a,'comparison/report.json','v12_camera_comparison_v1')
    for d in (plan,prepared,report,cameras,comparison):
        if (d['run_id']!=control.RUN_ID or d['source_sha256']!=POLICY['references']['control']['source_sha256']
                or d['model_zip_sha256']!=cross.MODEL_SHA):raise ValueError('gauge_control_source_model')
    if (plan['status']!='complete' or len(plan['clips'])!=2 or len(prepared['clips'])!=2 or len(report['clips'])!=2
            or prepared['selection_sha256']!=plan['content_sha256'] or report['selection_sha256']!=plan['content_sha256']
            or report['prepared_sha256']!=prepared['content_sha256'] or cameras['prepared_sha256']!=prepared['content_sha256']
            or comparison['cameras_sha256']!=cameras['content_sha256'] or comparison['inference_report_sha256']!=report['content_sha256']
            or comparison['reference_zip_sha256']!=POLICY['references']['city']['sha256']
            or cameras['reference_report_sha256']!=city['report']['content_sha256']
            or plan['binding']!=prepared['binding'] or cameras['binding']!=prepared['binding']
            or prepared['binding']!=city['prepared']['binding']
            or report['model_weights']!=city['report']['model_weights']
            or report['inference_implementation_sha256']!=city['report']['inference_implementation_sha256']
            or len(cameras['clips'])!=6):raise ValueError('gauge_control_chain')
    clips=[]
    for i,(s,p,r) in enumerate(zip(plan['clips'],prepared['clips'],report['clips'],strict=True)):
        pred=_json(a,f'inference/clip-{i:02d}/predictions.json','v12_cross_clip_predictions_v1')
        manifest=_json(a,f'prepared/clip-{i:02d}/manifest.json')
        keys=[[s['sequence'],f['frame']] for f in s['frames']]
        if (s['environment']!='ArchVisInterior_ArchVis_RT' or s['split']!='validation' or len(keys)!=50
                or [k[1] for k in keys]!=list(range(keys[0][1],keys[0][1]+50))
                or manifest!=p or p['number']!=i or any(d['sequence']!=s['sequence'] for d in (p,r,pred))
                or pred['keys']!=keys or [f['frame'] for f in p['frames']]!=[k[1] for k in keys]
                or pred['content_sha256']!=r['content_sha256'] or pred['run_id']!=control.RUN_ID):
            raise ValueError('gauge_control_order_keys')
        clips.append({'selection':s,'prepared':p,'prediction':pred,'origin':'control','origin_number':i})
    clips += [{**c,'origin':'city','origin_number':i} for i,c in enumerate(city['clips'])]
    if len({c['selection']['sequence'] for c in clips})!=6:raise ValueError('gauge_six_distinct_clips')
    for number,(c,cam) in enumerate(zip(clips,cameras['clips'],strict=True)):
        s,p=c['selection'],c['prepared'];keys=[[s['sequence'],f['frame']] for f in s['frames']]
        if (cam['sequence']!=s['sequence'] or cam['environment']!=s['environment']
                or [r['key'] for r in cam['frames']]!=keys or cam['group']!=('archvis' if number<2 else 'city')
                or cam['number']!=c['origin_number']):raise ValueError('gauge_camera_reference_keys')
        for j,(x,y,z) in enumerate(zip(s['frames'],p['frames'],cam['frames'],strict=True)):
            if set(x['files'])!={'left','right','pose'} or set(y['files'])!={'left','right','pose'}:
                raise ValueError('gauge_reference_roles')
            for role in ('left','right','pose'):
                item,observed=x['files'][role],y['files'][role]
                if (cross.identify(item['path'],'pose' if role=='pose' else 'rgb')!=(s['sequence'],keys[j][1],role)
                        or item['size_bytes']!=observed['size_bytes'] or not 0<item['size_bytes']<=8*cross.MIB):
                    raise ValueError('gauge_reference_file_key_size')
            if z['pose']!=y['files']['pose']:raise ValueError('gauge_reference_pose_pin')
            transform_gauge([z],[0.,0.,0.])  # finite/proper rotation and encoded reference consistency
        c['camera']=cam
    return {'clips':clips,'binding':prepared['binding'],'model_weights':report['model_weights'],
        'reference_environments':{name:{k:r[k] for k in ('device','device_name','torch','precision','threads')} for name,r in (('city',city['report']),('control',report))},
        'implementation':report['inference_implementation_sha256'],'reference_report_sha256':{
            'city':city['report']['content_sha256'],'control':report['content_sha256']}}


def load_references(city_zip,control_zip,output):
    output=Path(output);output.mkdir()
    for name,archive in (('city',city_zip),('control',control_zip)):
        pin=POLICY['references'][name];archive=_local(archive)
        if archive.stat().st_size!=pin['bytes']:raise ValueError('gauge_reference_zip_size')
        extract_pinned_zip(archive,pin['sha256'],output/name,max_zip_bytes=LIMITS['reference_zip_bytes'],
            max_output_bytes=LIMITS['reference_unpacked_bytes'],max_members=LIMITS['reference_members'])
    return checked_references(output)


def fixed_plan(inputs,references,rgb_root,pose_root,output):
    """Stat only the 900 fixed paths; never list directories, replace clips or choose frames."""
    budget=cross.Budget(LIMITS['metadata_seconds'],LIMITS['metadata_bytes'])
    _,spec,_,binding=cross.assets(inputs)
    if binding!=references['binding']:raise ValueError('gauge_input_binding')
    owners,_=official_splits(Path(inputs)/'split',spec,_Budget(ExtractedLimits(8*cross.MIB,60)))
    roots={'rgb':_local(rgb_root,directory=True),'pose':_local(pose_root,directory=True)};clips=[]
    output=Path(output);output.mkdir()
    try:
        for number,ref in enumerate(references['clips']):
            clip=ref['selection'];rows=[]
            if owners.get(clip['sequence'])!='validation':raise ValueError('gauge_validation_only')
            for chosen,observed in zip(clip['frames'],ref['prepared']['frames'],strict=True):
                files={}
                for role,item in chosen['files'].items():
                    budget.entry(len(item['path'].encode())+128)
                    info=_no_link(_payload_path(roots['pose' if role=='pose' else 'rgb'],item['path']))
                    pin=observed['files'][role]
                    if info.st_size!=pin['size_bytes']:raise ValueError('gauge_fixed_file_size_no_replacement')
                    files[role]={**item,'mtime_ns':info.st_mtime_ns,'sha256':pin['sha256']}
                rows.append({'frame':chosen['frame'],'files':files})
            clips.append({**clip,'frames':rows,'origin':ref['origin'],'origin_number':ref['origin_number'],
                          'reference_prediction_sha256':ref['prediction']['content_sha256']})
        result=seal({'schema':'v12_cross_selection_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
            'status':'complete','model_zip_sha256':cross.MODEL_SHA,'binding':binding,'references':POLICY['references'],
            'reference_report_sha256':references['reference_report_sha256'],'roots':{k:str(v) for k,v in roots.items()},
            'clips':clips,'policy':'Exact frozen six clips; no reselection or fallback','payload_bytes_read':0,
            'metadata_entries':budget.entries,'metadata_bytes_upper':budget.metadata_bytes,'elapsed_seconds':time.monotonic()-budget.start})
        cross.write_report(output/'selection.json',result);return result
    except BaseException as error:
        cross.write_report(output/'failure.json',{'state':'failed','reason':str(error)[:512],'type':type(error).__name__});raise


def checked_plan(path,pin,references):
    plan=cross.checked_selection(path,pin,context=context())
    if plan['references']!=POLICY['references'] or plan['binding']!=references['binding'] or len(plan['clips'])!=6:
        raise ValueError('gauge_plan_reference_binding')
    for c,r in zip(plan['clips'],references['clips'],strict=True):
        if (c['sequence']!=r['selection']['sequence'] or c['reference_prediction_sha256']!=r['prediction']['content_sha256']
                or [x['frame'] for x in c['frames']]!=[x['frame'] for x in r['selection']['frames']]):
            raise ValueError('gauge_plan_order_keys')
        for a,b,z in zip(c['frames'],r['selection']['frames'],r['prepared']['frames'],strict=True):
            for role in ('left','right','pose'):
                if (a['files'][role]['path']!=b['files'][role]['path'] or a['files'][role]['size_bytes']!=z['files'][role]['size_bytes']
                        or a['files'][role]['sha256']!=z['files'][role]['sha256']):raise ValueError('gauge_plan_file_pin')
    return plan


def verify_prepared(path,pin,plan,references):
    doc=cross.read_sealed(Path(path)/'prepared.json','v12_cross_prepared_v1',pin,context=context())
    if (doc['selection_sha256']!=plan['content_sha256'] or doc['binding']!=plan['binding']
            or doc['model_zip_sha256']!=cross.MODEL_SHA or len(doc['clips'])!=6):raise ValueError('gauge_prepared_binding')
    for number,(entry,ref) in enumerate(zip(doc['clips'],references['clips'],strict=True)):
        if entry['number']!=number or entry['sequence']!=ref['selection']['sequence']:
            raise ValueError('gauge_prepared_order')
        for a,b in zip(entry['frames'],ref['prepared']['frames'],strict=True):
            if a['frame']!=b['frame'] or a['files']!=b['files']:raise ValueError('gauge_prepared_files')
    return doc


def camera_delta(rows,variant):
    if variant in ('baseline','zero_repeat'):return np.zeros(3)
    if variant=='world_x_plus_100m':return np.array([100000.,0.,0.])
    if variant!='first_midpoint':raise ValueError('gauge_variant_not_authorized')
    first=rows[0];r=np.asarray(first['rotation_matrix']);t=np.asarray(first['world_to_camera_translation_mm'])
    return -np.stack([-ri.T@ti for ri,ti in zip(r,t,strict=True)]).mean(0)


def transform_gauge(rows,delta):
    """Common canonical-world translation only: x'=x+delta, t'=t-R@delta."""
    delta=np.asarray(delta,dtype=np.float64)
    if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('gauge_delta_shape_finite')
    encoded=[];transformed=[];residuals=defaultdict(list)
    probe=np.array([137.,-29.,802.])  # fixed algebra probe, no GT selection
    for row in rows:
        r=np.asarray(row['rotation_matrix'],dtype=np.float64);t=np.asarray(row['world_to_camera_translation_mm'],dtype=np.float64)
        if r.shape!=(2,3,3) or t.shape!=(2,3) or not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError('gauge_camera_shape_finite')
        views={s:{'rotation_matrix':r[i].tolist(),'translation':t[i].tolist()} for i,s in enumerate(('left','right'))}
        original=encode_camera(views,100.)
        if not torch.equal(original.flatten(),torch.tensor(row['encoded_camera_18'],dtype=torch.float32)):
            raise ValueError('gauge_reference_encoded_camera')
        shifted=t-np.einsum('vij,j->vi',r,delta)
        new=encode_camera({s:{'rotation_matrix':r[i].tolist(),'translation':shifted[i].tolist()} for i,s in enumerate(('left','right'))},100.)
        if not torch.equal(new[:,:6],original[:,:6]):raise ValueError('gauge_rotation_changed')
        projection=np.einsum('vij,j->vi',r,probe)+t
        shifted_projection=np.einsum('vij,j->vi',r,probe+delta)+shifted
        q=r[1]@r[0].T
        stereo=t[1]-q@t[0];shifted_stereo=shifted[1]-q@shifted[0]
        centers=-np.einsum('vji,vj->vi',r,t);shifted_centers=-np.einsum('vji,vj->vi',r,shifted)
        residuals['projection_camera_space_max_abs_mm'].append(float(np.abs(projection-shifted_projection).max()))
        residuals['relative_stereo_translation_max_abs_mm'].append(float(np.abs(stereo-shifted_stereo).max()))
        residuals['center_shift_max_abs_mm'].append(float(np.abs(shifted_centers-centers-delta).max()))
        residuals['float32_translation_roundtrip_max_abs_mm'].append(float(np.abs(new[:,6:].double().numpy()*100-shifted).max()))
        # Audit the actual encoded representation separately from exact float64 geometry.
        columns=original[:,:6].double().numpy().reshape(2,2,3)
        encoded_r=np.stack((columns[:,0],columns[:,1],np.cross(columns[:,0],columns[:,1])),axis=-1)
        encoded_t=original[:,6:].double().numpy()*100;encoded_new_t=new[:,6:].double().numpy()*100
        residuals['float32_projection_camera_space_max_abs_mm'].append(float(np.abs(
            np.einsum('vij,j->vi',encoded_r,probe)+encoded_t
            -np.einsum('vij,j->vi',encoded_r,probe+delta)-encoded_new_t).max()))
        encoded_q=encoded_r[1]@encoded_r[0].T
        residuals['float32_relative_stereo_translation_max_abs_mm'].append(float(np.abs(
            (encoded_t[1]-encoded_q@encoded_t[0])-(encoded_new_t[1]-encoded_q@encoded_new_t[0])).max()))
        encoded.append(new);transformed.append(shifted.tolist())
    maxima={k:max(v) for k,v in residuals.items()}
    if any(v>TOL['invariance_mm'] for k,v in maxima.items() if not k.startswith('float32')):
        raise ValueError('gauge_physical_invariance_failed')
    if maxima['float32_translation_roundtrip_max_abs_mm']>TOL['encoding_roundtrip_mm']:
        raise ValueError('gauge_encoding_roundtrip_limit')
    return torch.stack(encoded),{'delta_canonical_world_mm':delta.tolist(),'transformed_translation_mm':transformed,
        'encoded_camera_18':torch.stack(encoded).flatten(1).tolist(),'max_residuals':maxima,
        'definitions':"x'=x+delta; t'=t-R@delta; same delta for both views and every frame; R/device GT/RGB unchanged"}


def paired_metrics(prediction,baseline,repeat):
    result={}
    for name,x in prediction.items():
        base=baseline[name];noise=(repeat[name]-base).abs();difference=(x-base).abs()
        limit=TOL['atol_mm']+TOL['rtol']*base.abs()+2*noise
        norms=(x-base).double().norm(dim=-1)
        result[name]={'l2_sum_mm':float(norms.sum()),'joint_count':norms.numel(),'l2_mean_mm':float(norms.mean()),
            'l2_max_mm':float(norms.max()),'coordinate_max_abs_mm':float(difference.max()),
            'coordinates_above_repeat_aware_tolerance':int((difference>limit).sum()),'coordinate_count':difference.numel(),
            'above_fraction':float((difference>limit).double().mean())}
    return result


def reference_comparison(prediction,xy,reference):
    errors={};match=True
    for name,x in prediction.items():
        old=torch.tensor(reference['prediction'][name],dtype=x.dtype)
        if old.shape!=x.shape or not torch.isfinite(old).all():raise ValueError('gauge_reference_prediction_shape')
        close=torch.allclose(x,old,atol=TOL['atol_mm'],rtol=TOL['rtol']);match &= close
        errors[name]={'within_tolerance':close,'max_abs_mm':float((x-old).abs().max())}
    old_xy=torch.tensor(reference['predicted_2d_native'],dtype=xy.dtype)
    close=torch.allclose(xy,old_xy,atol=TOL['atol_native_px'],rtol=TOL['rtol']);match &= close
    errors['two_d']={'within_tolerance':close,'max_abs_native_px':float((xy-old_xy).abs().max())}
    return {'status':'within_tolerance' if match else 'reference_mismatch','details':errors,
            'interpretation':'Cross-run/GPU/version comparison; mismatch is reported separately from within-run paired evidence'}


def repeat_stability(base,repeat):
    stable=all(torch.allclose(base[k],repeat[k],atol=TOL['atol_mm'],rtol=TOL['rtol']) for k in base)
    return {'status':'stable_within_fixed_tolerance' if stable else 'inconclusive_repeat_unstable',
            'atol_mm':TOL['atol_mm'],'rtol':TOL['rtol'],
            'max_abs_mm':{k:float((base[k]-repeat[k]).abs().max()) for k in base}}


@torch.inference_mode()
def evaluate(inputs,plan,prepared_path,prepared_pin,references,model_path,output,device,*,budget=None):
    budget=budget or cross.Budget(LIMITS['infer_seconds']);output=Path(output);output.mkdir()
    prepared=verify_prepared(prepared_path,prepared_pin,plan,references)
    _,spec,_,binding=cross.assets(inputs)
    if binding!=plan['binding']:raise ValueError('gauge_evaluation_input_binding')
    completed=[]
    try:
        models,pins,model_report,implementation=inference.view._models(model_path,spec,True,training_sha256=cross.MODEL_SHA,run_id=inference.view.FULL_RUN_ID)
        if pins!=references['model_weights'] or implementation!=references['implementation']:
            raise ValueError('gauge_model_or_implementation_mismatch')
        models={k:v.to(device) for k,v in models.items()};models['spec']=spec
        for number,(entry,ref) in enumerate(zip(prepared['clips'],references['clips'],strict=True)):
            budget.check();root=Path(prepared_path)/f'clip-{number:02d}';p=entry['tensor']
            raw=_regular_read(root/'clip.pt',32*cross.MIB,p['bytes'])
            if sha(raw)!=p['sha256']:raise ValueError('gauge_prepared_tensor_pin')
            data=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if (data['images'].shape!=(50,2,3,128,128) or data['camera'].shape!=(50,2,9)
                    or data['target'].shape!=(50,62,3) or not all(torch.isfinite(v).all() for v in data.values())):
                raise ValueError('gauge_prepared_tensor_contract')
            rows=ref['camera']['frames'];keys=[r['key'] for r in rows]
            if (not torch.equal(data['target'],torch.tensor(ref['prediction']['gt_target_device_mm']))
                    or not torch.equal(data['target_valid'],torch.tensor(ref['prediction']['gt_target_valid']))):
                raise ValueError('gauge_reference_gt_changed')
            base_camera,audit=transform_gauge(rows,[0.,0.,0.])
            if not torch.equal(base_camera,data['camera']):raise ValueError('gauge_original_camera_tensor_mismatch')
            directory=output/f'clip-{number:02d}';directory.mkdir()
            baseline,features=inference.predict(models,data['images'],base_camera,keys,model_report['config'],device)
            repeat=inference.sample_features(models,features,base_camera,keys,model_report['config'],device)
            stability=repeat_stability(baseline,repeat);budget.check()
            predictions={'baseline':baseline,'zero_repeat':repeat};audits={'baseline':audit,'zero_repeat':audit}
            for variant in VARIANTS[2:]:
                camera,audits[variant]=transform_gauge(rows,camera_delta(rows,variant))
                predictions[variant]=inference.sample_features(models,features,camera,keys,model_report['config'],device);budget.check()
            prefix=inference.sample_features(models,features[:25],base_camera[:25],keys[:25],model_report['config'],device)
            causal={k:float((baseline[k][:25]-prefix[k]).abs().max()) for k in baseline}
            if any(not torch.allclose(baseline[k][:25],prefix[k],atol=TOL['atol_mm'],rtol=TOL['rtol']) for k in baseline):
                raise ValueError('gauge_baseline_causality_failed')
            summaries={};xy=None
            for variant,prediction in predictions.items():
                budget.check();metrics,xy,_=inference.metric_sums(prediction,features,data,spec)
                paired=paired_metrics(prediction,baseline,repeat)
                sensitive=any(paired[k]['coordinates_above_repeat_aware_tolerance']>0 for k in paired)
                decision=('inconclusive_repeat_unstable' if stability['status']!='stable_within_fixed_tolerance' else
                          'exceeds_repeat_aware_tolerance' if sensitive else 'within_repeat_aware_tolerance')
                payload=seal({'schema':'v12_camera_gauge_prediction_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
                    'sequence':entry['sequence'],'environment':entry['environment'],'keys':keys,'variant':variant,
                    'prediction':{k:v.tolist() for k,v in prediction.items()},'camera':audits[variant],
                    'frame_metrics':metrics,'aggregate':inference.aggregate(metrics),'paired':paired,'decision':decision,
                    'features_sha256':sha(features.contiguous().numpy().tobytes()),
                    'noise':'same keyed sequence/frame seed71 offsets100/200; fresh sampler per call',
                    'gt_conditioning':False,'reference_prediction_sha256':ref['prediction']['content_sha256']})
                observation=control.observation({'sequence':entry['sequence'],'environment':entry['environment'],
                    'frames':rows},{'keys':keys,'prediction':payload['prediction'],'gt_target_device_mm':data['target'].tolist(),
                        'gt_target_valid':data['target_valid'].tolist(),'aggregate':payload['aggregate'],'content_sha256':payload['content_sha256']},spec)
                payload=seal({**{k:v for k,v in payload.items() if k!='content_sha256'},'body_centroid_bias_mean_mm':observation['body_centroid_bias_mean_mm'],
                    'centered_body_error_mean_mm':observation['centered_body_error_mean_mm'],
                    'centered_body_error_sum_mm':observation['centered_body_error_sum_mm'],
                    'centered_body_valid_joints':observation['centered_body_valid_joints']})
                cross.write_report(directory/(variant+'.json'),payload)
                summaries[variant]={k:payload[k] for k in ('content_sha256','aggregate','paired','decision','body_centroid_bias_mean_mm','centered_body_error_mean_mm')}
                enforce_result_budget(output)
            ref_check=reference_comparison(baseline,xy,ref['prediction'])
            artifacts=render_comparison(directory,root,entry,predictions,xy,data,spec,budget,
                LIMITS['result_bytes']-8*cross.MIB-enforce_result_budget(output))
            clip=seal({'schema':'v12_camera_gauge_clip_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
                'number':number,'sequence':entry['sequence'],'environment':entry['environment'],'keys':keys,
                'reference_comparison':ref_check,'repeat_stability':stability,'variants':summaries,'artifacts':artifacts,
                'baseline_prefix25_max_abs_mm':causal,'gt_target_device_mm':data['target'].tolist(),
                'gt_target_valid':data['target_valid'].tolist(),'predicted_2d_native':xy.tolist(),
                'render_bounds_device_mm':render_bounds(data,predictions),
                'render_scope':'one shared device-coordinate projection over GT and all three unique variants, entire clip'})
            cross.write_report(directory/'report.json',clip);completed.append(clip);enforce_result_budget(output)
        result=seal({'schema':'v12_camera_gauge_report_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
            'model_zip_sha256':cross.MODEL_SHA,'model_weights':pins,'inference_implementation_sha256':implementation,
            'selection_sha256':plan['content_sha256'],'prepared_sha256':prepared_pin,'references':POLICY['references'],
            'reference_report_sha256':references['reference_report_sha256'],'status':'complete','clips':[
                {k:c[k] for k in ('number','sequence','environment','content_sha256','reference_comparison','repeat_stability','variants')} for c in completed],
            'reference_match_all':all(c['reference_comparison']['status']=='within_tolerance' for c in completed),
            'reference_environments':references['reference_environments'],
            'repeat_stable_all':all(c['repeat_stability']['status']=='stable_within_fixed_tolerance' for c in completed),
            'sensitivity_definition':'coordinate |variant-base| > .01mm + 1e-5*|base| + 2*|zero_repeat-base|; report count, denominator, joint L2 mean/max; repeat unstable => inconclusive',
            'comparison_tolerance':TOL,'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'device_vram_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
            'torch':str(torch.__version__),'threads':torch.get_num_threads(),'precision':'fp32','ddim_steps':10,'seed':71,
            'noise_offsets':{'body':100,'hand':200},'nominal_fps':25,'recorded_timestamps_verified':False,
            'stereo_sync_measured':False,'original_motion_independence_verified':False,
            'elapsed_seconds':time.monotonic()-budget.start,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'interpretation':'World-coordinate representation sensitivity only; not a trained fix or generalization proof; no GT-based delta tuning'})
        cross.write_report(output/'report.json',result);write_table(output,result);enforce_result_budget(output);return result
    except BaseException as error:
        cross.write_report(output/'failure.json',{'state':'failed','type':type(error).__name__,'reason':str(error)[:2000],
            'completed_clips':[c['sequence'] for c in completed]});raise


def enforce_result_budget(output):
    total=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    if total>LIMITS['result_bytes']-2*cross.MIB:raise ValueError('gauge_result_file_sum')
    return total


def write_table(output,report):
    rows=['# 공통 world 원점 이동 민감도','',
        '| 환경/clip | variant | body mm | 중심 mm | bone 예측/GT mm | spread 예측/GT mm | paired body mean/max mm | 판정 |',
        '| --- | --- | ---: | ---: | --- | --- | --- | --- |']
    for clip in report['clips']:
        for variant,v in clip['variants'].items():
            s=v['aggregate']['body_shape_sums'];p=v['paired']['body']
            rows.append(f"| {clip['environment']}/{clip['number']} | {variant} | {v['aggregate']['body']['mean']:.2f} | {v['body_centroid_bias_mean_mm']:.2f} | {s['predicted_mean_bone_mm']:.2f}/{s['target_mean_bone_mm']:.2f} | {s['predicted_mean_spread_mm']:.2f}/{s['target_mean_spread_mm']:.2f} | {p['l2_mean_mm']:.3f}/{p['l2_max_mm']:.3f} | {v['decision']} |")
    rows += ['',f"이전 실행 baseline 일치: {report['reference_match_all']}; 같은 실행 Δ0 반복 안정: {report['repeat_stable_all']}.",
        '카메라 원점 표현만 바꾼 무재학습 진단입니다. GT 오차로 변환을 선택하거나 튜닝하지 않습니다. 다른 GPU/버전의 이전 실행 불일치와 같은 실행 paired 변화를 구분합니다.',
        '판정은 coordinate 절대·상대 tolerance와 Δ0 반복 차이의 2배를 넘는지 기준입니다. 단순한 0 초과 판정이 아닙니다. 반복 불안정시 inconclusive입니다. 명목25FPS·시간동기·motion 독립성 미검증을 유지합니다.']
    (Path(output)/'comparison.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def render_comparison(output,clip_root,entry,predictions,xy,data,spec,budget,available,*,labels=None,title='camera gauge'):
    from PIL import Image,ImageDraw,ImageFont
    view=inference.view;names=('baseline','first_midpoint','world_x_plus_100m')
    target=data['target'].tolist()
    for t in range(50):
        for j in range(62):
            if not data['target_valid'][t,j]:target[t][j]=None
    arrays=[target]+[predictions[n]['whole'].tolist() for n in names]
    flat=[[p for frame in a for p in frame] for a in arrays];bounds=render_bounds(data,predictions)
    projected=view._shared_3d_groups(flat,220)
    projected=[[g[t*62:(t+1)*62] for t in range(50)] for g in projected]
    pictures=[];files=[];size=220
    for t,row in enumerate(entry['frames']):
        budget.check();image=Image.new('RGB',(952,810),view.DARK);draw=ImageDraw.Draw(image)
        font=ImageFont.load_default(size=13)
        draw.text((12,8),f"{title} | clip {entry['number']} / frame {row['frame']} | nominal25FPS",fill='white',font=font)
        draw.text((12,29),f"shared device xyz bounds mm: {np.round(bounds['min'],1)} .. {np.round(bounds['max'],1)}",fill='white',font=font)
        for col,label in enumerate(labels or ('GT',)+names):draw.text((12+col*235,54),label,fill='white',font=font)
        for v,side in enumerate(('left','right')):
            pin=row['files'][side];raw=_regular_read(clip_root/pin['path'],8*cross.MIB,pin['size_bytes'])
            if sha(raw)!=pin['sha256']:raise ValueError('gauge_render_raw_hash')
            for col in range(4):
                if col==0:
                    points=data[side+'_xy'][t].tolist()
                    for j in range(62):
                        if not data[side+'_xy_valid'][t,j]:points[j]=None
                else:points=xy[t,v].tolist()
                image.paste(view._rgb_tile(raw,points,spec,size),(12+col*235,80+v*235))
        for col in range(4):
            tile=Image.new('RGB',(size,size),'#202a38');view._draw_joints(tile,projected[col][t],spec)
            image.paste(tile,(12+col*235,550))
        draw.text((12,784),'Device GT unchanged; RGB/2D identical for every predicted variant; repeat stored as numbers.',fill='white',font=font)
        if t in (0,25,49):
            name=f"frame-{row['frame']:09d}.png"
            with (output/name).open('xb') as stream:image.save(inference.CappedOutput(stream,available),format='PNG')
            available-=(output/name).stat().st_size;files.append(name)
        pictures.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=128));image.close()
    with (output/'comparison.gif').open('xb') as stream:
        pictures[0].save(inference.CappedOutput(stream,available),format='GIF',save_all=True,append_images=pictures[1:],duration=40,loop=0,optimize=False)
    for p in pictures:p.close()
    files.append('comparison.gif')
    return [{'name':n,'bytes':(output/n).stat().st_size,'sha256':sha((output/n).read_bytes())} for n in files]


def render_bounds(data,predictions):
    points=torch.cat([data['target'][data['target_valid']]]+[predictions[n]['whole'].reshape(-1,3) for n in ('baseline','first_midpoint','world_x_plus_100m')])
    return {'min':points.amin(0).tolist(),'max':points.amax(0).tolist(),'unit':'device_mm','scope':'all50 frames, GT valid and all3 unique variants'}

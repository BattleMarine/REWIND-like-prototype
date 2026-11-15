"""Evaluation-only upper-condition intervention; body and observation inputs stay fixed."""

import io
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from PIL import Image,ImageDraw,ImageFont

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from .teacher import V1Body,V1Hand
from .data import _hand_indices,attach_predicted_hands
from .run_stages import CascadeStages
from . import upper_condition_bundle as bundle,cross_inference as infer,camera_gauge as gauge,cross_data as cross


IMPLEMENTATIONS=('rewind_v1/teacher.py','rewind_v1/attention.py','rewind_v1/data.py','rewind_m2/models.py',
    'rewind_m2/data.py','rewind_m2/diffusion.py','rewind_m2/streaming.py','rewind_m3/models.py',
    'rewind_v1/camera_training.py','scripts/visualize_archvis_v1.py')


def load_models(root,inputs,device):
    cfg=inputs['run']['config'];spec=inputs['spec'];models={}
    for name in IMPLEMENTATIONS:
        raw=(bundle.ROOT/name).read_bytes();lf=raw.replace(b'\r\n',b'\n')
        if inputs['run']['source'][name] not in (sha(raw),sha(lf),sha(lf.replace(b'\n',b'\r\n'))):raise ValueError('v151_frozen_implementation:'+name)
    for role in ('body','hand'):
        name=f'train/first_midpoint-{role}-u1000.pt'
        raw=_regular_read(Path(root)/name,128*cross.MIB)
        if sha(raw)!=bundle.POLICY['model_sha256'][role]:raise ValueError('v151_frozen_model_hash')
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if (state['role']!=role or state['arm']!='first_midpoint' or state['step']!=1000
                or any(digest(inputs['run'].get(k))!=digest(v) for k,v in state['binding'].items())
                or state['initial']!=inputs['run']['initial'][role]
                or not all(torch.isfinite(v).all() for v in state['model'].values())):raise ValueError('v151_frozen_model_binding')
        names=infer.view._group_names(spec,role);edges=[(names.index(a),names.index(b)) for a,b in spec['edges'] if a in names and b in names]
        model=V1Body(cfg['teacher'],len(names),edges) if role=='body' else V1Hand(cfg['teacher'],len(names),edges,len(spec['upper_joint_names']))
        model.load_state_dict(state['model'],strict=True);models[role]=model.to(device).eval().requires_grad_(False)
    models['spec']=spec
    return models,cfg


@torch.inference_mode()
def sample_group(model,role,features,camera,keys,cfg,spec,device):
    names=infer.view._group_names(spec,role);indices=[spec['joint_names'].index(n) for n in names]
    x=features.reshape(1,len(keys),len(spec['joint_names']),14).to(device)
    offset=bundle.POLICY['noise_offsets'][role]
    noise=torch.stack([keyed_frame_noise(s,f,len(names),cfg['seed']+offset) for s,f in keys])[None].to(device)
    schedule=DiffusionSchedule(cfg['teacher']['diffusion_steps'],cfg['teacher']['beta_start'],cfg['teacher']['beta_end'])
    valid=torch.ones((1,len(keys)),dtype=torch.bool,device=device)
    result=schedule.sample(model,x[...,indices,:].flatten(-2),camera[None].to(device),valid,noise,cfg['teacher']['sampling_steps'])*cfg['teacher']['target_scale_mm']
    if not torch.isfinite(result).all():raise ValueError('v151_nonfinite_prediction')
    return result[0].cpu()


def upper_values(body,target,valid,spec,condition):
    if condition=='predicted_upper':
        idx=[spec['body_joint_names'].index(n) for n in spec['upper_joint_names']]
        values=body[:,idx];mask=torch.ones(values.shape[:-1],dtype=torch.bool)
    elif condition=='gt_upper_diagnostic':
        idx=[spec['joint_names'].index(n) for n in spec['upper_joint_names']]
        mask=valid[:,idx];values=torch.where(mask[...,None],target[:,idx],0.)
    else:raise ValueError('v151_diagnostic_condition_only')
    if not torch.isfinite(values).all():raise ValueError('v151_upper_finite')
    return values,mask


def sample_hand(model,upper,valid,features,camera,keys,cfg,spec,device):
    conditioned=BoundHand(model,upper[None].to(device)/cfg['teacher']['target_scale_mm'],valid[None].to(device)).eval()
    return sample_group(conditioned,'hand',features,camera,keys,cfg,spec,device)


def prediction_dict(body,local,spec):
    global_hand=attach_predicted_hands(body,local,spec)
    names=infer.view._group_names(spec,'body')+infer.view._group_names(spec,'hand')
    values=dict(zip(names,torch.cat((body,global_hand),dim=-2).unbind(-2)))
    return {'body':body,'hand_local':local,'hand_global':global_hand,'whole':torch.stack([values[n] for n in spec['joint_names']],dim=-2)}


def errors(predicted,target,mask):
    error=(predicted.double()-target.double()).norm(dim=-1);values=error[mask]
    return {'sum_mm':float(values.sum()),'count':values.numel(),'mean_mm':float(values.mean()) if values.numel() else None,
            'per_frame':[{'sum_mm':float(e[m].sum()),'count':int(m.sum())} for e,m in zip(error,mask,strict=True)]}


def shape(predicted,target,mask,spec):
    names=infer.view._group_names(spec,'hand');edges=[(names.index(a),names.index(b)) for a,b in spec['edges'] if a in names and b in names]
    clip=SimpleNamespace(target_mm=target[None],target_valid=mask[None],time_valid=torch.ones(1,len(target),dtype=torch.bool))
    sums=CascadeStages._shape(predicted[None],clip,edges)
    return {**sums,**{f'{label}_mean_{metric}_mm':sums[f'{label}_{metric}']/sums[denom] if sums[denom] else None
        for label in ('predicted','target') for metric,denom in (('bone','bones'),('spread','frames'))}}


def factorial_metrics(body,locals,target,valid,spec):
    fingers,wrists=_hand_indices(spec['joint_names'],spec)
    body_wrists=[spec['body_joint_names'].index(spec['joint_names'][w]) for w in wrists]
    truth=target[:,fingers];gt_wrist=target[:,wrists];pred_wrist=body[:,body_wrists]
    common=valid[:,fingers]&valid[:,wrists];local_truth=truth-gt_wrist
    result={};globals={}
    for condition,local in locals.items():
        local_error=errors(local,local_truth,common);attachments={};globals[condition]={}
        for wrist,values in (('predicted_wrist',pred_wrist),('gt_wrist_diagnostic',gt_wrist)):
            points=local+values;globals[condition][wrist]=points
            common_error=errors(points,truth,common)
            attachments[wrist]={'common_mask_error':common_error}
            if wrist=='predicted_wrist':attachments[wrist]['parent_native_mask_error']=errors(points,truth,valid[:,fingers])
            else:
                delta=((points.double()-truth.double()).norm(dim=-1)-(local.double()-local_truth.double()).norm(dim=-1)).abs()[common]
                residual=float(delta.max()) if delta.numel() else 0.
                if residual>bundle.POLICY['tolerance']['attachment_identity_atol_mm']:raise ValueError('v151_gt_wrist_local_identity')
                attachments[wrist]['local_global_error_identity_max_abs_mm']=residual
        result[condition]={'local_common_mask':local_error,'local_shape_common_mask':shape(local,local_truth,common,spec),'attachments':attachments}
    return result,globals,common,local_truth


def reference_check(current,parent):
    report={};match=True
    for name,x in current.items():
        old=torch.tensor(parent['variants']['baseline']['prediction'][name],dtype=x.dtype)
        if old.shape!=x.shape or not torch.isfinite(old).all():raise ValueError('v151_parent_prediction_contract')
        close=torch.allclose(x,old,atol=.01,rtol=1e-5);match &= close
        report[name]={'max_abs_mm':float((x-old).abs().max()),'within_tolerance':close}
    return {'status':'within_tolerance' if match else 'cross_run_mismatch','details':report,
        'interpretation':'Cross-run mismatch may involve GPU/environment but cause is not presumed; review pins, prefix and repeat separately; no new best or re-tuning'}


@torch.inference_mode()
def run(root,inputs,output,device):
    output=Path(output);output.mkdir();budget=cross.Budget(bundle.LIMITS['infer_seconds']);completed=[]
    models,cfg=load_models(root,inputs,device)
    spec=inputs['spec'];tol=bundle.POLICY['tolerance']
    if any(m.training or any(p.requires_grad for p in m.parameters()) for r,m in models.items() if r in ('body','hand')):
        raise ValueError('v151_models_must_be_frozen_eval')
    fingerprints_before={r:bundle.parent.model_fingerprint(models[r]) for r in ('body','hand')}
    try:
        for number,clip in enumerate(inputs['clips']):
            budget.check();data=clip['data'];features=data['features'];camera=clip['camera'];keys=clip['keys']
            input_before={n:sha(x.contiguous().numpy().tobytes()) for n,x in
                (('features',features),('camera',camera),('target',data['target']),('target_valid',data['target_valid']))}
            body=sample_group(models['body'],'body',features,camera,keys,cfg,spec,device)
            upper_a,mask_a=upper_values(body,data['target'],data['target_valid'],spec,'predicted_upper')
            upper_b,mask_b=upper_values(body,data['target'],data['target_valid'],spec,'gt_upper_diagnostic')
            a=sample_hand(models['hand'],upper_a,mask_a,features,camera,keys,cfg,spec,device);budget.check()
            b=sample_hand(models['hand'],upper_b,mask_b,features,camera,keys,cfg,spec,device);budget.check()
            repeat=sample_hand(models['hand'],upper_a,mask_a,features,camera,keys,cfg,spec,device)
            current=prediction_dict(body,a,spec);repeated=prediction_dict(body,repeat,spec);changed=prediction_dict(body,b,spec)
            stability=gauge.repeat_stability(current,repeated);paired=gauge.paired_metrics(changed,current,repeated)
            effect=('inconclusive_A_repeat_unstable' if stability['status']!='stable_within_fixed_tolerance' else
                'exceeds_A_repeat_aware_tolerance' if paired['hand_local']['coordinates_above_repeat_aware_tolerance'] else 'within_A_repeat_aware_tolerance')
            prefix_body=sample_group(models['body'],'body',features[:25],camera[:25],keys[:25],cfg,spec,device)
            prefix_upper,prefix_mask=upper_values(prefix_body,data['target'][:25],data['target_valid'][:25],spec,'predicted_upper')
            prefix_a=sample_hand(models['hand'],prefix_upper,prefix_mask,features[:25],camera[:25],keys[:25],cfg,spec,device)
            prefix_b=sample_hand(models['hand'],upper_b[:25],mask_b[:25],features[:25],camera[:25],keys[:25],cfg,spec,device)
            causal={'body':float((body[:25]-prefix_body).abs().max()),'A_local':float((a[:25]-prefix_a).abs().max()),'B_local':float((b[:25]-prefix_b).abs().max())}
            if any(not torch.allclose(x[:25],p,atol=tol['atol_mm'],rtol=tol['rtol']) for x,p in ((body,prefix_body),(a,prefix_a),(b,prefix_b))):
                raise ValueError('v151_prefix25_causality')
            metrics,globals,common,local_gt=factorial_metrics(body,{'A_predicted_upper':a,'B_gt_upper_diagnostic':b},data['target'],data['target_valid'],spec)
            directory=output/f'clip-{number:02d}';directory.mkdir()
            artifacts=render(directory,number,spec,data,body,a,b,globals,common,local_gt,budget,
                bundle.LIMITS['result_bytes']-4*cross.MIB-result_bytes(output))
            input_after={n:sha(x.contiguous().numpy().tobytes()) for n,x in
                (('features',features),('camera',camera),('target',data['target']),('target_valid',data['target_valid']))}
            if input_before!=input_after:raise ValueError('v151_frozen_input_mutated')
            result=seal({'schema':'v151_upper_clip_v1','run_id':bundle.RUN_ID,'source_sha256':digest(bundle.source_identity()),
                'number':number,'sequence':keys[0][0],'keys':keys,'parent_prediction_sha256':clip['parent']['content_sha256'],
                'body_prediction_mm':body.tolist(),'local_predictions_mm':{'A_predicted_upper':a.tolist(),'B_gt_upper_diagnostic':b.tolist(),'A_numerical_repeat':repeat.tolist()},
                'global_predictions_mm':{c:{w:x.tolist() for w,x in group.items()} for c,group in globals.items()},
                'target_device_mm':data['target'].tolist(),'target_valid':data['target_valid'].tolist(),'common_hand_mask':common.tolist(),
                'upper_joint_order':spec['upper_joint_names'],'upper_condition_mm':{'A':upper_a.tolist(),'B_GT':upper_b.tolist()},
                'upper_valid':{'A':mask_a.tolist(),'B_GT':mask_b.tolist()},'gt_upper_contains_wrists':all(w in spec['upper_joint_names'] for w in spec['wrists'].values()),
                'metrics':metrics,'body_error':errors(body,data['target'][:,[spec['joint_names'].index(n) for n in spec['body_joint_names']]],data['target_valid'][:,[spec['joint_names'].index(n) for n in spec['body_joint_names']]]),
                'cross_run_parent_comparison':reference_check(current,clip['parent']),'same_run_A_repeat':stability,'paired_upper_change':paired,
                'local_condition_effect_status':effect,
                'causal_prefix25_max_abs_mm':causal,'features_sha256':sha(features.numpy().tobytes()),'normalized_camera_18':camera.flatten(1).tolist(),
                'input_fingerprints_before':input_before,'input_fingerprints_after':input_after,
                'artifacts':artifacts,'scope':'GT-conditioned diagnostic, not deployable performance or independent generalization; 2x2 isolates attachment only, not wrist content inside GT upper'})
            cross.write_report(directory/'predictions.json',result);completed.append(result);result_bytes(output)
        fingerprints_after={r:bundle.parent.model_fingerprint(models[r]) for r in ('body','hand')}
        if fingerprints_before!=fingerprints_after:raise ValueError('v151_frozen_model_mutated')
        report=seal({'schema':'v151_upper_report_v1','run_id':bundle.RUN_ID,'source_sha256':digest(bundle.source_identity()),
            'parent_zip_sha256':bundle.POLICY['parent_zip_sha256'],'parent_source_sha256':bundle.POLICY['parent_source_sha256'],
            'parent_report_sha256':bundle.POLICY['parent_report_sha256'],'bundle_manifest_sha256':inputs['bundle']['content_sha256'],
            'model_sha256':bundle.POLICY['model_sha256'],'status':'complete','clips':[{'number':c['number'],'sequence':c['sequence'],
                'content_sha256':c['content_sha256'],'metrics':c['metrics'],'same_run_A_repeat':c['same_run_A_repeat'],
                'cross_run_parent_comparison':c['cross_run_parent_comparison']} for c in completed],
            'elapsed_seconds':time.monotonic()-budget.start,'device':str(device),'torch':str(torch.__version__),'precision':'fp32',
            'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'device_vram_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'parent_environment':inputs['run']['environment'],'seed':71,'offsets':{'body':100,'hand':200},'ddim_steps':10,
            'model_fingerprints_before':fingerprints_before,'model_fingerprints_after':fingerprints_after,
            'same_run_repeat_scope':'A hand resampling only, with identical saved body/upper/features/camera/noise; body is not resampled for repeat',
            'sampling_calls_per_clip':{'body_full':1,'A_local_full':1,'B_local_full':1,'A_repeat_full':1,'body_prefix25':1,'A_prefix25':1,'B_prefix25':1},
            'gt_usage':'GT upper is evaluation-only intervention; GT wrist attachment is postprocessing only; original GT-upper train gate unchanged',
            'evaluation_scope':'same six development/regression clips, nominal25FPS; no independent generalization, training or frontend inference'})
        cross.write_report(output/'report.json',report);write_table(output,completed);result_bytes(output);return report
    except BaseException as error:
        cross.write_report(output/'failure.json',{'state':'failed','type':type(error).__name__,'reason':str(error)[:2000],
            'completed_clips':[c['sequence'] for c in completed]});raise


def result_bytes(output):
    total=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    if total>bundle.LIMITS['result_bytes']-2*cross.MIB:raise ValueError('v151_result_sum_limit')
    return total


def write_table(output,clips):
    rows=['# v1.5.1 GT 상체 조건 진단','',
        '| clip | condition | local mm | local bone pred/GT mm | local spread pred/GT mm | pred wrist global mm | GT wrist global mm | count |',
        '| --- | --- | ---: | --- | --- | ---: | ---: | ---: |']
    for c in clips:
        for condition,m in c['metrics'].items():
            s=m['local_shape_common_mask'];a=m['attachments']
            def n(x):return 'N/A' if x is None else f'{x:.3f}'
            rows.append('| '+' | '.join([str(c['number']),condition,n(m['local_common_mask']['mean_mm']),
                n(s['predicted_mean_bone_mm'])+'/'+n(s['target_mean_bone_mm']),n(s['predicted_mean_spread_mm'])+'/'+n(s['target_mean_spread_mm']),
                n(a['predicted_wrist']['common_mask_error']['mean_mm']),n(a['gt_wrist_diagnostic']['common_mask_error']['mean_mm']),str(m['local_common_mask']['count'])])+' |')
    rows+=['','모든 표 값은 공통 finger&wrist GT mask를 사용합니다. 부모 native-mask 지표는 JSON에서 별도로 표시합니다.',
        'GT upper에는 wrist 입력도 포함합니다. 2×2는 후처리 부착 효과만 분리하며 조건 안의 wrist 효과까지 독립 분해하지 않습니다.',
        '같은 실행 A-repeat와 이전 환경 baseline 불일치를 구분합니다. GT 조건은 진단이며 배포·독립 일반화 성능이 아닙니다.']
    (Path(output)/'comparison.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def render(output,number,spec,data,body,a,b,globals,mask,local_gt,budget,available):
    view=infer.view;fingers,wrists=_hand_indices(spec['joint_names'],spec);idx=[spec['joint_names'].index(n) for n in spec['body_joint_names']]
    # Each row uses one shared axis across GT and all columns, never per-prediction rescaling.
    body_gt=data['target'][:,idx];body_valid=data['target_valid'][:,idx]
    # Render local hands as a 40-joint subset using original edge names.
    def subset(names):return {**spec,'joint_names':names,'edges':[(a,b) for a,b in spec['edges'] if a in names and b in names]}
    hand_spec=subset(infer.view._group_names(spec,'hand'));body_spec=subset(spec['body_joint_names'])
    def project(groups,valid):
        if not valid.any():return [[[None]*groups[0].shape[1] for _ in range(len(groups[0]))] for _ in groups]
        points=[]
        for x in groups:
            values=x.tolist()
            for t in range(len(values)):
                for j in range(len(values[t])):
                    if not valid[t,j]:values[t][j]=None
            points.append([p for frame in values for p in frame])
        projected=view._shared_3d_groups(points,200)
        joints=groups[0].shape[1]
        return [[g[t*joints:(t+1)*joints] for t in range(len(groups[0]))] for g in projected]
    body_proj=project([body_gt,body,body],body_valid)
    local_proj=project([local_gt,a,b],mask)
    global_proj=project([data['target'][:,fingers],globals['A_predicted_upper']['predicted_wrist'],globals['B_gt_upper_diagnostic']['predicted_wrist']],mask)
    def bounds_for(group,valid):
        p=torch.cat([x[valid] for x in group])
        if not p.numel():return {'min':None,'max':None,'unit':'mm','projection':None}
        uv=torch.tensor(view._project(p.tolist()),dtype=torch.float64);lo=uv.amin(0);hi=uv.amax(0)
        scale=136/max(float((hi-lo).max()),1.)
        return {'min':p.amin(0).tolist(),'max':p.amax(0).tolist(),'unit':'mm',
            'projection':{'basis':'u=cos(.38)*x+sin(.38)*z; v=-y','source_uv_min_mm':lo.tolist(),'source_uv_max_mm':hi.tolist(),
                'pixels_per_mm':scale,'pixel_min':(100-(hi-lo)*scale/2).tolist(),'pixel_max':(100+(hi-lo)*scale/2).tolist(),'tile_pixels':200}}
    row_bounds={'body':bounds_for([body_gt,body,body],body_valid),'local_hand':bounds_for([local_gt,a,b],mask),
        'global_pred_wrist':bounds_for([data['target'][:,fingers],globals['A_predicted_upper']['predicted_wrist'],globals['B_gt_upper_diagnostic']['predicted_wrist']],mask)}
    pictures=[];files=[]
    for t in range(len(a)):
        budget.check();image=Image.new('RGB',(738,810),view.DARK);draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=13)
        draw.text((12,8),f'v1.5.1 GT-CONDITION DIAGNOSTIC | clip {number} frame {t}',fill='white',font=font)
        for col,label in enumerate(('GT','A: predicted upper','B: GT upper diagnostic')):draw.text((12+col*244,34),label,fill='white',font=font)
        for row,(projection,dspec,title) in enumerate(((body_proj,body_spec,'Fixed body / device mm'),
                (local_proj,hand_spec,'Local hands: both wrist origins overlap at (0,0,0) / primary shape'),(global_proj,hand_spec,'Global hands / same predicted wrist attachment'))):
            y=62+row*244;draw.text((12,y),title,fill='white',font=font)
            scope=list(row_bounds.values())[row]
            bounds_text='no valid GT points' if scope['min'] is None else f"xyz mm: {[round(v,1) for v in scope['min']]} .. {[round(v,1) for v in scope['max']]}"
            draw.text((12,y+17),bounds_text,fill='white',font=ImageFont.load_default(size=12))
            for col in range(3):
                tile=Image.new('RGB',(200,200),'#202a38');view._draw_joints(tile,projection[col][t],dspec);image.paste(tile,(22+col*244,y+38))
        draw.text((12,795),'Same axes per row/all frames; GT condition is not deployable evaluation.',fill='white',font=font)
        if t in (0,len(a)//2,len(a)-1):
            name=f'frame-{t:09d}.png'
            with (output/name).open('xb') as stream:image.save(infer.CappedOutput(stream,available),format='PNG')
            available-=(output/name).stat().st_size;files.append(name)
        pictures.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=128));image.close()
    with (output/'comparison.gif').open('xb') as stream:
        pictures[0].save(infer.CappedOutput(stream,available),format='GIF',save_all=True,append_images=pictures[1:],duration=40,loop=0,optimize=False)
    for p in pictures:p.close()
    files.append('comparison.gif')
    return {'files':[{'name':n,'bytes':(output/n).stat().st_size,'sha256':sha((output/n).read_bytes())} for n in files],
        'bounds_per_row_mm':row_bounds,'projection':'Separate local/device coordinate rows; shared GT/A/B axis within each row and all frames; local wrist origins overlap; no raw RGB'}

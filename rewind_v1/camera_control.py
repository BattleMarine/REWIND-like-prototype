"""Pinned city reference recovery and descriptive camera audit; never changes model input."""

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from rewind_archive.contracts import digest
from rewind_archive.matching import identify
from rewind_m1.local_rgb import _regular_read
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha, validate_pose_geometry, official_splits
from rewind_prepare.extracted import _local, _payload_path, _raw_pose, _Budget, ExtractedLimits
from rewind_prepare.temporal_selection import check_pin, seal
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from . import cross_data as cross


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/v12_camera_control_001.json'
POLICY=json.loads(CONFIG.read_bytes())
RUN_ID=POLICY['run_id']
LIMITS=POLICY['limits']


def source_identity():
    names=('configs/v12_camera_control_001.json','scripts/build_v12_camera_control.py')
    return {**cross.source_identity(),**{n:sha((ROOT/n).read_bytes()) for n in names}}


def context():
    return cross.EvaluationContext(RUN_ID,POLICY['environments'],source_identity)


def _sealed(root,name,schema):
    doc=json.loads(_regular_read(root/name,32*cross.MIB))
    check_pin(doc,schema,doc.get('content_sha256'))
    if doc.get('run_id')!=cross.RUN_ID:raise ValueError('control_reference_run')
    return doc


def checked_reference(root):
    """Validate the complete receipt plus ordered selection/preparation/prediction chain."""
    root=_local(root,directory=True)
    receipt=json.loads(_regular_read(root/'recovery_receipt.json',cross.MIB))
    if receipt.get('schema')!='v12_cross_result_recovery_v1' or receipt.get('run_id')!=cross.RUN_ID:
        raise ValueError('control_reference_receipt')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(receipt['files'])|{'recovery_receipt.json'}:raise ValueError('control_reference_members')
    for name,pin in receipt['files'].items():
        path=_local(root/name)
        if not path.is_relative_to(root) or pin['bytes']>LIMITS['reference_unpacked_bytes']:
            raise ValueError('control_reference_path')
        raw=_regular_read(path,LIMITS['reference_unpacked_bytes'],pin['bytes'])
        if sha(raw)!=pin['sha256']:raise ValueError('control_reference_member_hash')
    plan=_sealed(root,'selection/selection.json','v12_cross_selection_v1')
    prepared=_sealed(root,'prepared/prepared.json','v12_cross_prepared_v1')
    report=_sealed(root,'inference/report.json','v12_cross_report_v1')
    for doc in (plan,prepared,report):
        if (doc['source_sha256']!=POLICY['reference_source_sha256'] or doc['model_zip_sha256']!=cross.MODEL_SHA
                or doc['status']!='complete'):raise ValueError('control_reference_source_model_status')
    if (prepared['selection_sha256']!=plan['content_sha256'] or report['selection_sha256']!=plan['content_sha256']
            or report['prepared_sha256']!=prepared['content_sha256'] or plan['binding']!=prepared['binding']
            or len(plan['clips'])!=4 or len(prepared['clips'])!=4 or len(report['clips'])!=4):
        raise ValueError('control_reference_chain')
    if (report['precision'],report['ddim_steps'],report['seed'],report['noise_offsets'])!=('fp32',10,71,{'body':100,'hand':200}):
        raise ValueError('control_reference_inference_settings')
    expected_epochs={'frontend':9,'body':10,'hand':9}
    for role,epoch in expected_epochs.items():
        if report['model_weights'][role]['epoch']!=epoch:raise ValueError('control_reference_checkpoint')
    clips=[];seen=set();counts=dict.fromkeys(cross.ENVIRONMENTS,0)
    for i,(a,b,c) in enumerate(zip(plan['clips'],prepared['clips'],report['clips'],strict=True)):
        prediction=_sealed(root,f'inference/clip-{i:02d}/predictions.json','v12_cross_clip_predictions_v1')
        manifest=json.loads(_regular_read(root/f'prepared/clip-{i:02d}/manifest.json',32*cross.MIB))
        frames=[r['frame'] for r in a['frames']]
        seq=a['sequence'];env=a['environment']
        if (seq in seen or env not in counts or seq.split('/')[0]!=env or a['split']!='validation'
                or len(frames)!=50 or frames!=list(range(frames[0],frames[0]+50)) or manifest!=b
                or b['number']!=i or any(d['sequence']!=seq for d in (b,c,prediction))
                or any(d['environment']!=env or d['split']!='validation' for d in (b,prediction))
                or [r['frame'] for r in b['frames']]!=frames or prediction['keys']!=[[seq,f] for f in frames]
                or c['content_sha256']!=prediction['content_sha256'] or c['aggregate']!=prediction['aggregate']):
            raise ValueError('control_reference_order_key_binding')
        for selected,observed in zip(a['frames'],b['frames'],strict=True):
            if set(selected['files'])!={'left','right','pose'} or set(observed['files'])!={'left','right','pose'}:
                raise ValueError('control_reference_roles')
            for role in ('left','right','pose'):
                x,y=selected['files'][role],observed['files'][role]
                if (identify(x['path'],'pose' if role=='pose' else 'rgb')!=(seq,selected['frame'],role)
                        or x['size_bytes']!=y['size_bytes'] or not 0<x['size_bytes']<=8*cross.MIB
                        or y['path']!=f"raw/{role}/{selected['frame']:09d}."+('json' if role=='pose' else 'png')
                        or len(y['sha256'])!=64):raise ValueError('control_reference_file_key_hash')
        seen.add(seq);counts[env]+=1
        clips.append({'selection':a,'prepared':b,'prediction':prediction})
    if set(counts.values())!={2}:raise ValueError('control_reference_environment_quota')
    return {'selection':plan,'prepared':prepared,'report':report,'clips':clips}


def load_reference(archive,destination):
    archive=_local(archive)
    if archive.stat().st_size!=POLICY['reference_bytes']:raise ValueError('control_reference_zip_size')
    extract_pinned_zip(archive,POLICY['reference_sha256'],destination,
        max_zip_bytes=LIMITS['reference_zip_bytes'],max_output_bytes=LIMITS['reference_unpacked_bytes'],
        max_members=LIMITS['reference_members'])
    return checked_reference(destination)


def camera_record(native,spec,calibration,sequence,frame,pose_pin):
    residuals=defaultdict(list)
    cameras=validate_pose_geometry(native,spec,calibration,residuals)
    if not residuals['projection_px']:raise ValueError('control_no_projection_checks')
    encoded=encode_camera(cameras,100.)
    centers=[];translations=[];rotations=[]
    for side in ('left','right'):
        r=np.asarray(cameras[side]['rotation_matrix'],dtype=np.float64)
        t=np.asarray(cameras[side]['translation'],dtype=np.float64)
        rotations.append(r);translations.append(t);centers.append(-r.T@t)
    t,c=np.stack(translations),np.stack(centers)
    return {'key':[sequence,frame],'pose':pose_pin,'encoded_camera_18':encoded.flatten().tolist(),
        'rotation_matrix':np.stack(rotations).tolist(),'world_to_camera_translation_mm':t.tolist(),
        'canonical_world_camera_center_mm':c.tolist(),
        'translation_norm_mm':np.linalg.norm(t,axis=1).tolist(),'center_norm_mm':np.linalg.norm(c,axis=1).tolist(),
        'right_minus_left_translation_mm':(t[1]-t[0]).tolist(),
        'right_minus_left_center_mm':(c[1]-c[0]).tolist(),
        'baseline_mm':float(np.linalg.norm(c[1]-c[0])),
        'right_minus_left_encoded_9':(encoded[1]-encoded[0]).tolist(),
        'rotation_difference_frobenius':float(np.linalg.norm(rotations[1]-rotations[0])),
        'geometry_max':{k:max(v) for k,v in residuals.items() if v}}


def recover_cameras(inputs,prepared,reference,pose_root,output,budget):
    """ArchVis cached raw + exactly the pinned 200 city pose files; no city RGB read."""
    _,spec,calibration,binding=cross.assets(inputs)
    if reference['selection']['binding']!=binding:raise ValueError('control_reference_input_binding')
    owners,_=official_splits(Path(inputs)/'split',spec,_Budget(ExtractedLimits(8*cross.MIB,60)))
    clips=[];local_bytes=0;city_before=budget.read_bytes
    prepared=Path(prepared)
    doc=json.loads(_regular_read(prepared/'prepared.json',32*cross.MIB))
    cross.read_sealed(prepared/'prepared.json','v12_cross_prepared_v1',doc['content_sha256'],context=context())
    sources=[('archvis',i,c,None) for i,c in enumerate(doc['clips'])]
    sources += [('city',i,c['prepared'],c) for i,c in enumerate(reference['clips'])]
    for group,number,entry,ref in sources:
        if owners.get(entry['sequence'])!='validation':raise ValueError('control_official_validation_only')
        rows=[]
        for j,row in enumerate(entry['frames']):
            budget.check();pin=row['files']['pose']
            if group=='archvis':
                raw=_regular_read(_local(prepared/f'clip-{number:02d}'/pin['path']),8*cross.MIB,pin['size_bytes'])
                local_bytes+=len(raw)
            else:
                item=ref['selection']['frames'][j]['files']['pose']
                raw=budget.read(_payload_path(pose_root,item['path']),pin['size_bytes'])
            if sha(raw)!=pin['sha256']:raise ValueError('control_pose_hash_mismatch_no_replacement')
            rows.append(camera_record(_raw_pose(raw,row['frame']),spec,calibration,entry['sequence'],row['frame'],pin))
        # Verify the recorded encoded floats are exactly the tensor used for inference.
        if group=='archvis':
            tensor_pin=entry['tensor']
            tensor_raw=_regular_read(prepared/f'clip-{number:02d}/clip.pt',32*cross.MIB,tensor_pin['bytes'])
            if sha(tensor_raw)!=tensor_pin['sha256']:raise ValueError('control_prepared_tensor_hash')
            import io
            tensor=torch.load(io.BytesIO(tensor_raw),weights_only=True,map_location='cpu')['camera']
            if not torch.equal(tensor,torch.tensor([r['encoded_camera_18'] for r in rows]).reshape(-1,2,9)):
                raise ValueError('control_camera_tensor_alignment')
        clips.append({'group':group,'number':number,'sequence':entry['sequence'],'environment':entry['environment'],'frames':rows,
                      'prediction_sha256':ref['prediction']['content_sha256'] if ref else None})
    result=seal({'schema':'v12_camera_records_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
        'reference_zip_sha256':POLICY['reference_sha256'],'reference_source_sha256':POLICY['reference_source_sha256'],
        'reference_revision':POLICY['reference_revision'],'reference_report_sha256':reference['report']['content_sha256'],
        'model_zip_sha256':cross.MODEL_SHA,'prepared_sha256':doc['content_sha256'],'binding':binding,'clips':clips,
        'city_pose_source_bytes':budget.read_bytes-city_before,'archvis_pose_local_reread_bytes':local_bytes,
        'total_preparation_source_bytes':budget.read_bytes,
        'definitions':{'views':['left','right'],'encoded_order':'each view: R column0 xyz, R column1 xyz, t_xyz/100mm; float32',
            'transform':'x_camera_mm=R*x_canonical_world_mm+t_mm',
            'center':'canonical world mm: -R.T@t; not native UE coordinates',
            'differences':'right minus left; baseline is center-difference L2 norm',
            'inference':'original encode_camera100 unchanged; audit never normalizes or replaces inputs'}})
    Path(output).mkdir();cross.write_report(Path(output)/'cameras.json',result);return result


def statistics(values):
    x=np.asarray(values,dtype=np.float64)
    if not x.size or not np.isfinite(x).all():raise ValueError('control_statistics_finite_nonempty')
    return {**{k:fn(x,axis=0).tolist() for k,fn in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]},
        **{f'p{p:02d}':np.percentile(x,p,axis=0,method='linear').tolist() for p in (5,50,95)},'count':len(x)}


def clip_statistics(rows):
    fields=('encoded_camera_18','world_to_camera_translation_mm','canonical_world_camera_center_mm',
            'translation_norm_mm','center_norm_mm','right_minus_left_translation_mm','right_minus_left_center_mm',
            'baseline_mm','right_minus_left_encoded_9','rotation_difference_frobenius')
    return {name:statistics([r[name] for r in rows]) for name in fields}


def observation(clip,prediction,spec):
    if [r['key'] for r in clip['frames']]!=prediction['keys']:raise ValueError('control_comparison_frame_alignment')
    names=spec['joint_names'];idx=[names.index(n) for n in spec['body_joint_names']]
    pred=np.asarray(prediction['prediction']['body']);gt=np.asarray(prediction['gt_target_device_mm'])[:,idx]
    valid=np.asarray(prediction['gt_target_valid'])[:,idx]
    centroid=[];centered_sum=0.;centered_count=0
    for p,g,m in zip(pred,gt,valid,strict=True):
        if m.any():
            pc,gc=p[m].mean(0),g[m].mean(0)
            centroid.append(float(np.linalg.norm(pc-gc)))
            centered_sum+=float(np.linalg.norm((p[m]-pc)-(g[m]-gc),axis=-1).sum())
            centered_count+=int(m.sum())
    agg=prediction['aggregate']
    return {'sequence':clip['sequence'],'environment':clip['environment'],'frames':len(clip['frames']),
        'prediction_sha256':prediction['content_sha256'],'body_centroid_bias_mean_mm':float(np.mean(centroid)) if centroid else None,
        'centroid_valid_frames':len(centroid),'centered_body_error_sum_mm':centered_sum,
        'centered_body_valid_joints':centered_count,
        'centered_body_error_mean_mm':centered_sum/centered_count if centered_count else None,
        'centered_error_definition':'per-frame valid body-joint centroids removed separately; joint-weighted L2; no rotation/scale alignment; diagnostic, not official MPJPE',
        'metrics':agg,'camera_summary':clip_statistics(clip['frames']),
        'interpretation':'observational only; camera rotation, RGB appearance, motion and input quality also vary'}


def compare(inputs,cameras,inference,reference,output,budget,*,inference_report):
    _,spec,_,_=cross.assets(inputs);output=Path(output);output.mkdir()
    check_pin(cameras,'v12_camera_records_v1',cameras['content_sha256'])
    if (cameras['run_id']!=RUN_ID or cameras['source_sha256']!=digest(source_identity())
            or cameras['reference_zip_sha256']!=POLICY['reference_sha256']
            or cameras['reference_report_sha256']!=reference['report']['content_sha256']
            or cameras['model_zip_sha256']!=cross.MODEL_SHA
            or cameras['prepared_sha256']!=inference_report['prepared_sha256']
            or cameras['binding']!=reference['prepared']['binding']):raise ValueError('control_camera_comparison_binding')
    check_pin(inference_report,'v12_cross_report_v1',inference_report['content_sha256'])
    if (inference_report['run_id']!=RUN_ID or inference_report['source_sha256']!=digest(source_identity())
            or inference_report['model_zip_sha256']!=cross.MODEL_SHA):raise ValueError('control_inference_comparison_binding')
    archvis=[c for c in cameras['clips'] if c['group']=='archvis']
    cities=[c for c in cameras['clips'] if c['group']=='city']
    if ([c['sequence'] for c in archvis]!=[c['sequence'] for c in inference_report['clips']]
            or [c['sequence'] for c in cities]!=[c['prepared']['sequence'] for c in reference['clips']]):
        raise ValueError('control_camera_clip_membership')
    rows=[]
    for clip in cameras['clips']:
        budget.check()
        if clip['group']=='city':pred=reference['clips'][clip['number']]['prediction']
        else:
            pred=json.loads(_regular_read(Path(inference)/f"clip-{clip['number']:02d}/predictions.json",8*cross.MIB))
            check_pin(pred,'v12_cross_clip_predictions_v1',pred['content_sha256'])
            if (pred['run_id']!=RUN_ID or pred['content_sha256']!=inference_report['clips'][clip['number']]['content_sha256']):
                raise ValueError('control_prediction_run_or_pin')
        rows.append(observation(clip,pred,spec))
    environments=sorted({c['environment'] for c in cameras['clips']})
    by_environment={env:clip_statistics([r for c in cameras['clips'] if c['environment']==env for r in c['frames']]) for env in environments}
    baseline=np.asarray([r['encoded_camera_18'] for c in cameras['clips'] if c['group']=='archvis' for r in c['frames']])
    outside={env:((np.asarray([r['encoded_camera_18'] for c in cameras['clips'] if c['environment']==env for r in c['frames']])<baseline.min(0))|
                  (np.asarray([r['encoded_camera_18'] for c in cameras['clips'] if c['environment']==env for r in c['frames']])>baseline.max(0))).mean(0).tolist() for env in environments}
    artifact=distribution_plot(cameras,output/'camera_distributions.png',budget)
    result=seal({'schema':'v12_camera_comparison_v1','run_id':RUN_ID,'source_sha256':digest(source_identity()),
        'model_zip_sha256':cross.MODEL_SHA,'reference_zip_sha256':POLICY['reference_sha256'],
        'cameras_sha256':cameras['content_sha256'],'by_environment':by_environment,'by_clip':rows,
        'status':inference_report['status'],'archvis_clips':len(archvis),
        'archvis_frames':sum(len(c['frames']) for c in archvis),'target_archvis_frames':100,
        'prepared_sha256':inference_report['prepared_sha256'],'inference_report_sha256':inference_report['content_sha256'],
        'encoded_outside_archvis_sample_minmax_fraction':outside,'artifacts':[artifact],
        'statistics_definition':'population std ddof0; percentiles NumPy linear interpolation over frames',
        'limits':'ArchVis validation sample only, not full training distribution or OOD proof; observations not causal; name proxy only, nominal25FPS; timestamps/stereo/motion independence unverified'})
    cross.write_report(output/'report.json',result)
    table=['# 카메라 조건과 실패 형태의 관찰 비교','',
        f"ArchVis 확보: {result['archvis_frames']}/100frame, {result['archvis_clips']}/2clip; {result['status']}",'',
        '| 환경 / clip | frame | t norm L/R mm | 중심 오차 mm | 중심 제거 진단 mm | body MPJPE mm | bone 예측/GT mm | spread 예측/GT mm |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | --- |']
    for row in rows:
        shape=row['metrics']['body_shape_sums'];tn=row['camera_summary']['translation_norm_mm']['mean']
        def number(value):return 'N/A' if value is None else f'{value:.2f}'
        table.append('| '+ ' | '.join([row['environment']+'/'+row['sequence'].split('/')[-1],str(row['frames']),
            '/'.join(map(number,tn)),number(row['body_centroid_bias_mean_mm']),number(row['centered_body_error_mean_mm']),
            number(row['metrics']['body']['mean']),'/'.join(number(shape[k]) for k in ('predicted_mean_bone_mm','target_mean_bone_mm')),
            '/'.join(number(shape[k]) for k in ('predicted_mean_spread_mm','target_mean_spread_mm'))])+' |')
    table += ['', '관찰 비교이며 인과 증명이 아닙니다. 회전·RGB·motion·2D 품질 차이가 혼재합니다. 중심 제거 값은 회전/scale 정렬 없는 별도 진단으로 공식 MPJPE를 대체하지 않습니다.',
        'ArchVis 평가 표본의 min/max 범위를 전체 train 분포나 OOD 판정으로 해석하지 않습니다. 명목25FPS·stereo 동기·원본 motion 독립성 미검증이며 이름은 proxy입니다.',
        '카메라 center는 canonical world(mm)의 -Rᵀt이며 native UE 좌표가 아닙니다. 원래 encode_camera100 입력을 그대로 사용했습니다.']
    (output/'comparison.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    return result


def distribution_plot(cameras,path,budget):
    # Pillow is already required by inference; no plotting dependency is added.
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(1600,1600),'white');draw=ImageDraw.Draw(image)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    envs=sorted({c['environment'] for c in cameras['clips']});colors=['#1765b3','#cc512d','#32824b']
    draw.text((25,12),'Camera conditions: frame min / p05 / median / p95 / max; each environment has up to 100 frames',fill='black',font=font)
    for j,env in enumerate(envs):draw.text((25+520*j,45),env,fill=colors[j],font=font)
    plots=[(f'encoded {side} Rcol{col} {axis}',lambda r,k=k:r['encoded_camera_18'][k])
           for side,base in [('L',0),('R',9)] for col in range(2) for axis,k in zip('xyz',range(base+col*3,base+col*3+3))]
    plots += [(f'encoded {side} t{axis}/100mm',lambda r,k=k:r['encoded_camera_18'][k]) for side,base in [('L',6),('R',15)] for axis,k in zip('xyz',range(base,base+3))]
    plots += [(f'{side} translation norm (mm)',lambda r,k=k:r['translation_norm_mm'][k]) for k,side in enumerate(('L','R'))]
    plots += [(f'{side} canonical center norm (mm)',lambda r,k=k:r['center_norm_mm'][k]) for k,side in enumerate(('L','R'))]
    plots += [('stereo baseline (mm)',lambda r:r['baseline_mm']),('rotation L-R Frobenius',lambda r:r['rotation_difference_frobenius'])]
    for i,(label,get) in enumerate(plots):
        budget.check();x=25+(i%3)*525;y=95+(i//3)*185
        values=[[get(r) for c in cameras['clips'] if c['environment']==env for r in c['frames']] for env in envs]
        lo=min(map(min,values));hi=max(map(max,values));span=hi-lo or 1
        draw.text((x,y),label,fill='black',font=font)
        draw.text((x,y+22),f'range {lo:.3g} .. {hi:.3g}',fill='#555555',font=font)
        for j,v in enumerate(values):
            q=np.percentile(v,[0,5,50,95,100],method='linear');p=[x+15+450*(a-lo)/span for a in q];yy=y+65+j*25
            draw.line((p[0],yy,p[4],yy),fill=colors[j],width=2)
            draw.line((p[1],yy,p[3],yy),fill=colors[j],width=7)
            draw.ellipse((p[2]-4,yy-5,p[2]+4,yy+5),fill=colors[j])
    draw.text((25,1580),'Descriptive only: validation sample range; not training coverage, OOD proof or causal attribution.',fill='black',font=font)
    image.save(path);image.close()
    raw=path.read_bytes();return {'name':path.name,'bytes':len(raw),'sha256':sha(raw)}

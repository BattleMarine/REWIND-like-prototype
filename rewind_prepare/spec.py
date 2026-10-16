"""새학습규격을 검증한다. 기존 개발용·합성 설정을 수정하지 않는다."""
import copy
import re


def validate_spec(spec):
    required={'schema_version','id','preprocessing_version','adopted_date','status','native_rig_parent_verified',
        'joint_names','body_joint_names','upper_joint_names','hands','wrists','evaluation16','edges','coordinates','pinned_sources','limits'}
    if not isinstance(spec,dict) or set(spec)!=required or spec['schema_version']!='real_pose_spec_v1' or spec['status']!='project_training_assumption_adopted' or spec['native_rig_parent_verified'] is not False:
        raise ValueError('spec_schema_status')
    for key in ('id','preprocessing_version','adopted_date'):
        if not isinstance(spec[key],str) or not spec[key].strip():raise ValueError('spec_identity')
    groups={key:spec[key] for key in ('joint_names','body_joint_names','upper_joint_names','evaluation16')}
    for names in groups.values():
        if not isinstance(names,list) or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):raise ValueError('spec_joint_order')
    names,body,upper,evaluation=(groups[k] for k in ('joint_names','body_joint_names','upper_joint_names','evaluation16'))
    if list(map(len,(names,body,upper,evaluation)))!=[62,22,13,16] or not set(upper)<=set(body) or not set(evaluation)<=set(body):raise ValueError('spec_joint_subsets')
    if not isinstance(spec['hands'],dict) or set(spec['hands'])!={'left','right'} or spec['wrists']!={'left':'hand_l','right':'hand_r'}:raise ValueError('spec_hands_views')
    fingers=[]
    for side in ('left','right'):
        hand=spec['hands'][side];wrist=spec['wrists'][side]
        if not isinstance(hand,list) or len(hand)!=21 or len(set(hand))!=21 or hand[0]!=wrist or wrist not in body or wrist not in upper or any(n in body for n in hand[1:]):raise ValueError('spec_wrist_ownership')
        fingers+=hand[1:]
    if len(set(fingers))!=40 or names!=body+fingers:raise ValueError('spec_unique_order')
    seen=set();adjacency={n:set() for n in names}
    if not isinstance(spec['edges'],list) or len(spec['edges'])!=61:raise ValueError('spec_graph')
    for edge in spec['edges']:
        if not isinstance(edge,list) or len(edge)!=2 or any(n not in names for n in edge) or edge[0]==edge[1] or tuple(sorted(edge)) in seen:raise ValueError('spec_edge')
        seen.add(tuple(sorted(edge)));a,b=edge;adjacency[a].add(b);adjacency[b].add(a)
    visited=set();pending=[names[0]]
    while pending:
        n=pending.pop()
        if n in visited:continue
        visited.add(n);pending+=list(adjacency[n]-visited)
    if len(visited)!=62:raise ValueError('spec_graph_disconnected')
    c=spec['coordinates']
    fixed={'raw_unit':'cm','output_unit':'mm','mm_per_raw_unit':10.0,'unit_evidence_status':'adopted_metric_anchor_inference',
        'transform':'reflected_world_to_camera_v1','views':['left','right'],'image_sizes':[[1024,1024],[1024,1024]],'nominal_fps':25,'time_seconds_policy':'do_not_infer'}
    if not isinstance(c,dict) or any(c.get(k)!=v or isinstance(c.get(k),bool) for k,v in fixed.items()):raise ValueError('spec_coordinate_convention')
    if set(c)!=set(fixed)|{'unit_source','baseline_raw_range','raw_consistency_tolerance','projection_tolerance_px'} or not isinstance(c['unit_source'],str) or not c['unit_source'].startswith('https://'):
        raise ValueError('spec_coordinate_fields')
    if c['baseline_raw_range']!=[11.98,12.02] or c['raw_consistency_tolerance']!=1e-6 or c['projection_tolerance_px']!=.001:raise ValueError('spec_numeric_tolerances')
    pins=spec['pinned_sources']
    if not isinstance(pins,dict) or set(pins)!={'scaramuzza_sha256','split_sha256'} or not isinstance(pins['split_sha256'],dict) or set(pins['split_sha256'])!={'train','validation','test'}:raise ValueError('spec_source_pins')
    if any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in [pins['scaramuzza_sha256'],*pins['split_sha256'].values()]):raise ValueError('spec_source_hash')
    limits=spec['limits']
    maxima={'max_files':256,'max_file_bytes':4194304,'max_total_input_bytes':33554432,'max_output_bytes':67108864,'max_elapsed_seconds':30}
    if not isinstance(limits,dict) or set(limits)!=set(maxima) or any(type(limits[k]) is not int or not 0<limits[k]<=maxima[k] for k in maxima):raise ValueError('spec_limits')
    return copy.deepcopy(spec)

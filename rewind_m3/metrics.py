"""Body·hand global/local·고유 whole-body·wrist 오차를 분리한다."""
import torch
from rewind_m1.metrics import metric_sums


def errors(prediction,target,mask):
    result = metric_sums(prediction,target,mask)
    count = result['valid_joints']
    return {'mpjpe_mm':result['distance_sum_mm']/count if count else None,'valid_joints':count}


def cascade_metrics(body,local,global_fingers,data,roles,fingers):
    body_gt,body_mask = data.body_target_mm[:,-1],data.body_target_valid[:,-1]
    global_gt,global_mask = data.global_target_mm[:,-1],data.global_target_valid[:,-1]
    local_gt,local_mask = data.target_mm[:,-1],data.target_valid[:,-1]
    result = {'body':errors(body,body_gt,body_mask),'hand_global':errors(global_fingers,global_gt,global_mask),
              'hand_wrist_relative':errors(local,local_gt,local_mask),
              'whole_body_unique':errors(torch.cat((body,global_fingers),dim=1),torch.cat((body_gt,global_gt),dim=1),torch.cat((body_mask,global_mask),dim=1))}
    for side in ('left','right'):
        indices = [fingers.index(n) for n in roles['hands'][side][1:]]
        wrist = roles['body_joint_names'].index(roles['wrists'][side])
        result[side+'_hand_global'] = errors(global_fingers[:,indices],global_gt[:,indices],global_mask[:,indices])
        result[side+'_hand_wrist_relative'] = errors(local[:,indices],local_gt[:,indices],local_mask[:,indices])
        result[side+'_body_wrist'] = errors(body[:,wrist:wrist+1],body_gt[:,wrist:wrist+1],body_mask[:,wrist:wrist+1])
    result['body_wrist_attachment_error_mm'] = 0.0
    result['joint_names_unique'] = roles['body_joint_names']+fingers
    return result

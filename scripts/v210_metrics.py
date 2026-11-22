"""FP32 official MPJPE and FP64 wrist/finger error decomposition for v2.1.0."""

from collections import defaultdict

import numpy as np
import torch
from rewind_m3.cascade import UpperCondition
from rewind_v1.data import attach_predicted_hands, hand_clip


SIDES = ("left", "right")


def rows(stages, body, hand, clip, key, ledger):
    ledger.check()
    body_pred = body.predict(clip.select_joints(stages.body_idx), stages._noise(key, len(stages.body_names), 100)).cpu()
    upper = body_pred[..., stages.upper_idx, :]
    condition = UpperCondition(upper, clip.time_valid[..., None].expand(upper.shape[:-1]),
                               "predicted_frozen_body", body.checkpoint_sha256)
    local_clip = hand_clip(clip, stages.spec)
    ledger.check()
    local = hand.predict(local_clip, stages._noise(key, len(stages.fingers), 200), upper=condition).cpu()
    global_hand = attach_predicted_hands(body_pred, local, stages.spec)
    n = int(clip.time_valid.sum())
    if not 1 <= n <= 50 or not torch.equal(clip.time_valid, torch.arange(50)[None] < n):
        raise ValueError("v210_time_mask")
    names = stages.body_names + stages.fingers
    indices = stages.body_idx + stages.finger_idx
    result = {
        "frame_index": np.arange(key[1], key[1]+n, dtype=np.int64),
        "sequence_id": np.repeat(key[0], n), "clip_start": np.full(n, key[1], dtype=np.int64),
        "pred_device_mm": torch.cat((body_pred, global_hand), dim=2)[0,:n].detach().numpy(),
        "target_device_mm": clip.target_mm[0,:n,indices,:].numpy(), "target_valid": clip.target_valid[0,:n,indices].numpy(),
        "hand_local_pred_mm": local[0,:n].detach().numpy(), "hand_local_target_mm": local_clip.target_mm[0,:n].numpy(),
        "hand_local_valid": local_clip.target_valid[0,:n].numpy(), "seconds": clip.seconds[0,:n].numpy(),
    }
    if len(names) != len(set(names)) or result["target_device_mm"].shape != (n,len(names),3):
        raise ValueError("v210_joint_layout")
    for value in result.values():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("v210_nonfinite_prediction")
    return result


def official(pred, target, mask):
    x, y = torch.from_numpy(pred), torch.from_numpy(target)
    distance = torch.linalg.vector_norm((x-y)[torch.from_numpy(mask)], dim=-1)
    return float(distance.double().sum()), int(mask.sum())


def decompose(wrist, relative, observed_global):
    wrist, relative, observed_global = (v.astype(np.float64) for v in (wrist, relative, observed_global))
    w2 = np.square(wrist).sum(-1)
    r2 = np.square(relative).sum(-1)
    g2 = np.square(observed_global).sum(-1)
    cross = 2*(wrist*relative).sum(-1)
    return w2, r2, g2, cross, g2-w2-r2-cross


def sums(data, spec):
    pred, gt, valid = (data[k] for k in ("pred_device_mm", "target_device_mm", "target_valid"))
    if pred.dtype != np.float32 or gt.dtype != np.float32 or valid.dtype != np.bool_ or pred.shape != gt.shape or pred.shape[:-1] != valid.shape:
        raise ValueError("v210_global_arrays")
    local, local_gt, local_valid = (data[k] for k in ("hand_local_pred_mm", "hand_local_target_mm", "hand_local_valid"))
    body_names = spec["body_joint_names"]
    b = len(body_names)
    result = {"frames": len(pred), "official": {"body": official(pred[:,:b],gt[:,:b],valid[:,:b]),
              "hand_global": official(pred[:,b:],gt[:,b:],valid[:,b:]), "hand_local": official(local,local_gt,local_valid)}, "sides": {}}
    start = 0
    for side in SIDES:
        n = len(spec["hands"][side])-1
        wrist_index = body_names.index(spec["wrists"][side])
        w_valid = valid[:,wrist_index]
        sl = slice(start,start+n)
        fingers = slice(b+start,b+start+n)
        both = local_valid[:,sl]
        if not np.array_equal(both, valid[:,fingers] & w_valid[:,None]):
            raise ValueError("v210_common_mask")
        wrist_error = np.broadcast_to((pred[:,wrist_index].astype(np.float64)-gt[:,wrist_index])[:,None], local[:,sl].shape)[both]
        relative = local[:,sl][both].astype(np.float64)-local_gt[:,sl][both]
        global_error = pred[:,fingers][both].astype(np.float64)-gt[:,fingers][both]
        w2,r2,g2,cross,residual = decompose(wrist_error,relative,global_error)
        result["sides"][side] = {
            "wrist": official(pred[:,wrist_index], gt[:,wrist_index], w_valid),
            "local": official(local[:,sl],local_gt[:,sl],both),
            "global": official(pred[:,fingers],gt[:,fingers],valid[:,fingers]),
            "common_global": official(pred[:,fingers],gt[:,fingers],both), "common_count": int(both.sum()),
            "terms": {"wrist_mse_mm2": float(w2.sum()), "relative_mse_mm2": float(r2.sum()),
                      "global_mse_mm2": float(g2.sum()), "cross_mm2": float(cross.sum()),
                      "closure_mean_mm2": float(residual.sum())},
            "closure_max_abs_mm2": float(np.abs(residual).max(initial=0.)),
        }
        start += n
    return result


def merge(parts):
    def pool(values):
        total = sum(v[0] for v in values)
        count = sum(v[1] for v in values)
        return {"mpjpe_mm": total/count if count else None,"valid_joints":count}

    result = {"clips":len(parts), "frames":sum(p["frames"] for p in parts), "official":{}, "sides":{}}
    for role in ("body","hand_global","hand_local"):
        result["official"][role] = pool([p["official"][role] for p in parts])
    for side in SIDES:
        items = [p["sides"][side] for p in parts]
        n = sum(p["common_count"] for p in items)
        result["sides"][side] = {k:pool([p[k] for p in items]) for k in ("wrist","local","global","common_global")}
        result["sides"][side].update(common_count=n, terms={k:sum(p["terms"][k] for p in items)/n if n else None
                      for k in items[0]["terms"]}, closure_max_abs_mm2=max(p["closure_max_abs_mm2"] for p in items))
    return result


def aggregate(segments, spec, reference):
    return aggregate_scores(((key, sums(arrays, spec)) for key, arrays in segments), reference)


def aggregate_scores(segments, reference):
    by_scene = defaultdict(list)
    for key, scores in segments:
        by_scene[key[0].split("/",1)[0]].append(scores)
    scores = {name:merge(parts) for name,parts in sorted(by_scene.items())}
    def equal_weight(values):
        values = list(values)
        return sum(values)/len(values) if values and all(x is not None for x in values) else None
    six = [scores[s]["official"][k]["mpjpe_mm"] for s in scores for k in ("body","hand_global")]
    macro = equal_weight(six)
    sides = {}
    for side in SIDES:
        group = [s["sides"][side] for s in scores.values()]
        sides[side] = {k:equal_weight(a[k]["mpjpe_mm"] for a in group) for k in ("wrist","local","global","common_global")}
        sides[side]["terms"] = {k:equal_weight(a["terms"][k] for a in group) for k in group[0]["terms"]}
    delta = {s:{k:{"official_mm":reference["scores"][s][k+"_mpjpe_mm"],
                  "delta_mm":scores[s]["official"][k]["mpjpe_mm"]-reference["scores"][s][k+"_mpjpe_mm"]}
                for k in ("body","hand_global")} for s in scores}
    pooled = merge([p for parts in by_scene.values() for p in parts])
    prior = {"macro_mm":reference["macro_score_mm"], "macro_delta_mm":macro-reference["macro_score_mm"], "per_scenario":delta}
    if "hand_local_baseline" in reference:
        def difference(value, original):
            return {"official_mm":original["mpjpe_mm"], "official_valid_joints":original["valid_joints"],
                    "delta_mm":value["mpjpe_mm"]-original["mpjpe_mm"],
                    "valid_joint_delta":value["valid_joints"]-original["valid_joints"]}
        previous = reference["hand_local_baseline"]
        for scenario in scores:
            delta[scenario]["hand_local"] = difference(scores[scenario]["official"]["hand_local"],previous["per_scenario"][scenario])
        prior["pooled_hand_local"] = difference(pooled["official"]["hand_local"],previous["pooled"])
    return {"per_scenario":scores, "pooled":pooled, "macro_score_mm":macro,
            "sides_macro_equal_scenario":sides,"prior_comparison":prior}

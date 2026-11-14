"""Pinned v1.2 RGB predictions on local manifest members, with explicit independent T=1 scope."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha, validate_pose_geometry
from rewind_prepare.extracted import _local, _pose_targets
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_selection import seal
from rewind_v1.full_contract import AUDIT_ZIP_SHA, MANIFEST_SHA, RUN_ID, load_audit
from scripts import visualize_archvis_v1 as view


def _metrics(pred2d, pred3d, fields, spec):
    errors, inside = [], []
    for predicted, field in zip(pred2d, fields[:2]):
        for xy, target, valid in zip(predicted, field["coordinates"], field["valid_mask"]):
            if valid:
                error = sum((a-b)**2 for a,b in zip(xy,target))**.5
                errors.append(error)
                if all(0 <= value <= 1023 for value in target): inside.append(error)
    result = {"2d_valid_joint_views": len(errors), "2d_valid_mean_native_px": sum(errors)/len(errors) if errors else None,
              "2d_inside_joint_views": len(inside), "2d_inside_mean_native_px": sum(inside)/len(inside) if inside else None}
    for role, names in (("body",spec["body_joint_names"]),("hand_global",view._group_names(spec,"hand"))):
        values = []
        for name in names:
            i = spec["joint_names"].index(name)
            if fields[2]["valid_mask"][i]:
                values.append(sum((a-b)**2 for a,b in zip(pred3d[i],fields[2]["coordinates"][i]))**.5)
        result[role] = {"mpjpe_mm":sum(values)/len(values) if values else None,"valid_joints":len(values)}
    return result


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    if not 1 <= len(args.frames) <= 12 or sorted(set(args.frames)) != args.frames or args.frames[0] < 0:
        raise ValueError("v12_view_bounded_independent_frames")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("v12_view_new_output")
    torch.set_num_threads(1)
    manifest, _, _ = load_audit(args.audit)
    rows = {(r["sequence_id"],r["frame_index"]):r for r in manifest["records"]}
    keys = [(args.sequence,frame) for frame in args.frames]
    if any(key not in rows for key in keys):
        raise ValueError("v12_view_fixed_manifest_membership")
    spec_raw = _regular_read(_local(args.spec),view.MIB)
    if sha(spec_raw) != view.SPEC_SHA or manifest["provenance"]["spec_sha256"] != view.SPEC_SHA:
        raise ValueError("v12_view_spec_pin")
    spec = validate_spec(json.loads(spec_raw))
    inputs = view._prepared_sample(args.prepared)
    calibration_raw = _regular_read(_local(args.prepared/"metadata/scaramuzza.json"),view.MIB)
    if sha(calibration_raw) != spec["pinned_sources"]["scaramuzza_sha256"]:
        raise ValueError("v12_view_calibration_pin")
    calibration = json.loads(calibration_raw)
    items, cameras = [], []
    for key in keys:
        paths = {role:index[key][0] for role,index in inputs.items()}
        raw, doc, sources = view._load_frame(paths,key[1],rows[key]["files"],spec)
        residuals = defaultdict(list)
        camera = validate_pose_geometry(doc,spec,calibration,residuals)
        if not residuals["projection_px"]:
            raise ValueError("v12_view_no_geometry_checks")
        items.append((raw,_pose_targets(doc,spec),sources,{k:max(v) for k,v in residuals.items() if v}))
        cameras.append({"camera_pose":camera})
    models,pins,report,implementation = view._models(args.training,spec,True,training_sha256=args.training_sha256,run_id=RUN_ID)
    images = torch.stack([view.frame_image(raw,report["config"]["frontend"]["size"]) for raw,_,_,_ in items])
    pred2d,confidence,pred3d = view._predict(models,images,spec,cameras,keys,report["config"],with_teachers=True,independent=True)
    baseline = None
    if args.compare_training:
        previous,previous_pins,previous_report,previous_impl = view._models(args.compare_training,spec,True,
            training_sha256=args.compare_training_sha256,run_id=view.RUN_ID)
        if previous_report["config"]["frontend"]["size"] != report["config"]["frontend"]["size"]:
            raise ValueError("v12_view_comparison_preprocessing")
        old2d,oldconfidence,old3d = view._predict(previous,images,spec,cameras,keys,previous_report["config"],
                                                with_teachers=True,independent=True)
        oldrows,_ = view._selected_inputs(args.baseline_selection)
        baseline = {"run_id":view.RUN_ID,"training_sha256":args.compare_training_sha256,"model_weights":previous_pins,
                    "fixed_selection_membership":[key in oldrows for key in keys],"inference_implementation_sha256":previous_impl}
    args.output.mkdir()
    frames,artifacts = [],[]
    for i,(key,(raw,fields,sources,geometry)) in enumerate(zip(keys,items)):
        gt2d,gt3d = [f["coordinates"] for f in fields[:2]],fields[2]["coordinates"]
        scope = f"v1.2 {rows[key]['split'].upper()} sample / independent T=1"
        image = view.panel(raw,pred2d[i],gt2d,pred3d[i],gt3d,spec,key[0],key[1],scope,"REWIND v1.2")
        name = f"frame-{key[1]:06d}.png";image.save(args.output/name)
        frame = {"frame":key[1],"split":rows[key]["split"],"sources":sources,"geometry_max_residuals":geometry,
                 "predicted_2d_native":pred2d[i],"confidence":confidence[i],"predicted_3d_device_mm":pred3d[i],
                 "gt_2d_native":gt2d,"gt_2d_valid":[f["valid_mask"] for f in fields[:2]],
                 "gt_3d_device_mm":gt3d,"gt_3d_valid":fields[2]["valid_mask"],
                 "metrics":_metrics(pred2d[i],pred3d[i],fields,spec),"panel":name}
        names = [name]
        if baseline is not None:
            frame["baseline"] = {"predicted_2d_native":old2d[i],"confidence":oldconfidence[i],
                                  "predicted_3d_device_mm":old3d[i],"metrics":_metrics(old2d[i],old3d[i],fields,spec)}
            label = lambda version,p:version+" best "+" ".join(f"{r[0].upper()}{p[r]['epoch']}" for r in ("frontend","body","hand"))
            compared = view.comparison_panel(raw,old2d[i],pred2d[i],gt2d,old3d[i],pred3d[i],gt3d,spec,key[1],{},scope,
                model_labels=(label("v1.1",previous_pins),label("v1.2",pins)))
            name = f"comparison-frame-{key[1]:06d}.png";compared.save(args.output/name);names.append(name)
            frame["comparison_panel"] = name
        for name in names:
            data = (args.output/name).read_bytes();artifacts.append({"name":name,"bytes":len(data),"sha256":sha(data)})
        frames.append(frame)
    result = seal({"schema":"archvis_v12_visualization_v1","run_id":RUN_ID,"training_sha256":args.training_sha256,
        "manifest_sha256":MANIFEST_SHA,"audit_zip_sha256":AUDIT_ZIP_SHA,"spec_sha256":view.SPEC_SHA,
        "sequence":args.sequence,"split":rows[keys[0]]["split"],"joint_names":spec["joint_names"],
        "sample_selection":{"frames":args.frames,"basis":args.selection_reason},
        "scope":f"Local {rows[keys[0]]['split']} samples / independent T=1; no representative/generalization claim",
        "model_weights":pins,"inference_implementation_sha256":implementation,"baseline":baseline,
        "visualization_script_sha256":sha(Path(__file__).read_bytes()),
        "render_and_inference_helper_sha256":sha(Path(view.__file__).read_bytes()),
        "preprocessing":{"frontend_size":report["config"]["frontend"]["size"],"native_size":[1024,1024],"resize":"bilinear_align_corners"},
        "inference":{"device":"cpu","torch":str(torch.__version__),"threads":torch.get_num_threads(),
            "seed":report["config"]["seed"],"keyed_noise_offsets":{"body":100,"hand":200},
            "ddim_steps":report["config"]["teacher"]["sampling_steps"],"precision":"fp32",
            "independent_frame_reset":True,"context_frames":1,"full_50_frame_clip":False,"nominal_fps":25,
            "clock_verified":False,"hand_upper":"predicted_body","body_condition":"predicted_2d_and_camera_only",
            "ground_truth_usage":"comparison_and_geometry_audit_only; no GT joint model conditioning",
            "camera_pose_source":"local pose JSON camera metadata; not RGB-only camera estimation"},
        "elapsed_seconds":time.monotonic()-started,"frames":frames,"artifacts":artifacts})
    (args.output/"predictions.json").write_bytes(canonical(result)+b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training","audit","spec","prepared","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--training-sha256",required=True)
    parser.add_argument("--sequence",required=True)
    parser.add_argument("--frames",type=int,nargs="+",required=True)
    parser.add_argument("--selection-reason",default="Caller-selected local availability, fixed before inference")
    parser.add_argument("--compare-training",type=Path)
    parser.add_argument("--compare-training-sha256")
    parser.add_argument("--baseline-selection",type=Path)
    args = parser.parse_args()
    if args.compare_training and (not args.compare_training_sha256 or not args.baseline_selection):
        parser.error("comparison needs its fixed ZIP SHA and selection")
    result = run(args)
    print(canonical({"output":str(args.output),"content_sha256":result["content_sha256"],
                     "frames":[f["frame"] for f in result["frames"]],"split":result["split"]}).decode())


if __name__ == "__main__":main()

"""Fixed v2.1.0 CUDA evaluation, with independently verified per-group resume."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch
from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_m1.training import seed_run
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import check_pin
from rewind_v1 import v2_retention as retention
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from rewind_v1.v2_budget import Ledger
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import writer
from scripts.run_v2_gpu_training import identity_for, load_inputs, read_journal, training_config
from scripts.v210_metrics import aggregate, rows
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullCache, FullData, Training
from scripts.v2_gpu_training_storage import EVAL_SEED, GIB, MIB, publish_json, read_sealed, transfer_in


A = "reports/v2_gpu_training_128_v1/A-seed-42"
DESTINATION = "reports/v210_eval128_a_v1"
DIAGNOSTIC = f"{A}/diagnostics/invocation-roh3t3jf.json"
RECEIPT = f"{A}/checkpoints/step-000009.json"
COMPARE = f"{A}/comparison.json"
FEATURE = "9588afef53e4802089baf7c51235f18068d110d02b1cac8ca6371b8d7aa5e276"
FINAL = "7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4"
GROUP_SCHEMA = "rewind_v210_prediction_group_v1"
SUMMARY_SCHEMA = "rewind_v210_completed_v1"
CAP = 16*MIB
DEADLINE = 28800


def original_json(project, name, pinned, schema):
    path = project/name
    if path.resolve(strict=True) != path or not path.is_relative_to(project):
        raise ValueError("v210_input_path")
    raw = _regular_read(path,192*1024,pinned["bytes"])
    if hashlib.sha256(raw).hexdigest() != pinned["sha256"]:
        raise ValueError(f"v210_original_sha256: {name}")
    doc = json.loads(raw)
    check_pin(doc,schema,doc["content_sha256"])
    return doc


def input_file(folder, name):
    path = folder/name
    if Path(name).name != name or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f"v210_original_file: {name}")
    return path


def reference_scores(diagnostic, comparison):
    cursor = diagnostic["checkpoint_cursor"]
    [prior] = [r["validation"] for r in cursor["history"] if r["stage"]=="hand" and r["epoch"]==cursor["best"]["hand"]["epoch"]]
    return {**comparison,"hand_local_baseline":{"pooled":prior["joint_metrics"]["hand_local"],
           "per_scenario":{s:v["joint_metrics"]["hand_local"] for s,v in prior["per_scenario_reference"].items()}}}


def group_keys(data, ordinal):
    all_clips = [(split,key) for split, keys in data.clips.items() for key in keys]
    return all_clips[ordinal*24:ordinal*24+24]


def feature_group(project, pins, store, group, ledger, scratch):
    directory = project/A/"features"/FEATURE
    tag = f"group-{group:05d}"
    marker_path = f"{A}/features/{FEATURE}/{tag}.json"
    expected = pins["groups"][str(group)]
    source = original_json(project,marker_path,expected["marker"],"rewind_v2_gpu_features_group_v1")
    items = group_keys(store.data,group)
    folders = [store.names[s,key] for s,key in items]
    if source["identity"] != FEATURE or source["folders"] != folders or source["archive"] != expected["archive"]:
        raise ValueError("v210_fixed_feature_group")
    package = source["archive"]
    extracted = scratch/"features"
    local = scratch/"features.zip"
    ledger.reserve("read",package["bytes"]+1)
    transfer_in(input_file(directory,package["name"]),local,package,ledger)
    extract_pinned_zip(local,package["sha256"],extracted,max_zip_bytes=32*MIB,max_output_bytes=32*MIB,max_members=48)
    if set(p.name for p in extracted.iterdir()) != set(folders):
        raise ValueError("v210_feature_group_folders")
    store.root = extracted
    return [key for split,key in items if split=="validation"]


def expected_frames(keys, data):
    return [(k[0],k[1],f) for k in keys for f in range(k[1],k[1]+data.real_frames(k,"validation"))]


def segments(raw, keys, data, names):
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    frame_rows = list(zip(values["sequence_id"].tolist(),values["clip_start"].tolist(),values["frame_index"].tolist(),strict=True))
    if frame_rows != expected_frames(keys,data) or values.pop("joint_names").tolist()!=names:
        raise ValueError("v210_prediction_frame_order")
    offset = 0
    for key in keys:
        size = data.real_frames(key,"validation")
        end = offset + size
        yield key,{k:v[offset:end] for k,v in values.items()}
        offset = end


def read_group(path, marker, binding, keys, data, names, ledger):
    doc = read_sealed(marker,GROUP_SCHEMA)
    if doc["binding"] != binding or doc["keys"] != [list(k) for k in keys]:
        raise ValueError("v210_existing_group_binding")
    package = doc["artifact"]
    ledger.reserve("read",package["bytes"])
    raw = _regular_read(input_file(path,package["name"]),CAP,package["bytes"])
    ledger.observed("read",len(raw),persist=False)
    if hashlib.sha256(raw).hexdigest() != package["sha256"]:
        raise ValueError("v210_prediction_group_sha256")
    local = list(segments(raw,keys,data,names))
    if sum(len(row[1]["frame_index"]) for row in local)!=doc["real_frames"]:
        raise ValueError("v210_existing_group_frames")
    return local, doc


def evaluate_groups(project, output, work, pins, pipeline, ledger, owner, comparison, environment):
    data = pipeline.data
    frontend = pipeline._make_frontend()
    pipeline._load_best("frontend",frontend)
    store = FeatureStore(work/"feature-scratch",data,frontend_fingerprint(frontend.model))
    if store.identity != FEATURE:
        raise ValueError("v210_feature_frontend_identity")
    best = pipeline.progress["best"]
    best_pins = {role: item["sha256"] for role,item in best.items()}
    if best_pins != comparison["binding"]["model"]:
        raise ValueError("v210_selected_model")
    names = data.spec["body_joint_names"]+[n for side in ("left","right") for n in data.spec["hands"][side][1:]]
    basis = {"source":pins["base"]["source_sha256"],"code":pins["modules"],"checkpoint_receipt":FINAL,
             "cache_boundary":pins["base"]["final_boundary_sha256"],"feature_identity":FEATURE,"weights":best_pins,
             "eval_seed":EVAL_SEED,"precision":"fp32","upper":"predicted_frozen_body","sampling_steps":10}
    group_receipts, stream = [], []
    initialized = None
    for group in sorted(map(int,pins["groups"])):
        ledger.check()
        key_list = [key for split,key in group_keys(data,group) if split=="validation"]
        expected = {**basis,"group":group,"feature_input":pins["groups"][str(group)]}
        marker = output/f"group-{group:05d}.json"
        if not marker.exists():
            with tempfile.TemporaryDirectory(dir=work,prefix="group-") as directory:
                stage = Path(directory)
                if feature_group(project,pins,store,group,ledger,stage) != key_list:
                    raise ValueError("v210_feature_order")
                if initialized is None:
                    exemplar = store.load(key_list[0],"validation")
                    # The frozen constructor uses this map only to obtain the feature input SHA.
                    cascade = CommonEvaluation(data,{"train":{key_list[0]:exemplar},"validation":{}},pipeline.cfg,pipeline.device,0)
                    body,hand = cascade.make("body"),cascade.make("hand",best["body"]["sha256"])
                    pipeline._load_best("body",body)
                    pipeline._load_best("hand",hand)
                    initialized = cascade,body,hand
                cascade,body,hand = initialized
                start = time.monotonic()
                batches = [rows(cascade,body,hand,store.load(key,"validation"),key,ledger) for key in key_list]
                merged = {key:np.concatenate([x[key] for x in batches]) for key in batches[0]}
                merged["joint_names"] = np.asarray(names)
                path = stage/f"group-{group:05d}-{uuid.uuid4().hex}.npz"
                np.savez_compressed(path,**merged)
                package = {"name":path.name,"bytes":path.stat().st_size,"sha256":_hash_file(path)}
                if package["bytes"] > CAP:
                    raise ValueError("v210_result_group_cap")
                owner()
                retention.copy_verified(path,output/path.name,package,ledger,ledger.clock()+min(120,ledger.remaining()))
                publish_json(marker,GROUP_SCHEMA,{"binding":expected,"artifact":package,"keys":[list(k) for k in key_list],
                             "real_frames":len(merged["frame_index"]),"environment":environment,
                             "prediction_wall_seconds":time.monotonic()-start},owner)
        data_parts, receipt = read_group(output,marker,expected,key_list,data,names,ledger)
        stream.extend(data_parts)
        group_receipts.append(receipt)
        print(f"v2.1.0 group {group:05d} verified; {len(stream)} clips",flush=True)
    if [p[0] for p in stream] != data.clips["validation"] or len(stream)!=421 or len({(s,int(f)) for (s,_),a in stream for f in a["frame_index"]})!=16425:
        raise ValueError("v210_full_validation_coverage")
    final = {"binding":basis,"counts":{"groups":len(group_receipts),"clips":421,"real_frames":16425},
             "receipts":[g["content_sha256"] for g in group_receipts],"environments":[g["environment"] for g in group_receipts],
             "metrics":aggregate(stream,data.spec,comparison)}
    marker = output/"summary.json"
    if marker.exists():
        existing = read_sealed(marker,SUMMARY_SCHEMA)
        check_pin(existing,SUMMARY_SCHEMA,digest({"schema":SUMMARY_SCHEMA,**final}))
    else:
        existing = publish_json(marker,SUMMARY_SCHEMA,final,owner)
    return existing


def run(args,pins,report):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not 0 < time.monotonic()-args.started < 600:
        raise RuntimeError("v210_single_cuda_and_start_time")
    torch.set_num_threads(1)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    device = torch.device("cuda:0")
    for backend in (torch.backends.cuda.matmul,torch.backends.cudnn):
        backend.allow_tf32 = False
    seed_run(42)
    environment = {"gpu":torch.cuda.get_device_name(device),"vram_bytes":torch.cuda.get_device_properties(device).total_memory,
                   "torch":str(torch.__version__),"cuda":torch.version.cuda,"precision":"fp32"}
    report["environment"] = environment
    project,work = args.project.resolve(strict=True),args.work.resolve(strict=True)
    if work.is_relative_to(project) or not (project/"reports").is_dir() or (project/"reports").resolve()!=project/"reports":
        raise ValueError("v210_project_or_scratch")
    for name, sha in pins["modules"].items():
        if _hash_file(Path(__file__).resolve().parent/name)!=sha:
            raise ValueError("v210_overlay_sha256")
    outputs = project/DESTINATION
    if outputs.is_symlink():
        raise ValueError("v210_output_symlink")
    diagnostic = original_json(project,DIAGNOSTIC,pins["evidence"][DIAGNOSTIC],"rewind_v2_gpu_invocation_v1")
    step = original_json(project,RECEIPT,pins["evidence"][RECEIPT],"rewind_v2_gpu_training_checkpoint_v1")
    comparison = original_json(project,COMPARE,pins["evidence"][COMPARE],"rewind_v2_gpu_comparison_v1")
    base = pins["base"]
    identity = identity_for("A",base)
    if (diagnostic["checkpoint"]!=step or diagnostic["comparison"]!=comparison or step["content_sha256"]!=FINAL
            or not step["completed"] or step["identity"]!=identity or step["cursor"]!={"stage":"hand","epoch":3,"batch":0,"updates":197961}):
        raise ValueError("v210_A_final_scope")
    outputs.mkdir(exist_ok=True)
    cfg = training_config(42)
    limits = {**cfg,"resources":{**cfg["resources"],"read_bytes":4*GIB,"write_bytes":2*GIB},
              "terminal_reserve_seconds":180}
    ledger = Ledger(work/"usage",identity,limits)
    with writer(outputs,args.recover_writer) as owner, ledger.invocation("v210_eval"):
        ledger.started = args.started
        doc,spec,plan = load_inputs(SimpleNamespace(invocation=work,workspace=work,slot="A"),base)
        journal,anchor,entries = read_journal(project,doc,spec,base,ledger)
        cache = FullCache(work/"cache-metadata-only",journal,anchor,entries,spec,cfg,ledger)
        try:
            data = FullData(doc,spec,cfg,plan,cache,ledger,base["source_sha256"])
            pipe = Training(cfg,data,work/"unused",device,work/"run",outputs,ledger,owner,identity)
            pipe.binding["environment"] = {k:v for k,v in step["environment"].items() if k not in {"device_name","device_memory_bytes"}}
            package = step["archive"]
            got = retention.restore_run(input_file(project/A/"checkpoints",package["name"]),package,pipe.root,limits,ledger,pipe.binding)
            if got!=step["last"]["sha256"]:
                raise ValueError("v210_A_last_checkpoint_sha256")
            pipe._resume(got)
            if pipe.last!=step["last"] or any(pipe.progress[k]!=v for k,v in step["cursor"].items()):
                raise ValueError("v210_A_last_cursor")
            if pipe.cfg["teacher"]["sampling_steps"]!=10:
                raise ValueError("v210_DDIM_steps")
            summary = evaluate_groups(project,outputs,work,pins,pipe,ledger,owner,reference_scores(diagnostic,comparison),environment)
            report.update(status="completed", summary_sha256=summary["content_sha256"], counts=summary["counts"])
        finally:
            cache.close()
            report["usage"] = dict(ledger.current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project","work","pins"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--started",type=float,required=True)
    parser.add_argument("--recover-writer",default="")
    args = parser.parse_args()
    report = {"status":"failed"}
    try:
        run(args,json.loads(_regular_read(args.pins,2*MIB)),report)
    except BaseException as exc:
        report.update(status="paused" if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else "failed",
                      failure={"type":type(exc).__name__,"reason":str(exc)[:512]})
        raise
    finally:
        report["wall_seconds"] = time.monotonic()-args.started
        _atomic(args.work/"result.json",canonical(report))
        print(canonical(report).decode(),flush=True)


if __name__=="__main__":
    main()

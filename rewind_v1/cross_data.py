"""Validation-only preselection and bounded direct-file preparation for cross-scenario evaluation."""

from collections import defaultdict
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_archive.matching import contiguous_runs, identify
from rewind_m1.local_rgb import _regular_read, _no_link
from rewind_m2.data import encode_camera
from rewind_prepare.converter import official_splits, sha, validate_pose_geometry
from rewind_prepare.extracted import ExtractedLimits, _Budget, _local, _payload_path, _pose_targets, _raw_pose
from rewind_prepare.full_archvis import frozen_snapshot
from rewind_prepare.temporal_cache import _atomic, _check_part
from rewind_prepare.temporal_selection import check_pin, seal
from .data import _field, frame_image
from .training import _source


RUN_ID = "V12-CROSS-SCENARIO-001"
CONFIG_PATH=Path(__file__).resolve().parents[1]/"configs/v12_cross_scenario_001.json"
POLICY=json.loads(CONFIG_PATH.read_bytes())
LIMITS=POLICY["limits"]
ENVIRONMENTS = ("CityPark_Showcase", "DowntownWest_Demo_Environment")
MODEL_SHA = "48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506"
MIB, GIB = 1024**2, 1024**3
WIDTH, MAX_CLIPS = 50, 4
RESULT_BYTES = LIMITS["result_bytes"]


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/visualize_archvis_v1.py", "scripts/visualize_archvis_v12.py")
    return {**_source(), **{name:sha((root/name).read_bytes()) for name in names},
            "configs/v12_cross_scenario_001.json":sha(CONFIG_PATH.read_bytes())}


class EvaluationContext:
    """Explicit evaluation scope; omitted context preserves the original city contract."""
    def __init__(self, run_id=RUN_ID, environments=ENVIRONMENTS, source=source_identity):
        self.run_id,self.environments,self.source=run_id,tuple(environments),source


def evaluation_context(context=None):
    return context if context is not None else EvaluationContext()


def write_report(path, value):
    raw = canonical(value)+b"\n"
    if len(raw) > 32*MIB:
        raise ValueError("cross_report_bytes")
    _atomic(path, raw)


def read_sealed(path, schema, expected, *, context=None):
    context=evaluation_context(context)
    doc = json.loads(_regular_read(_local(path),32*MIB))
    check_pin(doc,schema,expected)
    if doc.get("source_sha256") != digest(context.source()) or doc.get("run_id") != context.run_id:
        raise ValueError("cross_source_or_run_binding")
    return doc


class Budget:
    def __init__(self, seconds, read_limit=None, entries=None):
        self.start,self.seconds = time.monotonic(),seconds
        self.read_limit=LIMITS["source_read_bytes"] if read_limit is None else read_limit
        self.entry_limit=LIMITS["metadata_entries"] if entries is None else entries
        self.read_bytes,self.entries = 0,0
        self.metadata_bytes = 0

    def check(self):
        if time.monotonic()-self.start >= self.seconds:
            raise TimeoutError("cross_time_limit")
        # Linux high-water RSS observation, not an OS memory-limit guarantee.
        try:
            import resource
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > LIMITS["observed_rss_bytes"]:
                raise MemoryError("cross_observed_rss_4GiB")
        except ImportError:
            pass

    def entry(self,size=128):
        self.check();self.entries += 1
        self.metadata_bytes += size
        if self.entries > self.entry_limit:
            raise ValueError("cross_metadata_entry_limit")
        if self.metadata_bytes > LIMITS["metadata_bytes"]:
            raise ValueError("cross_metadata_bytes_limit")

    def read(self,path,size):
        self.check()
        if self.read_bytes+size > self.read_limit:
            raise ValueError("cross_source_read_limit")
        raw = _regular_read(path,8*MIB,size)
        self.read_bytes += len(raw)
        self.check()
        return raw


def assets(inputs):
    inputs = _local(inputs,directory=True)
    snapshot,_,_ = frozen_snapshot(inputs)
    spec_raw = _regular_read(inputs/"spec.json",MIB)
    calibration_raw = _regular_read(inputs/"calibration.json",MIB)
    spec,calibration = json.loads(spec_raw),json.loads(calibration_raw)
    owners,pins = official_splits(inputs/"split",spec,_Budget(ExtractedLimits(8*MIB,60)))
    return snapshot,spec,calibration,{"spec_sha256":sha(spec_raw),"calibration_sha256":sha(calibration_raw),"split_files":pins}


def _listing(root,relative,role,budget):
    pattern = re.compile(r"frame_(\d+)\.json" if role=="pose" else r"final_(\d+)\.png")
    frames,duplicates = {},set()
    with os.scandir(_payload_path(root,relative)) as iterator:
        for item in iterator:
            budget.entry(len(item.name.encode("utf-8"))+128)
            match = pattern.fullmatch(item.name)
            if not match:continue
            if item.is_symlink() or not item.is_file(follow_symlinks=False):
                raise ValueError("cross_nonregular_source")
            frame = int(match[1])
            if frame in frames:duplicates.add(frame)
            frames[frame] = relative+"/"+item.name
    return frames,duplicates


def _proxy(sequence,train_sequences):
    parts = sequence.split("/")
    return {"status":"path_name_proxy_only_original_motion_unverified","person_name":parts[2],"motion_label":parts[-1],
            "train_person_matches":sorted(s for s in train_sequences if s.split("/")[2]==parts[2]),
            "train_motion_label_matches":sorted(s for s in train_sequences if s.split("/")[-1]==parts[-1]),
            "interpretation":"Name overlap suggests possible shared identity/motion; absence does not establish independence."}


def select(inputs,rgb_root,pose_root,output,*,budget=None,context=None):
    context=evaluation_context(context)
    budget = budget or Budget(LIMITS["metadata_seconds"],LIMITS["metadata_bytes"])
    snapshot,spec,_,binding = assets(inputs)
    output = Path(output).absolute()
    roots = {"rgb":_local(rgb_root,directory=True),"pose":_local(pose_root,directory=True)}
    if any(output.is_relative_to(root) for root in roots.values()):raise ValueError("cross_output_inside_sources")
    output.mkdir()
    owners,_ = official_splits(Path(inputs)/"split",spec,_Budget(ExtractedLimits(8*MIB,60)))
    train_sequences = {seq for seq,_ in snapshot["keys"]["train"]}
    candidates = {env:sorted(seq for seq,split in owners.items() if split=="validation" and seq.split("/")[0]==env)
                  for env in context.environments}
    selected,observations,counts = [],[],dict.fromkeys(context.environments,0)
    try:
        for env in context.environments:
            for sequence in candidates[env]:
                budget.check()
                if counts[env]>=2:
                    observations.append({"sequence":sequence,"status":"not_inspected_quota_reached"});continue
                try:
                    listings,duplicates = {},set()
                    for role in ("left","right","pose"):
                        relative = sequence+("/json" if role=="pose" else "/fisheye_final_image/camera_"+role)
                        listings[role],bad = _listing(roots["pose" if role=="pose" else "rgb"],relative,role,budget)
                        duplicates.update(bad)
                except OSError as error:
                    observations.append({"sequence":sequence,"status":"unavailable_metadata","type":type(error).__name__});continue
                common = set.intersection(*(set(v) for v in listings.values()))-duplicates
                union = set.union(*(set(v) for v in listings.values()))
                runs = contiguous_runs(sorted(common)) if common else []
                record = {"sequence":sequence,"duplicate_frames":sorted(duplicates),
                          "missing_by_role":{role:sorted(union-set(values)) for role,values in listings.items()},
                          "contiguous_runs":runs}
                usable = next(((first,last) for first,last in runs if last-first+1>=WIDTH),None)
                if usable is None:
                    observations.append({**record,"status":"no_50_frame_common_run"});continue
                first = usable[0];rows = []
                for frame in range(first,first+WIDTH):
                    files = {}
                    for role in ("left","right","pose"):
                        root = roots["pose" if role=="pose" else "rgb"]
                        path = _payload_path(root,listings[role][frame]);info = _no_link(path);budget.entry()
                        if not 0 < info.st_size <= 8*MIB:raise ValueError("cross_file_size")
                        files[role] = {"path":listings[role][frame],"size_bytes":info.st_size,"mtime_ns":info.st_mtime_ns}
                    rows.append({"frame":frame,"files":files})
                selected.append({"environment":env,"sequence":sequence,"split":"validation","frames":rows,
                                 "overlap_proxy":_proxy(sequence,train_sequences)})
                counts[env]+=1;observations.append({**record,"status":"selected","first_frame":first})
        if any(count==0 for count in counts.values()):raise ValueError("cross_scenario_zero_available_clips")
        doc = seal({"schema":"v12_cross_selection_v1","run_id":context.run_id,"source_sha256":digest(context.source()),
            "status":"complete" if all(count==2 for count in counts.values()) else "partial_availability",
            "model_zip_sha256":MODEL_SHA,"binding":binding,"roots":{k:str(v) for k,v in roots.items()},
            "policy":"Official validation only; lexical sequence order; earliest common contiguous50; no prediction-based selection or train fallback",
            "candidate_counts":{k:len(v) for k,v in candidates.items()},"selected_counts":counts,
            "clips":selected,"observations":observations,"metadata_entries":budget.entries,
            "metadata_bytes_upper":budget.metadata_bytes,"payload_bytes_read":0,
            "elapsed_seconds":time.monotonic()-budget.start,
            "nominal_fps":25,"recorded_timestamps_verified":False,"stereo_sync_measured":False,
            "original_motion_independence_verified":False})
        write_report(output/"selection.json",doc)
        return doc
    except BaseException as error:
        write_report(output/"failure.json",{"status":"failed","type":type(error).__name__,"reason":str(error)[:512],
                    "observations":observations,"selected_counts":counts,"metadata_entries":budget.entries})
        raise


def checked_selection(path,pin,*,context=None):
    context=evaluation_context(context)
    doc = read_sealed(path,"v12_cross_selection_v1",pin,context=context)
    if doc["model_zip_sha256"]!=MODEL_SHA:raise ValueError("cross_model_pin")
    clips = doc["clips"]
    if not len(context.environments)<=len(clips)<=2*len(context.environments) or len({c["sequence"] for c in clips})!=len(clips):
        raise ValueError("cross_selection_scope")
    for env in context.environments:
        if not 1<=sum(c["environment"]==env for c in clips)<=2:raise ValueError("cross_environment_quota")
    for clip in clips:
        frames = [r["frame"] for r in clip["frames"]]
        if (clip["environment"] not in context.environments or clip["split"]!="validation"
                or clip["sequence"].split("/")[0]!=clip["environment"]
                or len(frames)!=50 or frames!=list(range(frames[0],frames[0]+50))):
            raise ValueError("cross_clip_contiguity_or_split")
        for record in clip["frames"]:
            if set(record["files"])!={"left","right","pose"}:raise ValueError("cross_file_roles")
            for role,item in record["files"].items():
                if (identify(item["path"],"pose" if role=="pose" else "rgb")!=(clip["sequence"],record["frame"],role)
                        or type(item["size_bytes"]) is not int or not 0<item["size_bytes"]<=8*MIB):
                    raise ValueError("cross_file_key_or_size")
    return doc


def prepare(inputs,selection,pin,output,*,budget=None,context=None):
    context=evaluation_context(context)
    budget = budget or Budget(LIMITS["prepare_seconds"])
    doc = checked_selection(selection,pin,context=context)
    _,spec,calibration,binding = assets(inputs)
    if doc["binding"]!=binding:raise ValueError("cross_input_binding")
    owners,_ = official_splits(Path(inputs)/"split",spec,_Budget(ExtractedLimits(8*MIB,60)))
    if any(owners.get(c["sequence"])!="validation" for c in doc["clips"]):
        raise ValueError("cross_official_validation_only")
    output = Path(output).absolute();output.mkdir()
    roots = {k:_local(v,directory=True) for k,v in doc["roots"].items()}
    completed = []
    try:
        for number,clip in enumerate(doc["clips"]):
            budget.check()
            arrays = defaultdict(list);rows = []
            with tempfile.TemporaryDirectory(dir=output,prefix=".preparing-") as temporary:
                staged = Path(temporary)/"clip";staged.mkdir()
                for record in clip["frames"]:
                    raw,files = {},{}
                    for role,item in record["files"].items():
                        path = _payload_path(roots["pose" if role=="pose" else "rgb"],item["path"])
                        if _no_link(path).st_mtime_ns!=item["mtime_ns"]:raise ValueError("cross_source_changed_after_selection")
                        raw[role] = budget.read(path,item["size_bytes"])
                        relative = f"raw/{role}/{record['frame']:09d}."+("json" if role=="pose" else "png")
                        destination = staged/relative;destination.parent.mkdir(exist_ok=True,parents=True)
                        destination.write_bytes(raw[role])
                        files[role] = {"path":relative,"size_bytes":len(raw[role]),"sha256":sha(raw[role])}
                    native = _raw_pose(raw["pose"],record["frame"]);residuals = defaultdict(list)
                    camera = validate_pose_geometry(native,spec,calibration,residuals)
                    if not residuals["projection_px"]:raise ValueError("cross_no_projection_checks")
                    fields = _pose_targets(native,spec)
                    for name,field,dim in (("left_xy",fields[0],2),("right_xy",fields[1],2),("target",fields[2],3)):
                        value,valid = _field(field,dim);arrays[name].append(value);arrays[name+"_valid"].append(valid)
                    arrays["images"].append(frame_image(raw,[128,128]))
                    arrays["camera"].append(encode_camera(camera,100.))
                    rows.append({"frame":record["frame"],"files":files,
                                 "geometry_max":{k:max(v) for k,v in residuals.items() if v}})
                tensors = {k:torch.stack(v) for k,v in arrays.items()}
                torch.save(tensors,staged/"clip.pt")
                tensor_raw = _regular_read(staged/"clip.pt",32*MIB)
                entry = {"number":number,"sequence":clip["sequence"],"environment":clip["environment"],"split":"validation",
                         "frames":rows,"tensor":{"name":"clip.pt","bytes":len(tensor_raw),"sha256":sha(tensor_raw)},
                         "source_hash_scope":"Observed selected bytes only; not official archive integrity"}
                write_report(staged/"manifest.json",entry)
                staged.rename(output/f"clip-{number:02d}")
                completed.append(entry)
        result = seal({"schema":"v12_cross_prepared_v1","run_id":context.run_id,"source_sha256":digest(context.source()),
            "selection_sha256":pin,"binding":binding,"clips":completed,"status":doc["status"],
            "source_bytes_read":budget.read_bytes,"model_zip_sha256":MODEL_SHA,
            "elapsed_seconds":time.monotonic()-budget.start,
            "source_hash_scope":"Selected file identity, not official whole-dataset integrity"})
        write_report(output/"prepared.json",result);return result
    except BaseException as error:
        write_report(output/"failure.json",{"status":"failed","type":type(error).__name__,"reason":str(error)[:512],
                     "source_bytes_read":budget.read_bytes,"completed_clips":completed,"selection_sha256":pin})
        raise

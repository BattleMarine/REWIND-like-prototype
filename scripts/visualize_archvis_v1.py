"""Render fixed v1/v1.1 checkpoint predictions and annotations from local stereo RGB."""

import argparse
from collections import defaultdict
import hashlib
import io
import math
from pathlib import Path
import zipfile

from PIL import Image, ImageDraw, ImageFont
import torch

from rewind_archive.contracts import canonical
from rewind_archive.matching import identify, verify_extraction
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _hash, _regular_read
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m2.data import encode_camera
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_prepare.converter import sha, validate_pose_geometry
from rewind_prepare.extracted import IMAGE_SIZE, _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_selection import check_pin, seal
from rewind_v1.data import attach_predicted_hands, frame_image
from rewind_v1.diagnostic import HISTORICAL, validate_pilot
from rewind_v1.expansion import RUN_ID, validate_expansion
from rewind_v1.frontend import PaperHeatmapFrontend, predicted_features
from rewind_v1.teacher import V1Body, V1Hand
from rewind_v1.full_contract import MANIFEST_SHA, RUN_ID as FULL_RUN_ID, diagnostic_decision
from rewind_v1.full_training import FORMAT as FULL_FORMAT, validate_full_config


TRAINING_SHA = "8bd6b99d1119f5f0255ba63143d931c69fde40bd7be93f9b5171fd3eb2d7257d"
SPEC_SHA = "32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187"
MIB = 1024**2
BODY, HAND, DARK = "#ffb12a", "#4ce5ff", "#131923"
ROOT = Path(__file__).resolve().parents[1]


def _archive(path, pin, limit):
    stream = _local(path).open("rb")
    try:
        if not 0 < stream.seek(0, 2) <= limit:
            raise ValueError("v1_view_archive_size")
        stream.seek(0)
        if hashlib.file_digest(stream, "sha256").hexdigest() != pin:
            raise ValueError("v1_view_archive_pin")
        stream.seek(0)
        return stream
    except BaseException:
        stream.close()
        raise


def _member(archive, name, limit, receipt=None):
    info = archive.getinfo(name)
    if not 0 < info.file_size <= limit or archive.namelist().count(name) != 1:
        raise ValueError("v1_view_archive_member")
    raw = archive.read(info)
    if receipt is not None and (receipt["bytes"], receipt["sha256"]) != (len(raw), sha(raw)):
        raise ValueError("v1_view_receipt_pin:" + name)
    return raw


def _selected_pins(report, files, with_teachers, checkpoint_epoch):
    roles = ("frontend", "body", "hand") if with_teachers else ("frontend",)
    pins = {role: report["cursor"]["best"][role] for role in roles}
    target = roles[-1]
    if checkpoint_epoch is not None:
        if checkpoint_epoch not in (1, 5, 10) or report["input"]["run_id"] != RUN_ID:
            raise ValueError("v1_view_milestone_selection")
        pins[target] = report["cursor"]["milestones"][target][str(checkpoint_epoch)]
        if pins[target]["epoch"] != checkpoint_epoch:
            raise ValueError("v1_view_milestone_epoch")
    for role, pin in pins.items():
        allowed = {f"{kind}-{role}-e{pin['epoch']:03d}.pt" for kind in ("best", "milestone")}
        if (type(pin["epoch"]) is not int or not 1 <= pin["epoch"] <= report["config"]["epochs"][role]
                or pin["name"] not in allowed or not _hash(pin["sha256"])
                or any(files[pin["name"]][key] != pin[key] for key in ("bytes", "sha256"))):
            raise ValueError("v1_view_checkpoint_pin")
        if role == target and checkpoint_epoch is not None:
            expected = {name: pins[name]["sha256"] for name in roles[:roles.index(role)]}
            if pin["dependencies"] != expected:
                raise ValueError("v1_view_milestone_dependencies")
    return pins


def _models(path, spec, with_teachers, *, training_sha256=None, run_id="V1-DIAGNOSTIC-001", checkpoint_epoch=None):
    expansion = run_id == RUN_ID
    full = run_id == FULL_RUN_ID
    if (run_id not in ("V1-DIAGNOSTIC-001", RUN_ID, FULL_RUN_ID)
            or ((expansion or full) and not _hash(training_sha256))
            or (not (expansion or full) and training_sha256 not in (None, TRAINING_SHA))
            or (full and checkpoint_epoch is not None)):
        raise ValueError("v1_view_explicit_training_identity")
    training_sha256 = training_sha256 or TRAINING_SHA
    with _archive(path, training_sha256, (2048 if full else 1024 if expansion else 512)*MIB) as stream, zipfile.ZipFile(stream) as archive:
        receipt = read_json(_member(archive, "recovery_receipt.json", MIB))
        if receipt["schema"] != ("archvis_v12_training_recovery_v1" if full else "rewind_v1_recovery_v1"):
            raise ValueError("v1_view_training_receipt")
        if full:
            check_pin(receipt, "archvis_v12_training_recovery_v1", receipt.get("content_sha256"))
        files = receipt["files"]
        report = read_json(_member(archive, "run.json", MIB, files["run.json"]))
        expected_epochs = dict.fromkeys(("frontend", "body", "hand"), 10 if expansion or full else 1)
        if (report["status"] != "completed" or report["cursor"]["updates"] != (115720 if full else 5400 if expansion else 540)
                or report["input"]["run_id"] != run_id or report["config"]["epochs"] != expected_epochs
                or report["input"]["spec_sha256"] != SPEC_SHA
                or (not full and (report["input"]["selection_sha256"] != HISTORICAL["selection_sha256"]
                                  or report["input"]["cache_sha256"] != HISTORICAL["cache_sha256"]))):
            raise ValueError("v1_view_training_binding")
        if full:
            if (report["format"] != FULL_FORMAT or report["input"]["manifest_sha256"] != MANIFEST_SHA
                    or report["coverage"]["train_unique_frames"] != 20888
                    or report["coverage"]["frontend_validation_frames"] != 2236
                    or report["coverage"]["teacher_validation_frames"] != 1450):
                raise ValueError("v12_view_training_binding")
            decision = report["input"]["diagnostic_decision"]
            check_pin(decision, "archvis_v12_training_diagnostic_decision_v1", report["input"]["decision_sha256"])
            if canonical(decision) != canonical(diagnostic_decision(report["config_sha256"], report["input"]["plan_sha256"])):
                raise ValueError("v12_view_diagnostic_contract")
        (validate_full_config if full else validate_expansion if expansion else validate_pilot)(report["config"])
        selected = _selected_pins(report, files, with_teachers, checkpoint_epoch)
        implementations = ["rewind_v1/frontend.py", "rewind_v1/data.py", "rewind_m1/frontend.py",
                           "rewind_m1/rgb_frontend_artifacts.py", "rewind_m1/local_rgb.py"]
        if with_teachers:
            implementations.extend(("rewind_v1/teacher.py", "rewind_v1/attention.py", "rewind_m2/models.py",
                                    "rewind_m2/data.py", "rewind_m2/diffusion.py", "rewind_m2/streaming.py",
                                    "rewind_m3/models.py", "rewind_prepare/converter.py", "rewind_prepare/coordinates.py"))
        for name in implementations:
            raw = (ROOT / name).read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            if report["source_code_sha256"][name] not in (sha(raw), sha(lf), sha(lf.replace(b"\n", b"\r\n"))):
                raise ValueError("v1_view_model_implementation_pin:" + name)
        models, pins = {}, {}
        for role, pin in selected.items():
            name = pin["name"]
            raw = _member(archive, name, 128*MIB, files[name])
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            del raw
            dependencies = {key: v["sha256"] for key, v in pins.items()}
            if (state["stage"] != role or state["epoch"] != pin["epoch"] or state["score"] != pin["score"]
                    or state["dependencies"] != dependencies
                    or any(canonical(state["binding"][key]) != canonical(report[key]) for key in
                           ("format", "input", "config", "config_sha256", "source_code_sha256", "environment", "coverage"))):
                raise ValueError("v1_view_best_binding")
            if role == "frontend":
                model = PaperHeatmapFrontend(len(spec["joint_names"]))
            else:
                names = _group_names(spec, role)
                edges = [(names.index(a), names.index(b)) for a, b in spec["edges"] if a in names and b in names]
                cfg = report["config"]["teacher"]
                model = (V1Body(cfg, len(names), edges) if role == "body" else
                         V1Hand(cfg, len(names), edges, len(spec["upper_joint_names"])))
            model.load_state_dict(state["model"], strict=True)
            if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
                raise ValueError("v1_view_nonfinite_weights")
            models[role], pins[role] = model.eval().requires_grad_(False), {**files[name], "epoch": pin["epoch"],
                                                                        "dependencies": dependencies}
        return models, pins, report, {name: report["source_code_sha256"][name] for name in implementations}


def _group_names(spec, role):
    if role == "body":
        return spec["body_joint_names"]
    return [name for side in ("left", "right") for name in spec["hands"][side][1:]]


def _audited_inputs(path):
    with _archive(path, HISTORICAL["audit_recovery_zip_sha256"], 160*MIB) as stream, zipfile.ZipFile(stream) as archive:
        cache = read_json(_member(archive, "cache/cache_manifest.json", 12*MIB))
        check_pin(cache, "archvis_temporal_raw_cache_v1", HISTORICAL["cache_sha256"])
        camera = read_json(_member(archive, "audit/camera_candidates.json", 8*MIB))
        check_pin(camera, "archvis_v1_camera_candidates_v1", HISTORICAL["camera_candidates_sha256"])
    rows = {tuple(row["key"]): row for part in cache["parts"] for row in part["records"]}
    return rows, {tuple(row["key"]): row for row in camera["rows"]}


def _selected_inputs(path):
    raw = _regular_read(_local(path), 16*MIB)
    plan = read_json(raw)
    check_pin(plan, "archvis_temporal_selection_v1", HISTORICAL["selection_sha256"])
    return {tuple(row["key"]): row for row in plan["records"]}, sha(raw)


def _prepared_sample(prepared):
    inputs = {}
    for role in ("left", "right", "pose"):
        root = _local(prepared / "inputs" / role, directory=True)
        report = verify_extraction(root, allow_code_change=True)
        by_key = {}
        for item in report["files"]:
            sequence, frame, observed = identify(item["member"], "pose" if role == "pose" else "rgb")
            if observed != role or (sequence, frame) in by_key:
                raise ValueError("v1_view_prepared_role")
            by_key[sequence, frame] = (root / item["relative_path"], item)
        inputs[role] = by_key
    return inputs


def _load_frame(paths, frame, pins, spec):
    raw = {role: _regular_read(_local(path), 8*MIB) for role, path in paths.items()}
    sources = {role: {"bytes": len(value), "sha256": sha(value), "path": str(paths[role])} for role, value in raw.items()}
    if any((item["bytes"], item["sha256"]) != (pins[role]["size_bytes"], pins[role]["sha256"])
           for role, item in sources.items()):
        raise ValueError(f"v1_view_payload_pin:{frame}")
    return raw, _raw_pose(raw["pose"], frame), sources


def _draw_joints(image, points, spec):
    draw = ImageDraw.Draw(image)
    names = spec["joint_names"]
    for a, b in spec["edges"]:
        first, second = points[names.index(a)], points[names.index(b)]
        if first is not None and second is not None:
            draw.line([tuple(first), tuple(second)], fill=BODY if b in spec["body_joint_names"] else HAND, width=2)
    for name, point in zip(names, points):
        if point is not None:
            x, y = point
            draw.ellipse((x-3, y-3, x+3, y+3), fill=BODY if name in spec["body_joint_names"] else HAND,
                         outline="#14212c")


def _rgb_tile(raw, points, spec, size):
    with Image.open(io.BytesIO(raw), formats=["PNG"]) as source:
        tile = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    scaled = [(point[0]*(size-1)/1023, point[1]*(size-1)/1023)
              if point is not None and all(0 <= axis <= 1023 for axis in point) else None for point in points]
    _draw_joints(tile, scaled, spec)
    return tile


def _project(points):
    return [(math.cos(.38)*x+math.sin(.38)*z, -y) if point is not None else None
            for point in points for x, y, z in [point or (0., 0., 0.)]]


def _shared_3d_view(prediction, target, size):
    return _shared_3d_groups((prediction, target), size)


def _shared_3d_groups(groups, size):
    projected = [_project(points) for points in groups]
    coords = [p for group in projected for p in group if p is not None]
    lo_x, hi_x = min(p[0] for p in coords), max(p[0] for p in coords)
    lo_y, hi_y = min(p[1] for p in coords), max(p[1] for p in coords)
    scale = (size-64) / max(hi_x-lo_x, hi_y-lo_y, 1)
    return [[None if p is None else (size/2+(p[0]-(lo_x+hi_x)/2)*scale,
                                      size/2-(p[1]-(lo_y+hi_y)/2)*scale) for p in group] for group in projected]


def panel(raw, pred, gt, predicted3d, target3d, spec, sequence, frame, scope, model_label="REWIND v1"):
    size = 450
    rows = 3 if predicted3d is not None else 2
    canvas = Image.new("RGB", (size*2+36, 85+rows*(size+29)+12), DARK)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    title = ImageFont.load_default(size=25)
    draw.text((12, 8), f"{model_label}  |  trained prediction vs. annotation", font=title, fill="white")
    draw.text((12, 41), f"{sequence.split('/')[-1]}  /  frame {frame}  |  nominal 25 FPS", font=font, fill="white")
    draw.text((12, 63), f"{scope}  |  body: amber  /  fingers: cyan", font=font, fill="#b9c8d4")
    tiles = []
    for label, points in (("MODEL 2D", pred), ("GT 2D", gt)):
        tiles.append([(f"{side.upper()}  /  {label}", _rgb_tile(raw[side], points[i], spec, size))
                      for i, side in enumerate(("left", "right"))])
    if predicted3d is not None:
        tiles.append([])
        for label, points in zip(("MODEL 3D  /  device mm", "GT 3D  /  same axes and scale"),
                                 _shared_3d_view(predicted3d, target3d, size)):
            tile = Image.new("RGB", (size, size), "#202a38")
            _draw_joints(tile, points, spec)
            tiles[-1].append((label, tile))
    for row, entries in enumerate(tiles):
        for col, (label, tile) in enumerate(entries):
            x, y = 12+col*(size+12), 86+row*(size+29)
            draw.text((x, y), label, font=font, fill="white")
            canvas.paste(tile, (x, y+24))
    return canvas


def comparison_panel(raw, first2d, second2d, gt2d, first3d, second3d, gt3d, spec, frame, epochs, scope, *, model_labels=None):
    size = 360
    rows = 3 if second3d is not None else 2
    canvas = Image.new("RGB", (3*size+48, 108+rows*(size+30)), DARK)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((12, 8), f"Same stereo frame {frame}  /  independent T=1 reset", font=ImageFont.load_default(size=24), fill="white")
    conditioning = "camera-pose conditioned 3D" if second3d is not None else "2D only"
    draw.text((12, 39), f"{scope} / {conditioning}", font=font, fill="#b9c8d4")
    draw.text((12, 62), "Body: amber  /  fingers: cyan  /  3D shares axes and scale across all columns", font=font, fill="#b9c8d4")
    selected = " ".join(f"{role[0].upper()}{epoch}" for role, epoch in epochs.items())
    labels = (*model_labels, "GT annotation") if model_labels is not None else ("v1 1 epoch", f"v1.1 best  {selected}", "GT annotation")
    groups = (first2d, second2d, gt2d)
    for row, side in enumerate(("left", "right")):
        for col, (label, points) in enumerate(zip(labels, groups)):
            x, y = 12+col*(size+12), 92+row*(size+30)
            draw.text((x, y), f"{label} / {side}", font=font, fill="white")
            canvas.paste(_rgb_tile(raw[side], points[row], spec, size), (x, y+23))
    if second3d is not None:
        for col, (label, points) in enumerate(zip(labels, _shared_3d_groups((first3d, second3d, gt3d), size))):
            x, y = 12+col*(size+12), 92+2*(size+30)
            draw.text((x, y), f"{label} / device mm", font=font, fill="white")
            tile = Image.new("RGB", (size, size), "#202a38")
            _draw_joints(tile, points, spec)
            canvas.paste(tile, (x, y+23))
    return canvas


@torch.inference_mode()
def infer_3d(models, images, spec, camera_rows, keys, config):
    teacher = config["teacher"]
    scale = teacher["target_scale_mm"]
    schedule = DiffusionSchedule(teacher["diffusion_steps"], teacher["beta_start"], teacher["beta_end"])
    cameras = torch.stack([encode_camera(row["camera_pose"], scale) for row in camera_rows])[None]
    features = torch.cat([predicted_features(models["frontend"], images[i:i+2])
                          for i in range(0, len(keys), 2)]).reshape(1, len(keys), -1, 14)
    valid = torch.ones((1, len(keys)), dtype=torch.bool)

    def sample(role, model, offset):
        names = _group_names(spec, role)
        indices = [spec["joint_names"].index(name) for name in names]
        noise = torch.stack([keyed_frame_noise(seq, frame, len(names), config["seed"]+offset)
                             for seq, frame in keys])[None]
        return schedule.sample(model, features[..., indices, :].flatten(-2), cameras, valid, noise,
                               teacher["sampling_steps"]) * scale

    body = sample("body", models["body"], 100)
    upper = body[..., [spec["body_joint_names"].index(name) for name in spec["upper_joint_names"]], :]
    bound = BoundHand(models["hand"], upper/scale, valid[..., None].expand(upper.shape[:-1])).eval()
    hand = attach_predicted_hands(body, sample("hand", bound, 200), spec)
    points = dict(zip(_group_names(spec, "body")+_group_names(spec, "hand"), torch.cat((body, hand), dim=-2).unbind(-2)))
    return torch.stack([points[name] for name in spec["joint_names"]], dim=-2)[0]


def _predict(models, images, spec, cameras, keys, config, *, with_teachers, independent):
    native, confidence = [], []
    for group in images.split(2):
        _, xy, scores = models["frontend"](group)
        if not torch.isfinite(xy).all() or not torch.isfinite(scores).all():
            raise ValueError("v1_view_nonfinite_2d_prediction")
        native.extend(scale_xy(xy, config["frontend"]["size"], IMAGE_SIZE).tolist())
        confidence.extend(scores.tolist())
    predicted = None
    if with_teachers:
        predicted = (torch.cat([infer_3d(models, images[i:i+1], spec, cameras[i:i+1], keys[i:i+1], config)
                                for i in range(len(keys))]) if independent else
                     infer_3d(models, images, spec, cameras, keys, config))
        if not torch.isfinite(predicted).all():
            raise ValueError("v1_view_nonfinite_3d_prediction")
        predicted = predicted.tolist()
    return native, confidence, predicted


@torch.inference_mode()
def run(args):
    if (not 1 <= len(args.frames) <= 50 or sorted(set(args.frames)) != args.frames or args.frames[0] < 0
            or (not args.independent_frames and args.frames != list(range(args.frames[0], args.frames[0]+len(args.frames))))
            or not 1 <= args.display_count <= 12):
        raise ValueError("v1_view_bounded_contiguous_frames")
    spec_raw = _regular_read(_local(args.spec), MIB)
    if sha(spec_raw) != SPEC_SHA:
        raise ValueError("v1_view_spec_pin")
    spec = validate_spec(read_json(spec_raw))
    selection = getattr(args, "selection", None)
    selection_file_sha = None
    if selection:
        if not args.prepared or args.audit:
            raise ValueError("v1_view_selection_requires_local_prepared_inputs")
        rows, selection_file_sha = _selected_inputs(selection)
        cameras = {}
    else:
        rows, cameras = _audited_inputs(args.audit)
    keys = [(args.sequence, frame) for frame in args.frames]
    if args.prepared:
        inputs = _prepared_sample(args.prepared)
        if not args.independent_frames:
            raise ValueError("v1_view_prepared_requires_independent_frames")
    else:
        inputs = None
        if any(key not in rows or key not in cameras or rows[key]["split"] != cameras[key]["split"] for key in keys):
            raise ValueError("v1_view_sample_outside_audited_input")
    run_id = getattr(args, "run_id", "V1-DIAGNOSTIC-001")
    training_sha256 = getattr(args, "training_sha256", None) or TRAINING_SHA
    checkpoint_epoch = getattr(args, "checkpoint_epoch", None)
    compare_training = getattr(args, "compare_training", None)
    if compare_training and (run_id != RUN_ID or checkpoint_epoch is not None):
        raise ValueError("v1_view_comparison_requires_expansion_best")
    models, pins, report, implementation_pins = _models(args.training, spec, not args.no_3d,
        training_sha256=getattr(args, "training_sha256", None), run_id=run_id, checkpoint_epoch=checkpoint_epoch)
    size = report["config"]["frontend"]["size"]
    calibration = None
    if inputs and not args.no_3d:
        calibration_raw = _regular_read(_local(args.prepared/"metadata/scaramuzza.json"), MIB)
        if sha(calibration_raw) != spec["pinned_sources"]["scaramuzza_sha256"]:
            raise ValueError("v1_view_calibration_pin")
        calibration = read_json(calibration_raw)
    items, camera_rows = [], []
    for key, frame in zip(keys, args.frames):
        if inputs:
            paths = {role: indexed[key][0] for role, indexed in inputs.items()}
            expected = {role: indexed[key][1] for role, indexed in inputs.items()}
        else:
            paths = {side: args.rgb_dir/("camera_"+side)/f"final_{frame}.png" for side in ("left", "right")}
            paths["pose"] = args.pose_dir/f"frame_{frame}.json"
            expected = rows[key]["files"]
        raw, doc, source = _load_frame(paths, frame, expected, spec)
        items.append((raw, _pose_targets(doc, spec), source))
        if not args.no_3d:
            if inputs:
                residuals = defaultdict(list)
                camera = validate_pose_geometry(doc, spec, calibration, residuals)
                if not residuals["projection_px"]:
                    raise ValueError("v1_view_no_camera_reprojection")
                camera_rows.append({"camera_pose": camera})
            else:
                camera_rows.append(cameras[key])
    images = torch.stack([frame_image(raw, size) for raw, _, _ in items])
    native_2d, confidence, predicted = _predict(models, images, spec, camera_rows, keys, report["config"],
                                               with_teachers=not args.no_3d, independent=args.independent_frames)
    baseline = None
    if compare_training:
        baseline_models, baseline_pins, baseline_report, baseline_implementation = _models(compare_training, spec, not args.no_3d)
        if baseline_report["config"]["frontend"]["size"] != size:
            raise ValueError("v1_view_comparison_preprocessing")
        first2d, first_confidence, first3d = _predict(baseline_models, images, spec, camera_rows, keys,
            baseline_report["config"], with_teachers=not args.no_3d, independent=args.independent_frames)
        baseline = {"training_sha256": TRAINING_SHA, "model_weights": baseline_pins,
                    "inference_implementation_sha256": baseline_implementation}
    shown = list(range(min(len(keys), args.display_count)))
    if len(keys) > args.display_count:
        shown = [round(i*(len(keys)-1)/max(args.display_count-1, 1)) for i in shown]
    args.output.mkdir(exist_ok=False)
    frames, panels = [], []
    for index, (key, (raw, fields, source)) in enumerate(zip(keys, items)):
        gt2d = [field["coordinates"] for field in fields[:2]]
        gt3d = fields[2]["coordinates"]
        pred3d = None if predicted is None else predicted[index]
        frames.append({"frame": key[1], "nominal_seconds": key[1]/25, "sources": source, "predicted_2d_native": native_2d[index],
                       "confidence": confidence[index], "gt_2d_native": gt2d, "predicted_3d_device_mm": pred3d,
                       "gt_3d_device_mm": gt3d})
        if index in shown:
            scope = "Outside fixed selection / T=1 reset" if inputs else "Audited diagnostic sample"
            if key in rows and inputs:
                scope = "Local sample / independent frame"
            label = "REWIND v1.1" if run_id == RUN_ID else "REWIND v1"
            image = panel(raw, native_2d[index], gt2d, pred3d, gt3d, spec, args.sequence, key[1], scope, label)
            png = args.output / f"frame-{key[1]:06d}.png"
            image.save(png)
            frames[-1]["panel"] = {"name": png.name, "sha256": sha(png.read_bytes())}
            panels.append(image)
        if baseline is not None:
            frames[-1]["baseline"] = {"predicted_2d_native": first2d[index], "confidence": first_confidence[index],
                                     "predicted_3d_device_mm": None if first3d is None else first3d[index]}
            if index in shown:
                comparison = comparison_panel(raw, first2d[index], native_2d[index], gt2d,
                    None if first3d is None else first3d[index], pred3d, gt3d, spec, key[1],
                    {role: item["epoch"] for role, item in pins.items()}, scope)
                png = args.output / f"comparison-frame-{key[1]:06d}.png"
                comparison.save(png)
                frames[-1]["comparison_panel"] = {"name": png.name, "sha256": sha(png.read_bytes())}
    if len(panels) > 1 and not args.independent_frames:
        duration = [40*(shown[i+1]-index) if i+1 < len(shown) else 40 for i, index in enumerate(shown)]
        panels[0].save(args.output/"preview.webp", save_all=True, append_images=panels[1:], duration=duration, loop=0)
    result = seal({"schema": "archvis_v1_visualization_v1", "sequence": args.sequence,
                   "split": None if inputs else rows[keys[0]]["split"],
                   "fixed_selection_membership": [key in rows for key in keys],
                   "training_sha256": training_sha256, "run_id": run_id,
                   "checkpoint_selection": "best" if checkpoint_epoch is None else f"{('frontend' if args.no_3d else 'hand')}_epoch_{checkpoint_epoch}",
                   "audit_sha256": None if selection else HISTORICAL["audit_recovery_zip_sha256"],
                   "selection_sha256": HISTORICAL["selection_sha256"], "selection_file_sha256": selection_file_sha,
                   "raw_cache_manifest_sha256": HISTORICAL["cache_sha256"], "model_weights": pins, "spec_sha256": SPEC_SHA,
                   "inference_implementation_sha256": implementation_pins,
                   "visualization_script_sha256": sha(Path(__file__).read_bytes()),
                   "preprocessing": {"frontend_size": size, "native_size": IMAGE_SIZE, "resize": "bilinear_align_corners"},
                   "inference": {"device": "cpu", "torch": str(torch.__version__), "nominal_fps": 25,
                                 "clock_verified": False, "ground_truth_usage": "comparison_only",
                                 "camera_pose_source": ("local pose JSON camera metadata" if inputs else "pinned audit camera metadata")
                                 if predicted is not None else None,
                                 "gt_joints_used_for_camera_geometry_audit": bool(inputs and predicted is not None),
                                 "hand_upper": "predicted_body" if predicted is not None else None,
                                 "context_frames": 1 if args.independent_frames else len(keys),
                                 "independent_frame_reset": args.independent_frames,
                                 "full_50_frame_clip": len(keys)==50 and not args.independent_frames,
                                 "ddim_steps": report["config"]["teacher"]["sampling_steps"] if predicted is not None else None},
                   "validation_reference_only": True, "joint_names": spec["joint_names"],
                   "displayed_frames": [keys[index][1] for index in shown], "baseline": baseline, "frames": frames})
    (args.output/"predictions.json").write_bytes(canonical(result)+b"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "spec", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--audit", type=Path, help="Fixed original audit recovery ZIP")
    evidence.add_argument("--selection", type=Path, help="Fixed selection.json for --prepared local samples")
    for name in ("rgb-dir", "pose-dir", "prepared"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", required=True, type=int, nargs="+")
    parser.add_argument("--display-count", type=int, default=6)
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--independent-frames", action="store_true")
    parser.add_argument("--training-sha256", help="Required explicit ZIP SHA-256 for a v1.1 archive")
    parser.add_argument("--run-id", choices=("V1-DIAGNOSTIC-001", RUN_ID), default="V1-DIAGNOSTIC-001")
    parser.add_argument("--checkpoint-epoch", choices=(1, 5, 10), type=int,
                        help="Hand milestone with recorded upstream best models, or frontend milestone with --no-3d")
    parser.add_argument("--compare-training", type=Path, help="Fixed original 1epoch ZIP for a same-input comparison")
    torch.set_num_threads(1)
    args = parser.parse_args()
    if not ((args.prepared and not args.rgb_dir and not args.pose_dir)
            or (not args.prepared and args.rgb_dir and args.pose_dir)):
        parser.error("provide --prepared or both --rgb-dir and --pose-dir")
    result = run(args)
    print(canonical({"frames": result["displayed_frames"], "content_sha256": result["content_sha256"]}).decode())


if __name__ == "__main__":
    main()

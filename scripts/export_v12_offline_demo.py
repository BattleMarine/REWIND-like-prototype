"""Export original stereo RGB and pinned v1.2 CPU predictions to offline demo.json."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import torch

from rewind_archive.contracts import canonical, safe_path
from rewind_m1.local_rgb import _regular_read
from rewind_m2.data import encode_camera
from rewind_prepare.converter import official_splits, sha
from rewind_prepare.coordinates import camera_transform
from rewind_prepare.extracted import _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_selection import seal
from rewind_v1.cross_inference import predict
from rewind_v1.data import frame_image
from rewind_v1.full_contract import AUDIT_ZIP_SHA, MANIFEST_SHA, load_audit
from scripts import visualize_archvis_v1 as view
from scripts.v12_share_smoke import EXPECTED, PARENT_SHA


VIEW_NAMES = ("left", "right")
NATIVE_SIZE = [1024, 1024]
MAX_CONTEXT = 50


class SplitFiles:
    @staticmethod
    def read(path):
        return _regular_read(_local(path), 32*view.MIB)


def archive_members(archive, sequence, indices):
    requested = {f"{sequence}/fisheye_final_image/camera_{side}/final_{index}.png": (index, side)
                 for index in indices for side in VIEW_NAMES}
    found = {}
    for info in archive.infolist():
        start = info.filename.find(sequence + "/")
        if start == -1 or (start and info.filename[start-1] != "/"):
            continue
        key = requested.get(info.filename[start:])
        if key is None:
            continue
        if key in found or not 0 < info.file_size <= 8*view.MIB:
            raise ValueError(f"demo_RGB_member:{key}")
        found[key] = info
    if len(found) != len(requested):
        raise ValueError("demo_missing_stereo_archive_members")
    return found


def camera_input(pose, spec, scale):
    coord = spec["coordinates"]
    return encode_camera({side: camera_transform(pose.get("camera_"+side), coord["mm_per_raw_unit"],
                                                  coord["raw_consistency_tolerance"])[0]
                          for side in VIEW_NAMES}, scale)


def export_gt(pose, spec):
    fields = _pose_targets(pose, spec)
    masked = [[point if valid else None for point, valid in zip(field["coordinates"], field["valid_mask"])]
              for field in fields]
    return {"gt_xy": masked[:2], "gt_xyz": masked[2]}


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    if safe_path(args.sequence) != args.sequence or len(args.sequence.split("/")) != 6:
        raise ValueError("demo_dataset_sequence")
    if not 0 <= args.start <= 1_000_000 or not 1 <= args.frames <= MAX_CONTEXT or not 1 <= args.threads <= 8:
        raise ValueError("demo_bounded_single_clip")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("demo_new_output_required")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("demo_cuda_unavailable")
        device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        used_device = {"name": properties.name, "vram_bytes": properties.total_memory,
                       "compute_capability": [properties.major, properties.minor], "torch_cuda": torch.version.cuda}
    else:
        device = torch.device("cpu")
        used_device = {"name": platform.processor() or platform.machine()}
    spec_raw = SplitFiles.read(args.spec)
    if sha(spec_raw) != view.SPEC_SHA:
        raise ValueError("demo_spec_pin")
    spec = validate_spec(json.loads(spec_raw))
    owners, splits = official_splits(args.split_dir, spec, SplitFiles())
    if args.sequence not in owners:
        raise ValueError("demo_official_split_membership")
    indices = range(args.start, args.start + args.frames)
    audited = None
    if args.audit:
        manifest, _, _ = load_audit(args.audit)
        audited = {r["frame_index"]: r for r in manifest["records"] if r["sequence_id"] == args.sequence}
        if any(index not in audited or audited[index]["split"] != owners[args.sequence] for index in indices):
            raise ValueError("demo_v12_audit_membership")
    models, pins, model_report, implementation = view._models(
        args.training, spec, True, training_sha256=PARENT_SHA, run_id=view.FULL_RUN_ID)
    if any(pins[role]["sha256"] != expected for role, (_, expected) in EXPECTED.items()):
        raise ValueError("demo_original_best_checkpoints")
    config = model_report["config"]
    if config["seed"] != 71 or config["teacher"]["sampling_steps"] != 10:
        raise ValueError("demo_original_diffusion_settings")
    models = {role: model.to(device) for role, model in models.items()}
    models["spec"] = spec
    images, cameras, poses, frames = [], [], [], []

    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".demo-") as temp, ExitStack() as stack:
        out = Path(temp) / "result"
        for side in VIEW_NAMES:
            (out / "rgb" / side).mkdir(parents=True)
        members = None
        if args.rgb_zip:
            stream = stack.enter_context(view._archive(args.rgb_zip, args.rgb_zip_sha256, 1024*view.MIB))
            archive = stack.enter_context(zipfile.ZipFile(stream))
            members = archive_members(archive, args.sequence, indices)
        for index in indices:
            pixels, frame = {}, {"index": index, "sources": {}}
            for side in VIEW_NAMES:
                if members is None:
                    file = args.rgb_dir / f"camera_{side}" / f"final_{index}.png"
                    raw = _regular_read(_local(file), 8*view.MIB)
                else:
                    raw = archive.read(members[index, side])
                pixels[side] = raw
                name = f"rgb/{side}/{index:06d}.png"
                (out / name).write_bytes(raw)
                frame[side] = name
                frame["sources"][side] = {"sha256": sha(raw), "bytes": len(raw)}
            pose_raw = _regular_read(_local(args.pose_dir / f"frame_{index}.json"), 8*view.MIB)
            pose = _raw_pose(pose_raw, index)
            cameras.append(camera_input(pose, spec, config["teacher"]["target_scale_mm"]))
            images.append(frame_image(pixels, config["frontend"]["size"]))
            poses.append(pose)
            frame["sources"]["pose"] = {"sha256": sha(pose_raw), "bytes": len(pose_raw)}
            if audited is not None:
                for role, source in frame["sources"].items():
                    pin = audited[index]["files"][role]
                    if (source["sha256"], source["bytes"]) != (pin["sha256"], pin["size_bytes"]):
                        raise ValueError(f"demo_v12_audit_payload:{index}:{role}")
            frames.append(frame)
        print(f"Verified {len(frames)} stereo triplets; running frozen {device} clip inference", flush=True)
        keys = [(args.sequence, index) for index in indices]
        prediction, features = predict(models, torch.stack(images), torch.stack(cameras), keys, config, device)
        views = features.reshape(len(frames), len(spec["joint_names"]), 2, 7).transpose(1, 2)
        xy = (views[..., :2] + 1) * (NATIVE_SIZE[0]-1) / 2
        if not torch.isfinite(xy).all() or not torch.isfinite(views[..., 3]).all():
            raise ValueError("demo_nonfinite_2d")
        indices_by_joint = {name: index for index, name in enumerate(spec["joint_names"])}
        for slot, frame in enumerate(frames):
            frame.update(xy=xy[slot].tolist(), xyz=prediction["whole"][slot].tolist(),
                         confidence=views[slot, ..., 3].tolist())
            if not args.omit_gt:
                frame.update(export_gt(poses[slot], spec))
        doc = seal({
            "schema": "rewind_v12_offline_demo_v1", "model": "REWIND v1.2 F9 / B10 / H9",
            "sequence": args.sequence, "split": owners[args.sequence], "fps": 25,
            "native_size": NATIVE_SIZE, "view_order": list(VIEW_NAMES),
            "joint_names": spec["joint_names"], "edges": [[indices_by_joint[a], indices_by_joint[b]] for a,b in spec["edges"]],
            "body_indices": [indices_by_joint[n] for n in spec["body_joint_names"]],
            "hand_indices": {side: [indices_by_joint[n] for n in spec["hands"][side]] for side in VIEW_NAMES},
            "coordinates": {"xy": "native distorted image pixels, top left origin", "xyz": "device-relative millimetres",
                            "edge_source": "pinned v1.2 model spec", "missing_gt": "null"},
            "inference": {"device": str(device), "device_details": used_device,
                          "torch": str(torch.__version__), "threads": args.threads,
                          "context_frames": len(frames), "seed": config["seed"], "ddim_steps": 10,
                          "hand_conditioning": "predicted upper body", "camera": "original absolute stereo metadata",
                          "ground_truth_conditioning": False, "clock_verified": False, "precision": "fp32"},
            "provenance": {"training_sha256": PARENT_SHA, "models": pins, "implementations": implementation,
                           "pipeline_sha256": sha(Path(predict.__code__.co_filename).read_bytes()),
                           "spec_sha256": view.SPEC_SHA, "splits": splits, "rgb_zip_sha256": args.rgb_zip_sha256,
                           "v12_training_audit": {"zip_sha256": AUDIT_ZIP_SHA, "manifest_sha256": MANIFEST_SHA} if audited is not None else None,
                           "selection": "caller-fixed contiguous frames before looking at predictions"},
            "frames": frames,
        })
        result = canonical(doc)
        (out / "demo.json").write_bytes(result)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("demo_new_output_required")
        out.rename(args.output)
    print(f"{args.output/'demo.json'} SHA-256 {sha(result)} in {time.monotonic()-started:.3f}s", flush=True)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "spec", "split-dir", "pose-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    rgb = parser.add_mutually_exclusive_group(required=True)
    rgb.add_argument("--rgb-dir", type=Path)
    rgb.add_argument("--rgb-zip", type=Path)
    parser.add_argument("--rgb-zip-sha256")
    parser.add_argument("--audit", type=Path, help="Verify exact membership and payload hashes against the frozen v1.2 audit")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--omit-gt", action="store_true")
    args = parser.parse_args()
    if bool(args.rgb_zip) != bool(args.rgb_zip_sha256):
        parser.error("RGB ZIP and SHA-256 must be supplied together")
    run(args)


if __name__ == "__main__":
    main()

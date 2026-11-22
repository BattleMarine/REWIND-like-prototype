"""Verify the original manifest, cache chain, and representative saved tensors."""

import io
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from rewind_v1 import v2_cache
from scripts.run_v2_bulk_cache import PACKAGE_CAP, rows_for
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.v2_gpu_training_comparison import shared_clips
from scripts.v2_gpu128_a_bundle_format import BULK, LEGACY, FEATURE_IDENTITY, feature_groups
from scripts.v2_gpu128_a_bundle_local import pinned_journal
from scripts.v2_tuning_cpu_common import (MIB, BundleView, canonical, check_json_pin, load_scope, native_evidence,
                                         require, sha, staged_archive)


SHARD_SAMPLES = (0, 1062)
GROUP_SAMPLES = (0, 146)


def tensors(values, contract, label):
    require(values.keys() == contract.keys(), f"{label}: tensor roster")
    for name, (shape, dtype) in contract.items():
        value = values[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == dtype
                and bool(torch.isfinite(value).all()), f"{label}: {name} shape/dtype/finite")


def cache_tensors(folder, count, joints, spec):
    images = np.load(folder / "images.npy", mmap_mode="r", allow_pickle=False)
    values = torch.load(folder / "targets.pt", map_location="cpu", weights_only=True)
    f32, f64, mask = torch.float32, torch.float64, torch.bool
    tensors(values, {"xy": ((count, 2, joints, 2), f32), "xy_valid": ((count, 2, joints), mask),
                     "target_mm": ((count, joints, 3), f32), "target_valid": ((count, joints), mask),
                     "camera": ((count, 2, 9), f32), "rotation64": ((count, 2, 3, 3), f64),
                     "translation64": ((count, 2, 3), f64)}, "cache")
    require(images.shape == (count, 2, 3, 128, 128) and images.dtype == np.float32
            and bool(np.isfinite(images).all()) and 0 <= images.min() <= images.max() <= 1, "RGB image contract")
    del images
    rotation = values["rotation64"]
    eye = torch.eye(3, dtype=f64)
    require(torch.allclose(rotation @ rotation.transpose(-1, -2), eye, rtol=0, atol=1e-9)
            and torch.allclose(torch.linalg.det(rotation), torch.ones((count, 2), dtype=f64), rtol=0, atol=1e-9),
            "cached proper world-to-camera rotation")
    require(spec["coordinates"]["mm_per_raw_unit"] == 10
            and bool(values["target_valid"].any()) and bool(values["xy_valid"].any()), "cache units / valid targets")
    return {"valid_3d": int(values["target_valid"].sum()), "valid_2d": int(values["xy_valid"].sum())}


def features_from_group(path, folders, clips, joints):
    results = []
    with BundleView(path) as view:
        require(view.members.keys() == {f"{folder}/{name}" for folder in folders
                                         for name in ("receipt.json", "clip.pt")}, "feature group member roster")
        for folder in folders:
            row = clips[folder]
            receipt = json.loads(view.read(f"{folder}/receipt.json", 16 * 1024))
            check_json_pin(receipt, "archvis_v12_predicted_feature_v1")
            require(receipt["input_sha256"] == FEATURE_IDENTITY and receipt["split"] == row["split"]
                    and receipt["key"] == [row["sequence"], row["start_frame"]]
                    and receipt["real_frames"] == row["real_frames"], "feature clip identity mismatch")
            payload = view.read(f"{folder}/clip.pt", 4 * MIB, receipt["tensor"])
            values = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            f32, f64, mask = torch.float32, torch.float64, torch.bool
            tensors(values, {"features": ((1, 50, joints * 14), f32), "camera": ((1, 50, 2, 9), f32),
                             "target_mm": ((1, 50, joints, 3), f32), "target_valid": ((1, 50, joints), mask),
                             "time_valid": ((1, 50), mask), "seconds": ((1, 50), f64)}, "feature")
            real = row["real_frames"]
            require(torch.equal(values["time_valid"][0], torch.arange(50) < real)
                    and all(not bool(values[key][:, real:].any())
                            for key in ("features", "camera", "target_mm", "target_valid"))
                    and torch.equal(values["seconds"][0], torch.arange(row["start_frame"],
                                                                       row["start_frame"] + 50, dtype=f64) / 25),
                    "feature padding or nominal timestamp mismatch")
            results.append({"folder": folder, "key": receipt["key"], "real_frames": real})
    return results


def run(kit, project, index, workspace):
    meter = project.meter
    meter.mark("inputs/source-split")
    source, diagnostic, _, comparison = native_evidence(kit, project, index)
    doc, spec, plan = load_scope(workspace, source)
    split_counts = {split: sum(row["split"] == split for row in doc["records"])
                    for split in ("train", "validation", "test")}
    require(split_counts == {"train": 119586, "validation": 16425, "test": 0}
            and plan["counts"] == diagnostic["counts"]
            and {key: list(value) for key, value in shared_clips(plan).items()} == diagnostic["common_clips"],
            "manifest split/plan scope")
    # The original plan also sealed runtime free disk and its derived fit flag.
    stable_plan = {k: v for k, v in plan.items() if k not in ("content_sha256", "free_disk_bytes", "resource_fit")}
    by_folder = {}
    for split in ("train", "validation"):
        rows = sorted((row for row in plan["clips"] if row["split"] == split),
                      key=lambda row: (row["sequence"], row["start_frame"]))
        by_folder.update({f"{split}-{i:05d}": row for i, row in enumerate(rows)})
    meter.mark("inputs/cache-chain-and-inventory")
    journal, anchor, entries = pinned_journal(project, index, source, doc, spec, meter, workspace)
    package_names = {}
    for entry in entries:
        i = entry["index"]
        package = entry["package"]
        name = f"{BULK if i else LEGACY}/{package['name']}"
        project.assert_size(name, package)
        package_names[i] = name
    meter.mark("inputs/cache-shard-samples")
    samples = []
    for i in SHARD_SAMPLES:
        entry = entries[i]
        package = entry["package"]
        with tempfile.TemporaryDirectory(dir=workspace, prefix="shard-") as scratch:
            root = Path(scratch)
            copy = staged_archive(project, package_names[i], package, root)
            folder = root / f"shard-{i:05d}"
            extract_pinned_zip(copy, package["sha256"], folder, max_zip_bytes=PACKAGE_CAP,
                               max_output_bytes=PACKAGE_CAP, max_members=3)
            receipt = v2_cache.receipt(root, i, anchor["binding"] if i == 0 else journal.binding,
                                       rows_for(doc, i), meter)
            require(receipt["content_sha256"] == package["receipt_sha256"], "cache shard receipt mismatch")
            stats = cache_tensors(folder, entry["frames"], len(spec["joint_names"]), spec)
        samples.append({"path": package_names[i], "sha256": package["sha256"], "frames": entry["frames"], **stats})
    meter.mark("inputs/feature-inventory")
    groups = list(feature_groups(project, index["members"]).values())
    require(len(groups) == index["features"]["groups"]
            and sum(len(folders) for _, _, _, folders in groups) == len(by_folder)
            and diagnostic["timing"]["feature_identity"] == index["features"]["identity"] == FEATURE_IDENTITY,
            "feature store inventory scope")
    for _, name, pin, _ in groups:
        project.assert_size(name, pin)
    meter.mark("inputs/feature-group-samples")
    feature_samples = []
    for i in GROUP_SAMPLES:
        _, name, pin, folders = groups[i]
        with tempfile.TemporaryDirectory(dir=workspace, prefix="features-") as scratch:
            copy = staged_archive(project, name, pin, Path(scratch))
            clips = features_from_group(copy, folders, by_folder, len(spec["joint_names"]))
        feature_samples.append({"path": name, "sha256": pin["sha256"], "clips": clips})
    return {"manifest": doc["content_sha256"], "original_reported_plan_sha256": diagnostic["plan_sha256"],
            "observed_local_plan_sha256": plan["content_sha256"],
            "plan_without_runtime_disk_sha256": sha(canonical(stable_plan)), "splits": split_counts,
            "cache_boundary": entries[-1]["content_sha256"], "cache_files_size_checked": len(entries),
            "cache_selected_sha_checked": samples, "feature_identity": FEATURE_IDENTITY,
            "frontend_checkpoint_sha256": comparison["binding"]["model"]["frontend"],
            "feature_files_size_checked": len(groups), "feature_selected_sha_checked": feature_samples,
            "nominal_fps": 25, "producer_motion_independence": "unverified"}

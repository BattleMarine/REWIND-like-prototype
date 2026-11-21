"""Isolated v2 RGB cache resolution comparison on pinned train records."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m2.data import encode_camera
from rewind_prepare.coordinates import camera_transform
from rewind_prepare.extracted import IMAGE_SIZE, _pose_targets, _raw_pose
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_cache as cache, v2_contract as contract, v2_retention as retention
from rewind_v1.data import _field, frame_image
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, read_manifest
from scripts.v2_ordered_prefetch import elapsed, ordered_sources


MIB = 1024**2
GIB = 1024**3
SECONDS = 3600
WORKERS = 8
SOURCE_CAP = 4*GIB
REPORT_CAP = 32*MIB
SCENARIOS = ("ArchVisInterior_ArchVis_RT", "CityPark_Showcase", "DowntownWest_Demo_Environment")


def selection(records, count=128):
    result = {}
    for scenario in SCENARIOS:
        prefix = scenario + "/"
        rows = [r for r in records if r["split"] == "train" and r["sequence_id"].startswith(prefix)][:count]
        if len(rows) != count:
            raise ValueError("resolution_ab_train_coverage")
        result[scenario] = rows
    return result


def selection_pin(groups):
    return {scenario: {"records_sha256": digest(rows), "count": len(rows),
                       "source_bytes": sum(p["size_bytes"] for row in rows for p in row["files"].values()),
                       "first": [rows[0]["sequence_id"], rows[0]["frame_index"]],
                       "last": [rows[-1]["sequence_id"], rows[-1]["frame_index"]]}
            for scenario, rows in groups.items()}


def decode(raw, row, spec, size):
    doc = _raw_pose(raw["pose"], row["frame_index"])
    fields = _pose_targets(doc, spec)
    xy, valid = zip(*(_field(field, 2) for field in fields[:2]))
    xyz, xyz_valid = _field(fields[2], 3)
    coords = spec["coordinates"]
    cameras = {side: camera_transform(doc["camera_"+side], coords["mm_per_raw_unit"],
                                     coords["raw_consistency_tolerance"])[0] for side in ("left", "right")}
    return frame_image(raw, [size, size]), {
        "xy": scale_xy(torch.stack(xy), IMAGE_SIZE, [size, size]), "xy_valid": torch.stack(valid),
        "target_mm": xyz, "target_valid": xyz_valid, "camera": encode_camera(cameras, 100.),
        "rotation64": torch.tensor([c["rotation_matrix"] for c in cameras.values()], dtype=torch.float64),
        "translation64": torch.tensor([c["translation"] for c in cameras.values()], dtype=torch.float64),
    }


def stage_inputs(groups, rgb, pose, stage, ledger, metrics):
    expected = sum(item["source_bytes"] for item in selection_pin(groups).values())
    if expected > SOURCE_CAP:
        raise ValueError("resolution_ab_source_cap")
    ledger.reserve("write", expected)
    for rows in groups.values():
        with ordered_sources(rows, rgb, pose, WORKERS, ledger, metrics) as reader:
            for row, raw in zip(rows, reader, strict=True):
                for role, pin in row["files"].items():
                    dest = stage / ("pose" if role == "pose" else "rgb") / pin["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open("xb") as handle:
                        handle.write(raw[role])
                    ledger.observed("write", len(raw[role]), persist=False)
                del raw
    ledger.persist()


def part(path, ledger):
    size = path.stat().st_size
    return {"bytes": size, "sha256": _hash_file(path, expected_size=size, check_time=ledger.check)}


def write_variant(rows, spec, stage, out, size, ledger, *, source_roots=None, shard_identity=None):
    started = time.monotonic()
    out.mkdir()
    stats = {}
    shape = (len(rows), 2, 3, size, size)
    ledger.reserve("write", int(np.prod(shape))*4 + 2*MIB)
    images = np.lib.format.open_memmap(out/"images.npy", mode="w+", dtype=np.float32, shape=shape)
    targets = []
    sample = None
    try:
        rgb, pose = source_roots if source_roots is not None else (stage/"rgb", stage/"pose")
        with ordered_sources(rows, rgb, pose, WORKERS, ledger, stats) as reader:
            for i, (row, raw) in enumerate(zip(rows, reader, strict=True)):
                with elapsed(stats, "transform_seconds"):
                    image, target = decode(raw, row, spec, size)
                if i == len(rows)//2:
                    sample = image.numpy().copy()
                images[i] = image.numpy()
                targets.append(target)
                del raw
        with elapsed(stats, "serialize_and_hash_seconds"):
            images.flush()
    finally:
        del images
    with elapsed(stats, "serialize_and_hash_seconds"):
        with (out/"images.npy").open("r+b") as handle:
            os.fsync(handle.fileno())
        tensors = {key: torch.stack([t[key] for t in targets]) for key in targets[0]}
        if not all(torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError("resolution_ab_nonfinite")
        with (out/"targets.pt").open("xb") as handle:
            torch.save(tensors, handle)
            handle.flush()
            os.fsync(handle.fileno())
        files = {name: part(out/name, ledger) for name in ("images.npy", "targets.pt")}
        receipt = seal({"schema": "rewind_v2_resolution_ab_shard_v1", **(shard_identity or {}), "size": size,
                        "records_sha256": digest(rows), "files": files, "image_shape": list(shape),
                        "image_dtype": "float32", "targets": {n: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                                              for n, v in tensors.items()}})
        _atomic(out/"receipt.json", canonical(receipt))
        ledger.observed("write", sum(p["bytes"] for p in files.values()))
    with elapsed(stats, "zip_seconds"):
        payload_cap = sum(p["bytes"] for p in files.values())+MIB
        ledger.reserve("write", payload_cap)
        with zipfile.ZipFile(out/"shard.zip", "x") as archive:
            for name in ("images.npy", "targets.pt", "receipt.json"):
                info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
                info.external_attr = 0o100600 << 16
                with (out/name).open("rb") as src, archive.open(info, "w", force_zip64=True) as dst:
                    while block := src.read(MIB):
                        ledger.check()
                        dst.write(block)
        archive_pin = part(out/"shard.zip", ledger)
        ledger.observed("write", archive_pin["bytes"])
    stats["variant_wall_seconds"] = time.monotonic() - started
    return {"receipt": receipt, "zip": archive_pin, "timing": stats}, sample


def panel(samples, row, text, path):
    canvas = Image.new("RGB", (1068, 588), "#17212d")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), text + " | frame " + str(row["frame_index"]), fill="white")
    for col, size in enumerate((128, 256)):
        for side, label in enumerate(("left", "right")):
            rgb = np.rint(samples[size][side].transpose(1, 2, 0)*255).clip(0,255).astype(np.uint8)
            image = Image.fromarray(rgb)
            crop = image.crop((size//4, size//4, 3*size//4, 3*size//4))
            x, y = 10+col*530, 40+side*270
            draw.text((x, y), f"{size} {label} / full", fill="white")
            draw.text((x+262, y), "center 50% / 2x", fill="white")
            for i, item in enumerate((image, crop)):
                canvas.paste(item.resize((256,256), Image.Resampling.NEAREST), (x+i*262, y+12))
    canvas.save(path)


def verify_reference(rows, spec, stage, variant_root, ledger):
    """Re-encode the same sample with the actual v2 128 emitter outside the A/B times."""
    reference = variant_root.parent/"reference"
    reference.mkdir()
    binding = {"scope": "resolution_ab_128_reference"}
    item = cache.publish_shard(reference, 0, binding, rows, spec, stage/"rgb", stage/"pose", ledger)
    actual = json.loads((variant_root/"receipt.json").read_bytes())["files"]
    if item["files"] != actual:
        raise ValueError("resolution_ab_production_128_mismatch")
    return item["files"]


def write_reports(report, directory):
    _atomic(directory/"comparison.json", canonical(report))
    with (directory/"comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ("scenario", "size", "frames", "image_bytes", "targets_bytes", "zip_bytes",
                   "read_wait_seconds", "transform_seconds", "serialize_and_hash_seconds", "zip_seconds", "variant_wall_seconds")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["comparisons"]:
            shape = row["receipt"]["image_shape"]
            writer.writerow({"scenario": row["scenario"], "size": shape[-1], "frames": shape[0],
                             "image_bytes": row["receipt"]["files"]["images.npy"]["bytes"],
                             "targets_bytes": row["receipt"]["files"]["targets.pt"]["bytes"], "zip_bytes": row["zip"]["bytes"],
                             **{key: row["timing"][key] for key in columns[6:]}})


def run(args, pins):
    torch.set_num_threads(1)
    report = {"status": "failed", "comparisons": [], "reference_128": {}, "manifest_sha256": pins["manifest_content_sha256"],
              "worker_count": WORKERS, "torch_threads": torch.get_num_threads(), "device": "cpu",
              "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": platform.python_version(),
                              "pillow": Image.__version__},
              "scopes": {"stage": "Drive raw once", "variants": "same staged local input", "operating_cache": "separate experiment"}}
    output = args.work/"comparison"
    output.mkdir()
    started = time.monotonic()
    try:
        if available_ram() < 4*GIB or shutil.disk_usage(args.work).free < 6*GIB:
            raise OSError("resolution_ab_local_resources")
        if digest(contract.source()) != pins["consumer_source_identity"]:
            raise ValueError("resolution_ab_source_pin")
        for name in ("run_v2_resolution_ab.py", "v2_ordered_prefetch.py"):
            key = "scripts/"+name
            if contract.sha(_regular_read(Path(__file__).with_name(name), MIB)) != pins["consumer_overlay"][key]:
                raise ValueError("resolution_ab_overlay_pin")
        root = contract.ROOT
        with elapsed(report, "manifest_seconds"):
            extract_pinned_zip(root/pins["embedded_manifest"], pins["archive_sha256"], args.work/"manifest",
                               max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
            doc, spec = read_manifest(args.work/"manifest"/MANIFEST_NAME, pins, root/"reference")
            groups = selection(doc["records"])
            report["selection"] = selection_pin(groups)
            if report["selection"] != pins["sample"]:
                raise ValueError("resolution_ab_sample_pin")
        cfg = contract.config()
        cfg = {**cfg, "seconds_per_invocation": args.seconds, "terminal_reserve_seconds": 120,
               "resources": {**cfg["resources"], "read_bytes": 16*GIB, "write_bytes": 16*GIB}}
        ledger = Ledger(args.work/"usage", {"manifest": doc["content_sha256"], "sample": digest(report["selection"])}, cfg)
        stage = args.work/"raw"
        with ledger.invocation("resolution_ab"):
            ledger.started = started
            report["stage_io"] = {}
            report["ratios_256_over_128"] = {}
            with elapsed(report, "stage_wall_seconds"):
                stage_inputs(groups, args.rgb, args.pose, stage, ledger, report["stage_io"])
            for i, (scenario, rows) in enumerate(groups.items()):
                variants = args.work/f"scenario-{i}"
                variants.mkdir()
                samples = {}
                variants_report = {}
                for size in ((128,256) if i%2 == 0 else (256,128)):
                    result, samples[size] = write_variant(rows, spec, stage, variants/str(size), size, ledger)
                    variants_report[size] = result
                    report["comparisons"].append({**result, "scenario": scenario})
                small, large = variants_report[128], variants_report[256]
                if small["timing"]["source_bytes"] != large["timing"]["source_bytes"]:
                    raise ValueError("resolution_ab_variant_read_mismatch")
                report["ratios_256_over_128"][scenario] = {
                    "image_bytes": large["receipt"]["files"]["images.npy"]["bytes"]/small["receipt"]["files"]["images.npy"]["bytes"],
                    "zip_bytes": large["zip"]["bytes"]/small["zip"]["bytes"],
                    "variant_wall_seconds": large["timing"]["variant_wall_seconds"]/small["timing"]["variant_wall_seconds"],
                }
                panel(samples, rows[len(rows)//2], scenario, output/f"{scenario}.png")
                with elapsed(report, "reference_128_wall_seconds"):
                    report["reference_128"][scenario] = verify_reference(rows, spec, stage, variants/"128", ledger)
            if pins.get("records_sha256") == report["selection"][SCENARIOS[0]]["records_sha256"]:
                archived = pins["shard_files"]
                kernels = pins["producer_environment"]
                report["archvis_original_128"] = {
                    "published_receipt_sha256": pins["package"]["receipt_sha256"],
                    "same_runtime_versions": all(report["environment"][key] == kernels[key]
                                                 for key in ("torch", "numpy", "pillow")),
                    "files_match": report["reference_128"][SCENARIOS[0]] == archived,
                    "published_files": archived,
                }
            report["status"] = "passed"
            report["wall_before_retention_seconds"] = time.monotonic() - started
            write_reports(report, output)
            archive_path = args.work/"resolution-ab-results.zip"
            with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output.iterdir()):
                    archive.write(path, path.name)
            pin = part(archive_path, ledger)
            if pin["bytes"] > REPORT_CAP:
                raise ValueError("resolution_ab_report_cap")
            args.drive_results.mkdir(exist_ok=True)
            path = args.drive_results/f"{args.work.name}-{pin['sha256']}.zip"
            with elapsed(report, "retention_wall_seconds"):
                retention.copy_verified(archive_path, path, pin, ledger, ledger.clock()+min(120, ledger.remaining()))
                ledger.reserve("read", pin["bytes"])
                if part(path, ledger) != pin:
                    raise ValueError("resolution_ab_drive_readback")
                ledger.observed("read", pin["bytes"])
            report["retention"] = {"name": path.name, **pin}
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        write_reports(report, output)
        raise
    finally:
        _atomic(args.work/"result.json", canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "work", "rgb", "pose", "drive-results"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if not 120 < args.seconds <= SECONDS:
        parser.error("invalid independent worker time budget")
    roots = [p.resolve() for p in (args.work, args.rgb, args.pose, args.drive_results)]
    if (args.drive_results.name != "v2_resolution_ab_041837Z" or args.drive_results.parent.name != "cache"
            or args.drive_results.is_symlink()
            or any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[:i])):
        parser.error("trial uses a separate fixed output directory")
    run(args, json.loads(_regular_read(args.pins, 64*MIB)))


if __name__ == "__main__":
    main()

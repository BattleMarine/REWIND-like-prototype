"""Read only the pinned v2 cache shard and round-trip one tensor batch on CUDA."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import PIL
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_prepare.temporal_selection import check_pin
from rewind_v1 import v2_cache as cache, v2_contract as contract
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, MIB, read_manifest


MAX_SECONDS = 1800
RUNTIME_KEYS = ("torch", "numpy", "pillow")


class ProbeBudget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds
        self.remote_bytes = 0
        self.local_read_bytes = 0

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError("v2_gpu_cache_probe_time")

    def local_io(self, kind, size):
        if kind != "read":
            raise ValueError("v2_gpu_cache_probe_read_only")
        self.local_read_bytes += size


class VerifiedShard(cache.Cache):
    """Use Cache.frame for a verified prefix without creating a complete-cache catalog."""

    def __init__(self, root, spec, rows, budget):
        self.root, self.doc, self.spec, self.ledger = root, {"records": rows}, spec, budget
        self.index = {(row["sequence_id"], row["frame_index"]): i for i, row in enumerate(rows)}
        self.loaded = OrderedDict()


def assess_binding(original, current):
    differences = {key: {"producer": original.get(key), "runtime": current.get(key)}
                   for key in sorted(set(original) | set(current))
                   if key not in original or key not in current or original[key] != current[key]}
    if not differences:
        status = "match"
    elif set(differences) <= set(RUNTIME_KEYS) and original.keys() == current.keys():
        status = "runtime_mismatch"
    else:
        status = "non_runtime_mismatch"
    return {"status": status, "differences": differences, "strict_binding_match": not differences}


def load_catalog(raw, pins):
    if len(raw) != pins["catalog_bytes"] or contract.sha(raw) != pins["catalog_file_sha256"]:
        raise ValueError("v2_gpu_catalog_file_pin")
    doc = json.loads(raw)
    check_pin(doc, "rewind_v2_retained_catalog_v1", pins["catalog_content_sha256"])
    if (doc["status"] != "remote_verified" or doc["packages"] != [pins["package"]]
            or doc["completed_frames"] != cache.CHUNK or digest(doc["binding"]) != pins["producer_binding_sha256"]):
        raise ValueError("v2_gpu_catalog_package_or_binding")
    return doc


def copy_package(source, destination, package, budget):
    if source.is_symlink() or not source.is_file() or source.stat().st_size != package["bytes"]:
        raise ValueError("v2_gpu_shard_archive_size")
    size = package["bytes"]
    if not 1 <= size <= 64 * MIB:
        raise ValueError("v2_gpu_shard_archive_cap")
    hasher = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as target:
        while size:
            budget.check()
            block = incoming.read(min(MIB, size))
            if not block:
                raise ValueError("v2_gpu_shard_archive_short")
            budget.remote_bytes += len(block)
            size -= len(block)
            hasher.update(block)
            target.write(block)
        if incoming.read(1):
            budget.remote_bytes += 1
            raise ValueError("v2_gpu_shard_archive_grew")
    if hasher.hexdigest() != package["sha256"]:
        raise ValueError("v2_gpu_shard_archive_sha")


def verified_batch(root, doc, spec, catalog, pins, budget):
    rows = doc["records"][:cache.CHUNK]
    package = pins["package"]
    receipt = cache.receipt(root, 0, catalog["binding"], rows, budget)
    if (receipt["content_sha256"] != package["receipt_sha256"] or receipt["frames"] != cache.CHUNK
            or receipt["records_sha256"] != pins["records_sha256"] or receipt["files"] != pins["shard_files"]):
        raise ValueError("v2_gpu_shard_receipt_pin")
    keys = [(row["sequence_id"], row["frame_index"]) for row in rows[:2]]
    if len(keys) != 2 or any(row["split"] != "train" for row in rows[:2]):
        raise ValueError("v2_gpu_probe_batch_keys")
    reader = VerifiedShard(root, spec, rows, budget)
    try:
        values = [reader.frame(key, "train") for key in keys]
    finally:
        reader.close()
    tensors = {"image": torch.stack([image for image, _ in values])}
    for name in values[0][1]:
        tensors[name] = torch.stack([target[name] for _, target in values])
    masks = {name: tensor.sum(dim=-1).tolist() for name, tensor in tensors.items() if name.endswith("_valid")}
    if not all(v > 0 for pair in masks["xy_valid"] for v in pair) or not all(masks["target_valid"]):
        raise ValueError("v2_gpu_probe_empty_supervision")
    return keys, tensors, masks


def cuda_round_trip(batch):
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    gpu = {name: tensor.to(device, non_blocking=False) for name, tensor in batch.items()}
    torch.cuda.synchronize(device)
    transferred = time.monotonic()
    returned = {name: tensor.cpu() for name, tensor in gpu.items()}
    torch.cuda.synchronize(device)
    completed = time.monotonic()
    comparisons = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(gpu[name].device),
               "exact": gpu[name].device == device and gpu[name].dtype == tensor.dtype
               and returned[name].dtype == tensor.dtype and torch.equal(returned[name], tensor)}
        for name, tensor in batch.items()
    }
    return {
        "device_name": props.name, "device": str(device), "vram_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor], "tensors": comparisons,
        "h2d_bytes": sum(t.numel() * t.element_size() for t in batch.values()),
        "h2d_synchronized_seconds": transferred - started,
        "roundtrip_synchronized_seconds": completed - started,
        "all_exact": all(item["exact"] for item in comparisons.values()),
    }


def run(args, pins):
    budget = ProbeBudget(args.seconds)
    report = {
        "status": "failed", "stage": "preflight", "source_revision": pins["revision"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(torch.__version__), "numpy": str(np.__version__),
                        "pillow": PIL.__version__, "torch_cuda_build": torch.version.cuda},
        "producer_environment": pins["producer_environment"],
        "scope": "one_read_only_cuda_batch", "strict_restore_invoked": False,
        "limits": {"worker_seconds": args.seconds, "drive_input_bytes": pins["catalog_bytes"] + pins["package"]["bytes"] + 1},
    }
    try:
        torch.set_num_threads(1)
        cfg_raw = _regular_read(contract.ROOT / "configs/rewind_v2.json", MIB)
        if contract.sha(cfg_raw) != pins["config_sha256"] or digest(contract.source()) != pins["source_identity"]:
            raise ValueError("v2_gpu_frozen_source_config_pin")
        cfg = json.loads(cfg_raw)
        if available_ram() < 4 * 1024 * MIB:
            raise MemoryError("v2_gpu_ram_preflight")
        report["environment"]["cuda_available"] = torch.cuda.is_available()
        input_dir = args.input_dir
        if input_dir.is_symlink() or not input_dir.is_dir():
            raise ValueError("v2_gpu_missing_fixed_drive_input")
        report["stage"] = "catalog"
        budget.check()
        raw = _regular_read(input_dir / "catalog.json", 64 * 1024, pins["catalog_bytes"])
        budget.remote_bytes += len(raw)
        catalog = load_catalog(raw, pins)
        report["catalog_file_sha256"] = pins["catalog_file_sha256"]
        report["stage"] = "manifest"
        source = contract.ROOT
        extract_pinned_zip(source / pins["embedded_manifest"], pins["archive_sha256"], args.work / "manifest",
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        doc, spec = read_manifest(args.work / "manifest" / MANIFEST_NAME, pins, source / "reference")
        budget.check()
        report["stage"] = "binding"
        binding = assess_binding(catalog["binding"], cache.binding(doc, spec, cfg))
        report["strict_restore_binding"] = binding
        if binding["status"] == "non_runtime_mismatch":
            raise ValueError("v2_gpu_non_runtime_binding_mismatch")
        if not report["environment"]["cuda_available"]:
            raise RuntimeError("v2_gpu_cuda_required")
        report["stage"] = "package"
        package = pins["package"]
        archive = args.work / "shard.zip"
        copied = time.monotonic()
        copy_package(input_dir / package["name"], archive, package, budget)
        report["drive_shard_copy_seconds"] = time.monotonic() - copied
        report["shard"] = package
        root = args.work / "shard"
        root.mkdir()
        extract_pinned_zip(archive, package["sha256"], root / "shard-00000",
                           max_zip_bytes=64*MIB, max_output_bytes=64*MIB, max_members=3)
        report["stage"] = "batch"
        keys, batch, masks = verified_batch(root, doc, spec, catalog, pins, budget)
        report["batch"] = {"keys": [list(key) for key in keys], "split": "train", "valid_counts": masks}
        budget.check()
        report["stage"] = "cuda"
        report["cuda"] = cuda_round_trip(batch)
        budget.check()
        if not report["cuda"]["all_exact"]:
            raise ValueError("v2_gpu_cuda_batch_mismatch")
        report.update(status="passed", stage="complete")
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        raise
    finally:
        report["drive_input_bytes_observed"] = budget.remote_bytes
        report["local_logical_bytes_observed"] = budget.local_read_bytes
        report["worker_seconds"] = time.monotonic() - budget.started
        _atomic(args.work / "result.json", canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "work", "input-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seconds", required=True, type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= MAX_SECONDS:
        parser.error("seconds outside the probe limit")
    work, source = args.work.resolve(), args.input_dir.resolve()
    if work.is_relative_to(source) or source.is_relative_to(work):
        parser.error("work must be separate from the cache input directory")
    run(args, json.loads(_regular_read(args.pins, 32 * 1024)))


if __name__ == "__main__":
    main()

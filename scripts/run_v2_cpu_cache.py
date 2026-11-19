"""Prepare and retain one additional canonical v2 cache shard on CPU."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil
import time
import uuid

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import check_pin
from rewind_v1 import v2_cache as cache, v2_contract as contract, v2_retention as retention
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip


MIB = 1024**2
MANIFEST_CAP = 256 * MIB
ARCHIVE_CAP = 64 * MIB
SECONDS = 28_800
RESERVE = 1_320
MANIFEST_NAME = "audited_manifest.json"


def read_manifest(path, pins, reference):
    spec_raw = _regular_read(reference / "spec.json", MIB)
    if contract.sha(spec_raw) != pins["spec_sha256"]:
        raise ValueError("v2_pilot_spec_pin")
    raw = _regular_read(path, MANIFEST_CAP, pins["manifest_bytes"])
    if contract.sha(raw) != pins["manifest_file_sha256"]:
        raise ValueError("v2_pilot_manifest_file_pin")
    doc = json.loads(raw)
    del raw
    if (doc["content_sha256"] != pins["manifest_content_sha256"]
            or len(doc["records"]) != pins["frames"]):
        raise ValueError("v2_pilot_manifest_content_pin")
    spec = json.loads(spec_raw)
    owners, split_sources = contract.official_splits(reference / "split", spec, contract.SplitReader())
    return contract.validate(doc, spec, owners, split_sources), spec


def read_catalog(path, schema):
    raw = _regular_read(path, 32 * MIB)
    doc = json.loads(raw)
    check_pin(doc, schema, doc.get("content_sha256"))
    return contract.sha(raw), doc


def retained_prefix(retained, expected_sha):
    path = retained / "catalog.json"
    if expected_sha:
        actual, doc = read_catalog(path, "rewind_v2_retained_catalog_v1")
        if actual != expected_sha:
            raise ValueError("v2_pilot_catalog_pin")
        return doc
    if path.exists() or path.is_symlink():
        raise ValueError("v2_pilot_existing_catalog_requires_pin")
    return None


def probe_retention(retained):
    retained.mkdir(parents=True, exist_ok=True)
    path = retained / (".probe-" + uuid.uuid4().hex)
    try:
        _atomic(path, b"v2 cache retention preflight")
        if _regular_read(path, 128) != b"v2 cache retention preflight":
            raise OSError("v2_pilot_retention_readback")
    finally:
        path.unlink(missing_ok=True)


def run(args, pins):
    overall_started = time.monotonic()
    torch.set_num_threads(1)
    rgb, pose, work, retained = (p.resolve(strict=True) for p in
                                 (args.rgb_root, args.pose_root, args.work, args.retained.parent))
    retained /= args.retained.name
    if args.retained.is_symlink() or not all(p.is_dir() for p in (rgb, pose, work)):
        raise ValueError("v2_pilot_directories")
    for path in (rgb, pose, work):
        if path.is_relative_to(retained) or retained.is_relative_to(path):
            raise ValueError("v2_pilot_overlapping_output")
    if any(work.is_relative_to(root) or root.is_relative_to(work) for root in (rgb, pose)):
        raise ValueError("v2_pilot_overlapping_work")

    source = digest(contract.source())
    cfg_raw = _regular_read(contract.ROOT / "configs/rewind_v2.json", MIB)
    if contract.sha(cfg_raw) != pins["config_sha256"] or source != pins["source_identity"]:
        raise ValueError("v2_pilot_source_config_pin")
    cfg = json.loads(cfg_raw)
    ram_before = available_ram()
    if ram_before < cfg["resources"]["rss_bytes"]:
        raise MemoryError("v2_pilot_available_ram")
    if shutil.disk_usage(work).free < pins["manifest_bytes"] + 192 * MIB:
        raise ValueError("v2_pilot_disk_before_manifest")
    if args.manifest_archive.stat().st_size != pins["archive_bytes"]:
        raise ValueError("v2_pilot_manifest_archive_size")
    archive_info = extract_pinned_zip(
        args.manifest_archive, pins["archive_sha256"], work / "manifest",
        max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
    doc, spec = read_manifest(work / "manifest" / MANIFEST_NAME, pins, contract.ROOT / "reference")
    total = len(doc["records"])
    cache_root = args.cache.resolve()
    if (args.cache.is_symlink() or cache_root.parent != work.parent
            or any(cache_root.is_relative_to(p) or p.is_relative_to(cache_root) for p in (rgb, pose, work, retained))):
        raise ValueError("v2_pilot_cache_outside_workspace")
    catalog_sha = args.retained_catalog_sha256
    remote = retained_prefix(retained, catalog_sha)
    prior_count = len(remote["packages"]) if remote is not None else 0
    target = prior_count + 1
    first = prior_count * cache.CHUNK
    rows = doc["records"][first:first + cache.CHUNK]
    if not rows:
        raise ValueError("v2_pilot_no_next_shard")
    if first == 0 and len(rows) != cache.CHUNK:
        raise ValueError("v2_pilot_first_chunk_size")
    bind = cache.binding(doc, spec, cfg)
    if remote is not None and remote["binding"] != bind:
        raise ValueError("v2_pilot_retained_binding_requires_matching_runtime_and_source")
    local_count = 0
    if (cache_root / "catalog.json").exists():
        _, local = read_catalog(cache_root / "catalog.json", "rewind_v2_local_catalog_v1")
        if local["binding"] != bind or len(local["shards"]) > target:
            raise ValueError("v2_pilot_local_catalog_prefix")
        local_count = len(local["shards"])

    # At most one new shard. New VM recovery may also download the missing prefix.
    source_bytes = sum(pin["size_bytes"] for row in rows for pin in row["files"].values())
    if first == 0 and source_bytes != pins["first_128_source_bytes"]:
        raise ValueError("v2_pilot_first_source_bytes_pin")
    shard_upper = 64 * MIB
    missing_prefix = max(0, prior_count - local_count)
    if (source_bytes + (missing_prefix * 2 + 3) * shard_upper > cfg["resources"]["read_bytes"]
            or (missing_prefix + 2) * shard_upper > cfg["resources"]["write_bytes"]):
        raise ValueError("v2_pilot_io_preflight")
    disk_before = shutil.disk_usage(work).free
    if ((target + 2) * shard_upper > cfg["resources"]["cache_bytes"]
            or disk_before < (missing_prefix + 3) * shard_upper):
        raise ValueError("v2_pilot_disk_preflight")

    # Charge worker boot and manifest validation against the same invocation.
    policy = {**cfg, "seconds_per_invocation": min(SECONDS, args.worker_seconds),
              "external_timeout_seconds": min(SECONDS, args.worker_seconds)}
    clock = time.monotonic
    ledger = Ledger(work / "usage", {"manifest": doc["content_sha256"], "config": digest(cfg),
                                     "source": source}, policy, clock)
    report = {
        "status": "failed", "scope": "one_shard", "execution_device": "cpu",
        "runtime": {**{key: bind[key] for key in ("torch", "numpy", "pillow")},
                    "python": platform.python_version(), "torch_cuda_build": torch.version.cuda},
        "source_identity": source, "manifest": archive_info,
        "manifest_file_sha256": pins["manifest_file_sha256"],
        "manifest_content_sha256": doc["content_sha256"], "source_revision": pins["revision"],
        "config_sha256": pins["config_sha256"], "spec_sha256": pins["spec_sha256"],
        "prior_retained_shards": prior_count, "target_shards": target, "total_frames": total,
        "target_start_index": first, "target_frames": len(rows), "pinned_source_bytes": source_bytes,
        "scenario_split_counts": dict(Counter(f"{r['sequence_id'].split('/')[0]}/{r['split']}" for r in rows)),
        "first_key": [rows[0]["sequence_id"], rows[0]["frame_index"]],
        "last_key": [rows[-1]["sequence_id"], rows[-1]["frame_index"]],
        "outer_limit_seconds": SECONDS, "worker_limit_seconds": policy["seconds_per_invocation"],
        "terminal_reserve_seconds": RESERVE, "resource_limits": cfg["resources"],
        "available_ram_bytes_at_preflight": ram_before, "local_free_bytes_after_manifest": disk_before,
        "worker_preflight_seconds": round(clock() - overall_started, 3),
    }
    try:
        with ledger.invocation("cpu_cache_one_shard"):
            ledger.started = overall_started
            ledger.check()
            probe_retention(retained)
            if retained_prefix(retained, catalog_sha) != remote:
                raise ValueError("v2_pilot_retained_prefix_changed")
            if local_count < prior_count:
                started = clock()
                report["restore"] = retention.restore_cache(retained, catalog_sha, cache_root, doc, spec, cfg, ledger)
                report["restore_seconds"] = round(clock() - started, 3)
            else:
                if remote is not None:
                    ledger.import_usage(remote["usage"])
                report["restore"] = {"status": "local_prefix", "verified_during_prepare": local_count}
            # Finish computation early enough for one package and verified copy.
            started = clock()
            read_before = ledger.current["read_bytes_observed"]
            write_before = ledger.current["write_bytes_observed"]
            ledger.cfg = {**policy, "terminal_reserve_seconds": RESERVE + 2 * cfg["copy_seconds"]}
            try:
                local = cache.prepare(doc, spec, cfg, rgb, pose, cache_root, ledger, max_shards=target)
            finally:
                ledger.cfg = policy
                report["prepare_seconds"] = round(clock() - started, 3)
                report["prepare_read_bytes"] = ledger.current["read_bytes_observed"] - read_before
            if len(local["shards"]) != target:
                raise ValueError("v2_pilot_local_shard_count")
            report["local_catalog_sha256"] = local["content_sha256"]
            report["local_completed_frames"] = local["completed_frames"]
            report["shard_receipt"] = local["shards"][-1]
            report["newly_prepared_frames"] = len(rows) if ledger.current["write_bytes_observed"] > write_before else 0
            started = clock()
            report["retention"] = retention.user_copy_cache(cache_root, retained, doc, spec, cfg, ledger)
            actual, remote = read_catalog(retained / "catalog.json", "rewind_v2_retained_catalog_v1")
            if (actual != report["retention"]["catalog_file_sha256"]
                    or len(remote["packages"]) != target or remote["local_catalog_sha256"] != local["content_sha256"]):
                raise ValueError("v2_pilot_retained_catalog_readback")
            package = remote["packages"][-1]
            ledger.reserve("read", package["bytes"])
            final_package = retained / package["name"]
            if final_package.is_symlink() or _hash_file(
                    final_package, expected_size=package["bytes"], check_time=ledger.check) != package["sha256"]:
                raise ValueError("v2_pilot_retained_package_readback")
            ledger.observed("read", package["bytes"])
            report["retained_package"] = package
            report["retention_seconds"] = round(clock() - started, 3)
            report["status"] = "remote_verified"
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:512]}
        raise
    finally:
        if ledger.path.exists():
            snapshot = ledger.snapshot()
            report["usage"] = snapshot["invocations"][-1]
            report["imported_usage"] = snapshot["imports"]
        report["worker_wall_seconds"] = round(clock() - overall_started, 3)
        if report.get("prepare_read_bytes"):
            elapsed = report["prepare_seconds"]
            if elapsed > 0:
                report["prepare_source_mib_per_second"] = round(report["prepare_read_bytes"] / MIB / elapsed, 3)
        report_path = work / "result.json"
        _atomic(report_path, canonical(report))
        print(f"v2 cache pilot report: {report_path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "manifest-archive", "rgb-root", "pose-root", "work", "cache", "retained"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--retained-catalog-sha256")
    parser.add_argument("--worker-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if not RESERVE + 1_200 < args.worker_seconds <= SECONDS:
        parser.error("worker time must include compute and retention reservations")
    pins = json.loads(_regular_read(args.pins, MIB))
    print(canonical(run(args, pins)).decode(), flush=True)


if __name__ == "__main__":
    main()

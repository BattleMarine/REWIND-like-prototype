"""Restore a pinned first shard and read one standard frontend batch on CUDA."""

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_v1 import v2_cache as cache, v2_contract as contract, v2_retention as retention
from rewind_v1.run_stages import frame_batch
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from rewind_v1.v2_training import ReadOnlyCacheFrames
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, MIB, read_manifest
from scripts.run_v2_gpu_cache_probe import MAX_SECONDS, cuda_round_trip, load_catalog


def run(args, pins):
    started = time.monotonic()
    report = {"status": "failed", "stage": "preflight", "scope": "read_only_first_shard",
              "restore_function": "rewind_v1.v2_retention.restore_cache",
              "batch_function": "rewind_v1.run_stages.frame_batch",
              "producer_revision": pins["revision"], "producer_binding_sha256": pins["producer_binding_sha256"],
              "consumer_source_identity": pins["consumer_source_identity"],
              "catalog_file_sha256": pins["catalog_file_sha256"]}
    ledger = None
    try:
        torch.set_num_threads(1)
        cfg_raw = _regular_read(contract.ROOT / "configs/rewind_v2.json", MIB)
        source_identity = digest(contract.source())
        if contract.sha(cfg_raw) != pins["config_sha256"] or source_identity != pins["consumer_source_identity"]:
            raise ValueError("v2_read_only_consumer_source_config")
        cfg = json.loads(cfg_raw)
        if available_ram() < 4 * 1024 * MIB:
            raise MemoryError("v2_read_only_available_ram")
        report["stage"] = "catalog_binding"
        catalog_raw = _regular_read(args.input_dir / "catalog.json", 64*1024, pins["catalog_bytes"])
        report["catalog_read_bytes"] = len(catalog_raw)
        catalog = load_catalog(catalog_raw, pins)
        producer = catalog["binding"]
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": str(cache.np.__version__), "pillow": __import__("PIL").__version__,
                   "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        report["runtime"] = runtime
        report["producer_runtime"] = pins["producer_environment"]
        if not runtime["cuda_available"]:
            raise RuntimeError("v2_read_only_cuda_required")
        if runtime["torch_cuda_build"] != "13.0":
            raise RuntimeError("v2_read_only_cuda_build_required")
        report["stage"] = "manifest"
        source = contract.ROOT
        extract_pinned_zip(source / pins["embedded_manifest"], pins["archive_sha256"], args.work / "manifest",
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        doc, spec = read_manifest(args.work / "manifest" / MANIFEST_NAME, pins, source / "reference")
        policy = cache.ReadOnlyFirstShard(pins["producer_binding_sha256"], source_identity,
                                          pins["package"]["receipt_sha256"])
        current = cache.binding(doc, spec, cfg)
        report["strict_binding_match"] = producer == current
        policy.verify(producer, current)
        report["allowed_binding"] = {"producer_torch": producer["torch"], "consumer_torch": current["torch"],
                                     "changed_source_files": sorted(k for k in producer["source"]
                                                                    if producer["source"][k] != current["source"][k])}
        report["stage"] = "restore"
        limit = pins["package"]["bytes"]
        limits = {**cfg["resources"], "read_bytes": limit * 2, "write_bytes": limit}
        ledger_cfg = {**cfg, "seconds_per_invocation": args.seconds, "terminal_reserve_seconds": 20,
                      "external_timeout_seconds": args.seconds, "resources": limits}
        origin = {"manifest": doc["content_sha256"], "source": digest(producer["source"]), "config": digest(cfg)}
        ledger = Ledger(args.work / "usage", origin, ledger_cfg)
        root = args.work / "restored"
        with ledger.invocation("read_only_first_shard_cuda"):
            ledger.started = started
            ledger.check()
            before = time.monotonic()
            restored = retention.restore_cache(args.input_dir, pins["catalog_file_sha256"], root, doc, spec, cfg,
                                               ledger, read_only=policy)
            report["restore_seconds"] = time.monotonic() - before
            report["restore"] = restored
            report["stage"] = "batch"
            data = ReadOnlyCacheFrames(root, doc, spec, cfg, ledger, policy=policy,
                                      local_catalog_sha256=restored["local_catalog_sha256"])
            rows = doc["records"][:2]
            keys = [(r["sequence_id"], r["frame_index"]) for r in rows]
            try:
                timing = {}
                values = frame_batch(data, keys, "train", [128, 128], ledger, timing)
            finally:
                data.cache.close()
            batch = dict(zip(("image", "xy", "xy_valid"), values, strict=True))
            counts = batch["xy_valid"].sum(dim=-1).tolist()
            if not all(count > 0 for views in counts for count in views):
                raise ValueError("v2_read_only_unsupervised_batch")
            report["batch"] = {"keys": keys, "split": "train", "valid_counts": counts, "timing": timing}
            report["stage"] = "cuda"
            report["cuda"] = cuda_round_trip(batch)
            ledger.check()
            if not report["cuda"]["all_exact"]:
                raise ValueError("v2_read_only_cuda_mismatch")
        report.update(status="passed", stage="complete",
                      drive_verified_input_bytes=2 * pins["catalog_bytes"] + limit)
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        raise
    finally:
        if ledger is not None and ledger.path.is_file():
            report["usage"] = ledger.snapshot()["invocations"][-1]
        report["worker_seconds"] = time.monotonic() - started
        _atomic(args.work / "result.json", canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "work", "input-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seconds", required=True, type=int)
    args = parser.parse_args(argv)
    if not 20 < args.seconds <= MAX_SECONDS:
        parser.error("seconds outside the restoration probe limit")
    work, source = args.work.resolve(), args.input_dir.resolve()
    if work.is_relative_to(source) or source.is_relative_to(work) or args.input_dir.is_symlink():
        parser.error("work must be separate from the cache input directory")
    run(args, json.loads(_regular_read(args.pins, 32 * 1024)))


if __name__ == "__main__":
    main()

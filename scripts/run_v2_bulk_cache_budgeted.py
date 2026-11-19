"""Continue only from the approved four-shard boundary under durable CPU budgets."""

import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_v1 import v2_cache as cache, v2_contract as contract
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import Journal, add_shards, writer
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, MIB, read_manifest
from scripts.run_v2_gpu_cache_probe import load_catalog
from scripts.v2_bulk_quota import BASELINE_SHA, BASELINE_SHARDS, CLOSURE, DRIVE_CAP, METADATA_CAP, TARGET_SHARDS, TIME_CAP
from scripts.v2_bulk_quota import DriveQuota, TimeQuota, baseline_entries


def check_layout(args):
    paths = (args.work, args.bulk, args.legacy, args.rgb, args.pose)
    for i, path in enumerate(paths):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("v2_quota_existing_directory_required")
        resolved = path.resolve()
        for other in paths[:i]:
            second = other.resolve()
            if resolved.is_relative_to(second) or second.is_relative_to(resolved):
                raise ValueError("v2_quota_separate_directories_required")


def bootstrap(args, pins, cfg):
    torch.set_num_threads(1)
    if available_ram() < 4 * 1024 * MIB or shutil.disk_usage(args.work).free < 384 * MIB:
        raise OSError("v2_quota_disk_or_ram_preflight")
    root = contract.ROOT
    extract_pinned_zip(root / pins["embedded_manifest"], pins["archive_sha256"], args.work / "manifest",
                       max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
    doc, spec = read_manifest(args.work / "manifest" / MANIFEST_NAME, pins, root / "reference")
    if (len(doc["records"]) + cache.CHUNK - 1) // cache.CHUNK != TARGET_SHARDS:
        raise ValueError("v2_quota_manifest_shard_count")
    anchor = load_catalog(_regular_read(args.legacy / "catalog.json", 64*1024, pins["catalog_bytes"]), pins)
    current = cache.binding(doc, spec, cfg)
    cache.ReadOnlyFirstShard(pins["producer_binding_sha256"], pins["consumer_source_identity"],
                             pins["package"]["receipt_sha256"]).verify(anchor["binding"], {**current, "torch": "2.14.1+cu130"})
    if current["torch"] != "2.14.1+cpu" or torch.version.cuda is not None:
        raise RuntimeError("v2_quota_observed_cpu_build_required")
    return Journal(args.bulk, args.legacy, doc, anchor, current), spec


def run(args, pins):
    report = {"status": "failed", "stage": "preflight", "baseline_sha256": BASELINE_SHA,
              "target_shards": TARGET_SHARDS, "time_cap_seconds": TIME_CAP, "drive_cache_cap_bytes": DRIVE_CAP}
    ledger = None
    try:
        check_layout(args)
        if digest(contract.source()) != pins["consumer_source_identity"] or pins["baseline_boundary_sha256"] != BASELINE_SHA:
            raise ValueError("v2_quota_source_or_baseline_pin")
        raw_cfg = _regular_read(contract.ROOT / "configs/rewind_v2.json", MIB)
        if contract.sha(raw_cfg) != pins["config_sha256"]:
            raise ValueError("v2_quota_config_pin")
        cfg = json.loads(raw_cfg)
        with writer(args.bulk, args.recover_writer) as owner:
            report["stage"] = "budget"
            baseline = baseline_entries(args.bulk)
            drive = DriveQuota(args.bulk, baseline)
            report["drive_used_start_bytes"] = drive.used
            report["minimum_verified_shards"] = len(baseline)
            report["minimum_verified_frames"] = sum(entry["frames"] for entry in baseline)
            quota = TimeQuota(args.bulk, owner)
            more = any(p.name not in drive.excluded for p in args.bulk.glob("boundary-*.json"))
            history = quota.history(initialize=not more, required_tip=args.budget_tip, enforce_tip=True)
            report["time_charged_before_seconds"] = history["charged_seconds"]
            report["unsettled_previous_calls"] = history["unsettled"]
            report["budget_tip_sha256"] = history["tip"]
            report["time_remaining_seconds"] = TIME_CAP - history["charged_seconds"]
            if drive.used + METADATA_CAP > DRIVE_CAP:
                report.update(status="paused", stage="budget_exhausted", stop_reason="drive_cumulative_budget")
                return report
            ticket = quota.reserve(history, time.monotonic() - args.started)
            if ticket is None:
                report.update(status="paused", stage="budget_exhausted", stop_reason="cpu_cumulative_budget")
                return report
            report["time_reserved_seconds"] = ticket["seconds"]
            report["budget_tip_sha256"] = ticket["content_sha256"]
            print("Durable budget reservation SHA-256:", ticket["content_sha256"], flush=True)
            try:
                deadline = args.started + ticket["seconds"]
                _atomic(args.work / "quota-grant.json", canonical({"deadline": deadline}))
                report["stage"] = "manifest"
                journal, spec = bootstrap(args, pins, cfg)
                ledger_cfg = {**cfg, "seconds_per_invocation": ticket["seconds"] - CLOSURE,
                              "external_timeout_seconds": ticket["seconds"]}
                ledger = Ledger(args.work / "usage", {"bulk_binding": digest(journal.binding), "config": digest(cfg)}, ledger_cfg)
                report["runtime"] = {key: journal.binding[key] for key in ("torch", "numpy", "pillow")}
                report["runtime"]["python"] = platform.python_version()
                report["stage"] = "append"
                try:
                    with ledger.invocation("v2_bulk_cpu_budgeted"):
                        ledger.started = args.started
                        ledger.check()
                        add_shards(journal, BASELINE_SHA, TARGET_SHARDS, args.work, spec, cfg, args.rgb, args.pose,
                                   ledger, report, owner_check=owner, minimum_shards=BASELINE_SHARDS, drive_quota=drive)
                except TimeoutError as error:
                    if "boundary_sha256" not in report:
                        raise
                    report["stop_reason"] = str(error)[:256] or "timeout"
                if report["stop_reason"] == "target_reached":
                    if report["completed_shards"] != TARGET_SHARDS or report["completed_frames"] != pins["frames"]:
                        raise ValueError("v2_quota_completion_count")
                    report.update(status="passed", stage="complete")
                else:
                    report.update(status="paused", stage="boundary_retained")
            finally:
                try:
                    report["drive_used_end_bytes"] = drive.measure()
                finally:
                    settle = quota.settle(ticket, time.monotonic() - args.started)
                    report.update(time_measured_seconds=settle["measured_seconds"], time_charged_seconds=settle["seconds"],
                                  time_remaining_seconds=TIME_CAP-history["charged_seconds"]-settle["seconds"],
                                  budget_tip_sha256=settle["content_sha256"])
                    print("Budget settlement SHA-256:", settle["content_sha256"], flush=True)
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        raise
    finally:
        if ledger is not None and ledger.path.is_file():
            report["usage"] = ledger.snapshot()["invocations"][-1]
        report["elapsed_before_report_seconds"] = time.monotonic() - args.started
        _atomic(args.work / "result.json", canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "work", "legacy", "bulk", "rgb", "pose"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--recover-writer", default="")
    parser.add_argument("--budget-tip", default="")
    args = parser.parse_args(argv)
    if not 0 <= time.monotonic() - args.started <= 240:
        parser.error("notebook start time outside the pre-reservation limit")
    if args.recover_writer and not re.fullmatch(r"empty|[0-9a-f]{32}", args.recover_writer):
        parser.error("invalid writer recovery token")
    if args.budget_tip and not re.fullmatch(r"[0-9a-f]{64}", args.budget_tip):
        parser.error("invalid retained budget SHA")
    run(args, json.loads(_regular_read(args.pins, 32*1024)))


if __name__ == "__main__":
    main()

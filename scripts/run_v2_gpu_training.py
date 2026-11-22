"""One manual v2 GPU invocation against the final immutable 128 cache."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic, _hash_file, _locked
from rewind_v1 import v2_cache as cache, v2_contract as contract
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import Journal, writer
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, read_manifest
from scripts.run_v2_gpu_cache_probe import load_catalog
from scripts.v2_gpu_training_comparison import export_comparison, shared_clips
from scripts.v2_gpu_training_pipeline import FullCache, FullData, Training
from scripts.v2_gpu_training_storage import (BOUNDARY, EVAL_SEED, GIB, MIB, SEEDS, Checkpoints, DriveLedger,
                                            final_entries, publish_json, verify_first)


RUN_FOLDER = "v2_gpu_training_128_v1"
SECONDS = 28_800
STAGES = ("frontend", "body", "hand")
RESOURCE_LIMITS = {"read_bytes": 96*GIB, "write_bytes": 24*GIB, "cache_bytes": 52*GIB,
                   "feature_bytes": 2*GIB, "run_bytes": 2*GIB, "archive_bytes": GIB,
                   "checkpoint_bytes": 256*MIB, "rss_bytes": 7*GIB}


def training_config(seed):
    return {**contract.config(), "external_timeout_seconds": SECONDS, "run_id": "V2-FULL-128-3X3-v1",
            "resources": dict(RESOURCE_LIMITS), "training": {"initialization": "fresh", "seed": seed,
            "epochs": dict.fromkeys(STAGES, 3), "upper_policy": "gt_train_predicted_validation",
            "evaluation_policy": "per_scenario_reference"}}


def identity_for(mode, pins):
    seed = SEEDS[mode]
    return {"run_id": "V2-FULL-128-3X3-v1", "seed": seed, "slot": mode, "source": pins["source_sha256"],
            "manifest": pins["manifest_content_sha256"], "cache_boundary": BOUNDARY, "eval_seed": EVAL_SEED,
            "config": digest(training_config(seed))}


def load_inputs(args, pins, *, config=training_config, require_full_local_cache=True):
    suite = contract.ROOT
    if digest(contract.source()) != pins["training_core_sha256"]:
        raise ValueError("v2_gpu_training_core_pin")
    for name, expected in pins["training_modules"].items():
        if _hash_file(suite / name) != expected:
            raise ValueError(f"v2_gpu_training_module_pin: {name}")
    extract_pinned_zip(suite / pins["embedded_manifest"], pins["archive_sha256"], args.invocation / "manifest",
                       max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
    doc, spec = read_manifest(args.invocation / "manifest" / MANIFEST_NAME, pins, suite / "reference")
    cfg = config(SEEDS[args.slot])
    plan = contract.plan(doc, spec, cfg, args.workspace)
    n = plan["train_updates_per_epoch_at_batch2"]*3
    splits = {p: [r for r in plan["clips"] if r["split"] == p] for p in ("train", "validation")}
    if (n != 197961 or len(splits["train"]) != 3097 or len(splits["validation"]) != 421
            or sum(r["train"] for r in plan["counts"].values()) != 119586
            or sum(r["validation"] for r in plan["counts"].values()) != 16425
            or plan["estimated_feature_bytes"] > cfg["resources"]["feature_bytes"]
            or require_full_local_cache and plan["estimated_cache_bytes"] > cfg["resources"]["cache_bytes"]):
        raise ValueError("v2_gpu_training_coverage_or_resources")
    return doc, spec, plan


def read_journal(project, doc, spec, pins, ledger):
    base = project / "reports" / "cache"
    bulk, legacy = base / "v2_bulk_cache_041837Z", base / "v2_cpu_cache_041837Z"
    for folder in (bulk, legacy):
        if folder.is_symlink() or not folder.is_dir() or not folder.resolve().is_relative_to(project):
            raise ValueError("v2_gpu_cache_path")
    anchor = load_catalog(_regular_read(legacy / "catalog.json", 64*MIB, pins["catalog_bytes"]), pins)
    producer = {**cache.binding(doc, spec, contract.config()), "source": pins["bulk_producer_source"],
                "torch": "2.14.1+cpu", "numpy": "2.4.6", "pillow": "11.3.0"}
    if digest(producer["source"]) != pins["bulk_producer_source_identity"]:
        raise ValueError("v2_gpu_bulk_producer_source_pin")
    journal = Journal(bulk, legacy, doc, anchor, producer)
    entries = final_entries(journal, ledger)
    if sum(e["package"]["bytes"] for e in entries) > RESOURCE_LIMITS["cache_bytes"]:
        raise ValueError("v2_gpu_cache_package_total")
    return journal, anchor, entries


def open_cache128(args, project, doc, spec, pins, cfg, ledger, drive, owner):
    journal, anchor, entries = read_journal(project, doc, spec, pins, ledger)
    return FullCache(args.workspace / "cache", journal, anchor, entries, spec, cfg, ledger), len(entries)


PROFILE_128 = SimpleNamespace(run_folder=RUN_FOLDER, config=training_config, identity=identity_for,
                              load_inputs=load_inputs, open_cache=open_cache128, data=FullData)


def run(args, pins, report, profile=PROFILE_128):
    if not 0 < args.started <= time.monotonic() or time.monotonic()-args.started >= 600:
        raise TimeoutError("v2_gpu_start_time_mount_or_bootstrap")
    if args.resume_tip and not re.fullmatch(r"[0-9a-f]{64}", args.resume_tip):
        raise ValueError("v2_gpu_resume_tip_format")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.version.cuda is None:
        raise RuntimeError("v2_gpu_one_cuda_device_required")
    if available_ram() < 4*GIB or shutil.disk_usage(args.workspace).free < 4*GIB:
        raise OSError("v2_gpu_minimum_ram_or_disk")
    device = torch.device("cuda:0")
    torch.ones((1,), device=device).add_(1).cpu()
    project = args.project.resolve(strict=True)
    parent = project / "reports" / profile.run_folder
    if not parent.parent.is_dir() or parent.parent.resolve().parent != project or parent.is_symlink():
        raise ValueError("v2_gpu_result_parent")
    parent.mkdir(exist_ok=True)
    drive = parent / f"{args.slot}-seed-{SEEDS[args.slot]}"
    if drive.is_symlink():
        raise ValueError("v2_gpu_seed_result_link")
    drive.mkdir(exist_ok=True)
    identity = profile.identity(args.slot, pins)
    cfg = profile.config(identity["seed"])
    report.update(identity=identity, drive=str(drive), environment={"device": torch.cuda.get_device_name(device),
                  "vram_bytes": torch.cuda.get_device_properties(device).total_memory, "torch": str(torch.__version__),
                  "cuda_runtime": torch.version.cuda, "precision": "fp32"})
    with _locked(args.workspace), writer(drive, args.recover_writer) as owner:
        ledger = DriveLedger(drive / "usage", identity, cfg, owner)
        try:
            with ledger.session(args.started):
                if args.slot != "A":
                    report["verified_A_first_update_sha256"] = verify_first(parent / "A-seed-42" / "checkpoints",
                                                                            args.invocation, profile.identity("A", pins), ledger, cfg)
                doc, spec, plan = profile.load_inputs(args, pins)
                images, count = profile.open_cache(args, project, doc, spec, pins, cfg, ledger, drive, owner)
                report.update(counts=plan["counts"], cache_shards=count, plan_sha256=plan["content_sha256"],
                              common_clips={k: list(v) for k, v in shared_clips(plan).items()},
                              input_setup_seconds=time.monotonic()-args.started)
                data = profile.data(doc, spec, cfg, plan, images, ledger, pins["source_sha256"])
                pipeline = Training(cfg, data, args.workspace / "features", device, args.invocation / "run", drive,
                                    ledger, owner, identity)
                saver = Checkpoints(drive / "checkpoints", args.invocation, identity, pipeline, cfg, ledger, owner)
                pipeline.checkpoints = saver
                try:
                    history = saver.history()
                    tip = history[-1] if history else None
                    if args.resume_tip and (tip is None or tip["content_sha256"] != args.resume_tip):
                        raise ValueError("v2_gpu_resume_requires_current_tip")
                    if tip and tip["completed"]:
                        report.update(status="completed", checkpoint=tip, already_completed=True)
                        return
                    if tip:
                        resume = saver.restore(tip)
                        pipeline.first_required &= not (saver.root / "first-update.json").is_file()
                        print(f"Restored verified Drive checkpoint {tip['cursor']} receipt={tip['content_sha256']}", flush=True)
                    else:
                        pipeline.root.mkdir(exist_ok=False)
                        resume = None
                    report["start_or_restore_seconds"] = time.monotonic()-args.started
                    result = pipeline.run(resume_sha256=resume)
                    report.update(status=result["status"], timing=result["timing_this_invocation"], cache_restore=images.metrics,
                                  checkpoint_cursor=result["checkpoint_cursor"], trained_updates=result["cursor"]["updates"],
                                  gpu_peak=result.get("gpu_peak_this_invocation"))
                    if result.get("failure"):
                        report["pause_reason"] = result["failure"]
                    if result["status"] == "completed":
                        report["checkpoint"] = saver.save()
                        report["comparison"] = export_comparison(pipeline, drive, args.invocation, ledger, owner)
                        report["checkpoint"] = saver.save(completed=True)
                    else:
                        report["checkpoint"] = saver.save()
                    if report["checkpoint"] and report["checkpoint"]["last"] == pipeline.last:
                        shutil.rmtree(pipeline.root)
                except BaseException:
                    if pipeline.last and (not pipeline.first_required or pipeline.durable_progress["updates"] == 0):
                        try:
                            report["checkpoint"] = saver.save()
                        except Exception as failure:
                            report["terminal_backup_error"] = {"type": type(failure).__name__, "reason": str(failure)[:512]}
                    raise
                finally:
                    images.close()
        except BaseException as exc:
            record_failure(report, exc)
            raise
        finally:
            try:
                if time.monotonic()-args.started < SECONDS-120:
                    report["usage"] = ledger.snapshot()["invocations"][-1]
                    diagnostics = drive / "diagnostics"
                    diagnostics.mkdir(exist_ok=True)
                    receipt = publish_json(diagnostics / f"{args.invocation.name}.json", "rewind_v2_gpu_invocation_v1", report, owner)
                    report["drive_diagnostic_sha256"] = receipt["content_sha256"]
            except Exception as exc:
                report["diagnostic_publication_error"] = str(exc)[:512]


def record_failure(report, exc):
    status = "stopped" if isinstance(exc, KeyboardInterrupt) else "paused" if isinstance(exc, TimeoutError) else "failed"
    report.update(status=status, failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
    return status


def main(argv=None, *, profile=PROFILE_128):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("invocation", "workspace", "project", "pins"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--slot", choices=tuple(SEEDS), required=True)
    parser.add_argument("--started", required=True, type=float)
    parser.add_argument("--resume-tip", default="")
    parser.add_argument("--recover-writer", default="")
    args = parser.parse_args(argv)
    report = {"status": "failed", "slot": args.slot}
    try:
        run(args, json.loads(_regular_read(args.pins, 128*MIB)), report, profile)
    except BaseException as exc:
        if record_failure(report, exc) != "paused":
            raise
    finally:
        _atomic(args.invocation / "result.json", canonical(report))
        print(canonical(report).decode(), flush=True)


if __name__ == "__main__":
    main()

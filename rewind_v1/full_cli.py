"""Fixed v1.2 audit-to-training connection; public training requires supported CUDA."""

import argparse
from contextlib import suppress
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import MIB, _local
from rewind_prepare.temporal_cache import _atomic
from .device import require_cuda_training
from .full_cache import GIB, PreparationBudget, cache_binding, prepare_frames
from .full_contract import MANIFEST_SHA, RUN_ID, bind_inputs, diagnostic_decision, load_audit, training_plan
from .full_data import FullData
from .full_recovery import RemoteResults, _recover_locked, recover, restore
from .full_training import FullPipeline, validate_full_config
from .runner import Budget
from .training import _source


def _context(args):
    torch.set_num_threads(1)
    if digest(_source()) != args.source_sha256:
        raise ValueError("v12_training_source_pin")
    config_raw = _regular_read(_local(args.config), MIB)
    if sha(config_raw) != args.config_sha256:
        raise ValueError("v12_training_config_pin")
    config = validate_full_config(json.loads(config_raw))
    manifest, coverage, snapshot = load_audit(args.audit_archive)
    spec, spec_raw = bind_inputs(args.inputs, manifest, snapshot)
    plan = training_plan(manifest, coverage)
    return config, manifest, spec_raw, plan


def _disjoint(paths):
    resolved = [Path(p).absolute() for p in paths]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(resolved) for b in resolved[i+1:]):
        raise ValueError("v12_training_path_overlap")


def _backup_location(work):
    root = Path(work) / "recovery"
    root.mkdir(exist_ok=True)
    _local(root, directory=True)
    return Path(tempfile.mkdtemp(prefix="local-", dir=root)) / "training.zip"


def _identity(args, binding):
    return {"run_id": RUN_ID, "manifest_sha256": MANIFEST_SHA, "config_sha256": args.config_sha256,
            "source_sha256": args.source_sha256, "preprocessing_binding_sha256": digest(binding)}


def launch_prepare(args):
    cfg, manifest, spec_raw, plan = _context(args)
    _disjoint([args.output, args.inputs, args.rgb_root, args.pose_root] if args.rgb_root != args.pose_root
              else [args.output, args.inputs, args.rgb_root])
    args.work.mkdir(exist_ok=True)
    args.output.mkdir(exist_ok=True)
    # The initial free-space requirement is conservative even on a resumed cache.
    if shutil.disk_usage(args.output).free < 16*GIB:
        raise ValueError("v12_preparation_free_space_16GiB")
    status_dir = args.work / "preparation_status"
    status_dir.mkdir(exist_ok=True)
    with RemoteResults(args.drive_root, _identity(args, cache_binding(manifest, spec_raw))) as remote:
        remote.ensure_terminal_room()
        try:
            result = prepare_frames(manifest, spec_raw, args.rgb_root, args.pose_root, args.output,
                                    budget=PreparationBudget(), max_chunks=args.max_chunks)
            _atomic(status_dir / "run.json", canonical(result)+b"\n")
        except BaseException as exc:
            preparation = args.output / "preparation.json"
            if preparation.is_file():
                _atomic(status_dir / "run.json", _regular_read(preparation, MIB))
            _atomic(status_dir / "launch_error.json", canonical({"status":"failed", "type":type(exc).__name__,
                                                                 "reason":str(exc)[:512]})+b"\n")
            raise
        finally:
            archive = recover(status_dir, _backup_location(args.work))
            saved = remote.publish(archive, periodic=False)
            Path(archive["output"]).unlink()  # Only after remote size/hash readback; source/cache untouched.
    return {"status": result["status"], "completed_frames": result["completed_frames"], "backup": saved}


def launch_train(args):
    cfg, manifest, spec_raw, plan = _context(args)
    _disjoint([args.output, args.features, args.cache, args.inputs])
    args.work.mkdir(exist_ok=True)
    args.output.mkdir(exist_ok=True)
    if shutil.disk_usage(args.work).free < 24*GIB:
        raise ValueError("v12_training_free_space_24GiB")
    data, pending, report, training_error = None, None, None, None
    with RemoteResults(args.drive_root, _identity(args, cache_binding(manifest, spec_raw))) as remote:
        remote.ensure_terminal_room()
        def periodic(run, last_sha, periodic):
            nonlocal pending
            pending = _recover_locked(run, _backup_location(args.work), last_sha)
            saved = remote.publish(pending, periodic=True)
            Path(pending["output"]).unlink()
            pending = None
            return saved
        try:
            device = torch.device(require_cuda_training(args.device)["device"])
            torch.cuda.reset_peak_memory_stats(device)
            budget = Budget(cfg["limits"], shutdown_reserve=120)
            data = FullData(manifest, plan, spec_raw, args.cache, cfg, args.config_sha256, budget)
            pipeline = FullPipeline(cfg, args.config_sha256, data, device, args.output, budget, args.features, periodic)
            report = pipeline.run(args.resume_sha256, args.stop_after_updates)
        except BaseException as exc:
            training_error = exc
            _atomic(args.output / "launch_error.json", canonical({"status":"failed", "type":type(exc).__name__,
                                                                  "reason":str(exc)[:512]})+b"\n")
        finally:
            if data is not None:
                data.cache.close()
            # A failed periodic archive predates the final paused/failed report. Keep it,
            # and package the current metadata rather than publishing its stale 'running' status.
            archive = recover(args.output, _backup_location(args.work))
            saved = remote.publish(archive, periodic=False)
            Path(archive["output"]).unlink()
    if training_error is not None:
        raise training_error
    return {"status": report["status"], "updates": report["cursor"]["updates"],
            "last_checkpoint": report["last_checkpoint"], "planned_updates": report["planned_updates"], "backup": saved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source")
    for name in ("plan", "prepare", "train", "cache-pack", "cache-restore"):
        action = commands.add_parser(name)
        for option in ("config", "inputs", "audit-archive"):
            action.add_argument("--"+option, type=Path, required=True)
        for option in ("source-sha256", "config-sha256"):
            action.add_argument("--"+option, required=True)
        if name in ("plan", "cache-pack", "cache-restore"):
            action.add_argument("--output", type=Path, required=True)
        else:
            for option in ("output", "work", "drive-root"):
                action.add_argument("--"+option, type=Path, required=True)
        if name == "prepare":
            action.add_argument("--rgb-root", type=Path, required=True)
            action.add_argument("--pose-root", type=Path, required=True)
            action.add_argument("--max-chunks", type=int)
        if name == "train":
            action.add_argument("--cache", type=Path, required=True)
            action.add_argument("--features", type=Path, required=True)
            action.add_argument("--device", default="cuda")
            action.add_argument("--resume-sha256")
            action.add_argument("--stop-after-updates", type=int)
        if name == "cache-pack":
            action.add_argument("--cache", type=Path, required=True)
            action.add_argument("--number", type=int, required=True)
        if name == "cache-restore":
            action.add_argument("--retained", type=Path, required=True)
            action.add_argument("--catalog", type=Path, required=True)
            action.add_argument("--catalog-sha256", required=True)
    rec = commands.add_parser("recover")
    rec.add_argument("--run", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    res = commands.add_parser("restore")
    res.add_argument("--archive", type=Path, required=True)
    res.add_argument("--archive-sha256", required=True)
    res.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "source":
        result = {"source_sha256": digest(_source())}
    elif args.command == "plan":
        _, manifest, _, plan = _context(args)
        decision = diagnostic_decision(args.config_sha256, plan["content_sha256"])
        args.output.mkdir()
        for name, doc in (("plan.json", plan), ("diagnostic_decision.json", decision)):
            _atomic(args.output / name, canonical(doc)+b"\n")
        result = {"plan_sha256":plan["content_sha256"], "decision_sha256":decision["content_sha256"]}
    elif args.command == "prepare":
        result = launch_prepare(args)
    elif args.command == "train":
        result = launch_train(args)
    elif args.command == "cache-pack":
        from .full_cache_retention import pack_chunk
        _, manifest, spec_raw, _ = _context(args)
        result = pack_chunk(args.cache, manifest, spec_raw, args.number, args.output)
    elif args.command == "cache-restore":
        from .full_cache_retention import restore_cache
        _, manifest, spec_raw, _ = _context(args)
        result = restore_cache(args.retained, args.catalog, args.catalog_sha256, manifest, spec_raw, args.output)
    elif args.command == "recover":
        result = recover(args.run, args.output)
    else:
        result = restore(args.archive, args.archive_sha256, args.output)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()

"""Run or recover the bounded v1 frontend → body → hand pipeline."""

import argparse
from contextlib import suppress
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _no_link, _regular_read
from rewind_m1.training import seed_run
from rewind_prepare.converter import sha
from rewind_prepare.extracted import _local
from rewind_prepare.temporal_cache import _atomic, _check_part, _hash_file, _locked
from .device import require_a100
from .diagnostic import DiagnosticData, POLICY, validate_pilot
from .run_data import ActualData, FixtureData
from .runner import Budget, FORMAT, LAST, MIB, REPORT, STAGES, Pipeline, _read_checkpoint, validate_config
from .training import MAX_CHECKPOINT_BYTES, _source


def recover(root, output, max_bytes):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("rewind_v1_recovery_limit")
    root, output = _local(root, directory=True), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(root):
        raise ValueError("rewind_v1_recovery_output")
    with _locked(root):
        receipts = {}
        for name in (REPORT, "launch_error.json"):
            path = root / name
            if path.exists():
                data = _regular_read(path, 4*MIB)
                receipts[name] = {"name": name, "bytes": len(data), "sha256": sha(data)}
        if (root / LAST).exists():
            path = root / LAST
            size = _no_link(path).st_size
            if not 0 < size <= min(MAX_CHECKPOINT_BYTES, max_bytes):
                raise ValueError("rewind_v1_recovery_checkpoint_size")
            pin = _hash_file(path, size)
            state = _read_checkpoint(path, pin, MAX_CHECKPOINT_BYTES)
            if state["binding"]["format"] != FORMAT:
                raise ValueError("rewind_v1_recovery_checkpoint_format")
            receipts[LAST] = {"name": LAST, "bytes": size, "sha256": pin}
            for role, item in state["progress"]["best"].items():
                if role not in STAGES or item["name"] != f"best-{role}-e{item['epoch']:03d}.pt":
                    raise ValueError("rewind_v1_recovery_best_name")
                receipts[item["name"]] = {k: item[k] for k in ("name", "bytes", "sha256")}
            del state
        if not receipts:
            raise ValueError("rewind_v1_recovery_empty")
        receipt = canonical({"schema": "rewind_v1_recovery_v1", "files": receipts}) + b"\n"
        upper = sum(item["bytes"] for item in receipts.values()) + len(receipt) + 4096
        if upper > max_bytes or shutil.disk_usage(output.parent).free < upper:
            raise TimeoutError("rewind_v1_recovery_size")
        for item in receipts.values():
            _check_part(root / item["name"], item)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".v1-recovery-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for name in receipts:
                    archive.write(root / name, name)
                archive.writestr("recovery_receipt.json", receipt)
            if temporary.stat().st_size > max_bytes or output.exists():
                raise ValueError("rewind_v1_recovery_publish")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "sha256": _hash_file(output), "bytes": output.stat().st_size,
            "last_sha256": receipts.get(LAST, {}).get("sha256")}


def launch(args):
    diagnostic = args.command == "diagnostic"
    real_data = args.command in ("actual", "diagnostic")
    device = torch.device(require_a100(args.device)["device"] if real_data else "cpu")
    if real_data and digest(_source()) != args.source_sha256:
        raise ValueError("rewind_v1_launch_source_pin")
    raw = _regular_read(_local(args.config), 256*1024)
    if sha(raw) != args.config_sha256:
        raise ValueError("rewind_v1_launch_config_pin")
    cfg = validate_config(read_json(raw))
    if diagnostic:
        validate_pilot(cfg)
        torch.cuda.reset_peak_memory_stats(device)
    budget = Budget(cfg["limits"])
    output = Path(args.output).absolute()
    if real_data:
        paths = [args.cache, args.cache_local, args.diagnostic_bundle if diagnostic else args.review_dir]
        if any(output.is_relative_to(Path(path).absolute()) or Path(path).absolute().is_relative_to(output)
               for path in paths if path is not None):
            raise ValueError("rewind_v1_launch_output_overlap")
        if diagnostic and (args.cache_local is None or args.cache_local.resolve() == args.cache.resolve()):
            raise ValueError("rewind_v1_diagnostic_separate_local_cache")
    output.mkdir(exist_ok=True)
    output = _local(output, directory=True)
    seed_run(cfg["seed"])
    try:
        data_type = {"actual": ActualData, "diagnostic": DiagnosticData}
        data = data_type[args.command](args, cfg, budget) if real_data else FixtureData()
        pipeline = Pipeline(cfg, args.config_sha256, data, device, output, budget)
    except BaseException as exc:
        if not (output / LAST).exists():
            with suppress(BaseException):
                info = {"status": "failed", "data_kind": "dataset_sample" if real_data else "synthetic_fixture",
                        "type": type(exc).__name__, "reason": str(exc)[:512]}
                if diagnostic:
                    info.update(POLICY, run_kind="nominal_25fps_diagnostic", decision_sha256=args.decision_sha256)
                _atomic(output / "launch_error.json", canonical(info) + b"\n")
        raise
    report = pipeline.run(args.resume_sha256, args.stop_after_updates)
    result = {"status": report["status"], "data_kind": report["input"]["data_kind"], "coverage": {
        k: v for k, v in report["coverage"].items() if k != "teacher_validation_excluded_keys"},
        "last_checkpoint": report["last_checkpoint"], "updates": report["cursor"]["updates"]}
    if diagnostic:
        result.update(run_kind="nominal_25fps_diagnostic", validation_reference_only=True,
                      time_basis=POLICY["time_basis"], decision_sha256=args.decision_sha256)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("source")
    for name in ("synthetic", "actual", "diagnostic"):
        run = commands.add_parser(name)
        for arg in ("config", "output"):
            run.add_argument("--" + arg, type=Path, required=True)
        run.add_argument("--config-sha256", required=True)
        run.add_argument("--resume-sha256")
        run.add_argument("--stop-after-updates", type=int)
        if name != "synthetic":
            metadata = ("evidence", "review-dir") if name == "actual" else ("diagnostic-bundle",)
            for arg in ("selection", "spec", "cache", *metadata):
                run.add_argument("--" + arg, type=Path, required=True)
            for arg in ("selection-sha256", "cache-sha256", "source-sha256",
                        "evidence-sha256" if name == "actual" else "decision-sha256"):
                run.add_argument("--" + arg, required=True)
            run.add_argument("--cache-local", type=Path, required=name == "diagnostic")
            run.add_argument("--device", default="cuda")
    rec = commands.add_parser("recover")
    rec.add_argument("--run", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "source":
        result = {"source_sha256": digest(_source())}
    elif args.command == "recover":
        result = recover(args.run, args.output, args.max_bytes)
    else:
        result = launch(args)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()

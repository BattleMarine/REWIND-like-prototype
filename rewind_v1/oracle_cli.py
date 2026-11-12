"""Prepare, train, recover, or restore the fixed v1.4 oracle memorization diagnostic."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import _local
from rewind_prepare.temporal_cache import _atomic, _check_part, _hash_file, _locked
from rewind_prepare.temporal_selection import check_pin, seal
from .device import require_cuda_training
from .oracle_data import MIB, load_prepared, prepare, source_identity
from .oracle_overfit import FORMAT, OracleExperiment, validate_config
from .runner import Budget, _read_checkpoint


ARCHIVE_LIMIT = 1024**3
CHECKPOINT_LIMIT = 512*MIB


def _artifact_names(state):
    if state["binding"]["format"] != FORMAT:
        raise ValueError("v14_checkpoint_format")
    names = {"last.pt", "oracle_input.json", "oracle_clip.pt", "spec.json"}
    for step, item in state["snapshots"].items():
        if step not in {str(v) for v in state["binding"]["config"]["milestones"]} or item["name"] != f"models-{int(step):05d}.pt":
            raise ValueError("v14_snapshot_name")
        names.add(item["name"])
    for step, result in state["evaluations"].items():
        if step not in state["snapshots"] or result["prediction_file"] != f"predictions-{int(step):05d}.json":
            raise ValueError("v14_evaluation_name")
        names.update((result["prediction_file"], f"comparison-{int(step):05d}.png"))
    return names


def recover(run, output):
    run, output = _local(run, directory=True), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(run):
        raise ValueError("v14_recovery_new_output")
    with _locked(run):
        names = {"run.json"} if (run / "run.json").is_file() else set()
        last_hash = None
        if (run / "last.pt").is_file():
            last_hash = _hash_file(run / "last.pt")
            state = _read_checkpoint(run / "last.pt", last_hash, CHECKPOINT_LIMIT)
            names.update(_artifact_names(state))
            for item in state["snapshots"].values():
                _check_part(run / item["name"], item)
            for evaluation in state["evaluations"].values():
                for item in evaluation["artifacts"]:
                    _check_part(run / item["name"], item)
            del state
        if not names:
            raise ValueError("v14_no_recovery_artifacts")
        files = {name: {"bytes": _local(run / name).stat().st_size, "sha256": _hash_file(run / name)} for name in sorted(names)}
        receipt = seal({"schema": "rewind_v14_recovery_v1", "files": files, "last_sha256": last_hash})
        raw = canonical(receipt)+b"\n"
        upper = sum(item["bytes"] for item in files.values())+len(raw)+16384
        if upper > ARCHIVE_LIMIT or shutil.disk_usage(output.parent).free < upper:
            raise ValueError("v14_recovery_budget")
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".v14-recovery-", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as bundle:
                for name in files:
                    bundle.write(run / name, name)
                bundle.writestr("recovery_receipt.json", raw)
            if temporary.stat().st_size > ARCHIVE_LIMIT or output.exists():
                raise ValueError("v14_recovery_publish")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": _hash_file(output), "last_sha256": last_hash}


def restore(archive, archive_sha, output):
    from scripts.archvis_pilot_bootstrap import extract_pinned_zip

    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or shutil.disk_usage(output.parent).free < 2*ARCHIVE_LIMIT:
        raise ValueError("v14_restore_new_output_and_space")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v14-restore-") as directory:
        container = Path(directory) / "restored"
        container.mkdir()
        staged = container / "training"
        extract_pinned_zip(_local(archive), archive_sha, staged, max_zip_bytes=ARCHIVE_LIMIT,
                           max_output_bytes=ARCHIVE_LIMIT, max_members=32)
        receipt = json.loads(_regular_read(staged / "recovery_receipt.json", MIB))
        check_pin(receipt, "rewind_v14_recovery_v1", receipt.get("content_sha256"))
        if {p.name for p in staged.iterdir()} != set(receipt["files"]) | {"recovery_receipt.json"}:
            raise ValueError("v14_restore_members")
        for name, item in receipt["files"].items():
            _check_part(staged / name, item)
        state = _read_checkpoint(staged / "last.pt", receipt["last_sha256"], CHECKPOINT_LIMIT)
        if set(receipt["files"]) - {"run.json"} != _artifact_names(state):
            raise ValueError("v14_restore_required_artifacts")
        validate_config(state["binding"]["config"])
        if state["binding"]["source_code_sha256"] != source_identity():
            raise ValueError("v14_restore_source_changed")
        input_hash = state["binding"]["input_sha256"]
        del state
        load_prepared(staged, input_hash)
        prepared = container / "prepared"
        prepared.mkdir()
        for name in ("oracle_input.json", "oracle_clip.pt", "spec.json"):
            shutil.copyfile(staged / name, prepared / name)
        container.rename(output)
    return {"prepared": str(output / "prepared"), "run": str(output / "training"), "input_sha256": input_hash,
            "last_sha256": receipt["last_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("source")
    prep = actions.add_parser("prepare")
    for name in ("inputs", "diagnostic-bundle", "cache", "output"):
        prep.add_argument("--"+name, type=Path, required=True)
    prep.add_argument("--decision-sha256", required=True)
    prep.add_argument("--source-sha256", required=True)
    train = actions.add_parser("train")
    for name in ("config", "prepared", "output"):
        train.add_argument("--"+name, type=Path, required=True)
    for name in ("config-sha256", "input-sha256", "source-sha256"):
        train.add_argument("--"+name, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--resume-sha256")
    train.add_argument("--stop-after-updates", type=int)
    rec = actions.add_parser("recover")
    rec.add_argument("--run", required=True, type=Path)
    rec.add_argument("--output", required=True, type=Path)
    res = actions.add_parser("restore")
    res.add_argument("--archive", required=True, type=Path)
    res.add_argument("--archive-sha256", required=True)
    res.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command in ("prepare", "train") and digest(source_identity()) != args.source_sha256:
        raise ValueError("v14_source_pin")
    if args.command == "source":
        result = {"source_sha256": digest(source_identity())}
    elif args.command == "prepare":
        result = prepare(args)
    elif args.command == "train":
        device = torch.device(require_cuda_training(args.device, report=lambda info: None)["device"])
        raw = _regular_read(_local(args.config), MIB)
        if sha(raw) != args.config_sha256:
            raise ValueError("v14_config_pin")
        cfg = json.loads(raw)
        validate_config(cfg)
        if args.stop_after_updates is not None and args.stop_after_updates < 1:
            raise ValueError("v14_stop_update_budget")
        clip, inputs = load_prepared(args.prepared, args.input_sha256)
        if inputs.get("data_kind") != "dataset_sample":
            raise ValueError("v14_cli_requires_audited_dataset_clip")
        output = Path(args.output).absolute()
        prepared = _local(args.prepared, directory=True)
        if output.is_relative_to(prepared) or prepared.is_relative_to(output):
            raise ValueError("v14_output_prepared_overlap")
        output.mkdir(exist_ok=True)
        torch.cuda.reset_peak_memory_stats(device)
        budget = Budget(cfg["limits"], shutdown_reserve=120)
        experiment = OracleExperiment(cfg, clip, inputs, device, output, budget, prepared / "oracle_clip.pt")
        result = experiment.run(args.resume_sha256, args.stop_after_updates)
    elif args.command == "recover":
        result = recover(args.run, args.output)
    else:
        result = restore(args.archive, args.archive_sha256, args.output)
    print(canonical(result).decode())


if __name__ == "__main__":
    main()

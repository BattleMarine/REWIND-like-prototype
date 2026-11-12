"""Bounded v1.4 result backup using the existing copy/readback primitives."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _hash, _regular_read
from rewind_prepare.extracted import _local
from rewind_prepare.temporal_cache import _atomic
from rewind_v1.oracle_cli import ARCHIVE_LIMIT
from scripts.v11_recovery import _measure, _write


SCHEMA = "archvis_v14_result_backup_v1"
MARKER = b"archvis_v14_result_backup_attempt_v1\n"
MIB = 1024**2
SECONDS = 600


def prepare(drive_root):
    root = _local(drive_root, directory=True)
    for component in ("REWIND-like prototype", "reports", "v1_branch_experiments", "v14_oracle"):
        root = root / component
        root.mkdir(exist_ok=True)
        _local(root, directory=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    with (attempt / "preflight.txt").open("xb") as stream:
        _write(stream, MARKER)
    if _regular_read(attempt / "preflight.txt", MIB) != MARKER:
        raise ValueError("v14_drive_preflight")
    return {"attempt": str(attempt)}


def package(work, run):
    work = _local(work, directory=True)
    root = work / "recovery"
    root.mkdir(exist_ok=True)
    local = Path(tempfile.mkdtemp(prefix="local-", dir=_local(root, directory=True)))
    if run.is_dir() and any((run / name).is_file() for name in ("run.json", "last.pt")):
        archive = local / "training.zip"
        child = subprocess.run([sys.executable, "-m", "rewind_v1.oracle_cli", "recover", "--run", str(run),
                                "--output", str(archive)], cwd=Path(__file__).resolve().parents[1],
                               timeout=SECONDS, check=True, capture_output=True, text=True)
        result = json.loads(child.stdout)
        kind = "training" if result["last_sha256"] else "status"
    else:
        archive = local / "status.zip"
        raw = _regular_read(work / "notebook_training_result.json", MIB)
        with zipfile.ZipFile(archive, "x") as bundle:
            bundle.writestr("notebook_training_result.json", raw)
        with archive.open("rb") as stream:
            result = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                      "bytes": archive.stat().st_size, "last_sha256": None}
        kind = "status"
    pin = {"schema": SCHEMA, "archive": archive.name, "kind": kind,
           **{key: result[key] for key in ("bytes", "sha256", "last_sha256")}}
    _atomic(local / "local_receipt.json", canonical(pin)+b"\n")
    return {**pin, "local_zip": str(archive)}


def publish(work, local_zip, attempt):
    work, local_zip, attempt = _local(work, directory=True), _local(local_zip), _local(attempt, directory=True)
    if (local_zip.name not in ("training.zip", "status.zip") or local_zip.parent.parent != work / "recovery"
            or not local_zip.parent.name.startswith("local-") or not attempt.name.startswith("attempt-")
            or _regular_read(attempt / "preflight.txt", MIB) != MARKER):
        raise ValueError("v14_backup_paths")
    pin = json.loads(_regular_read(local_zip.parent / "local_receipt.json", MIB))
    if (pin["schema"] != SCHEMA or pin["archive"] != local_zip.name or not _hash(pin["sha256"])
            or not 0 < pin["bytes"] <= ARCHIVE_LIMIT or local_zip.stat().st_size != pin["bytes"]):
        raise ValueError("v14_backup_pin")
    status = {"state": "copying", "local_zip": str(local_zip), "drive_attempt": str(attempt)}
    progress = local_zip.parent / (attempt.name+".json")
    with progress.open("xb") as stream:
        _write(stream, canonical(status)+b"\n")
    partial, final = attempt / (local_zip.name+".partial"), attempt / local_zip.name
    deadline = time.monotonic()+SECONDS
    try:
        with (attempt / "publish.lock").open("xb"):
            if final.exists() or final.is_symlink() or (attempt / "receipt.json").exists() or (attempt / "receipt.json").is_symlink():
                raise FileExistsError("v14_backup_attempt_used")
            with local_zip.open("rb") as source, partial.open("xb") as target:
                if _measure(source, pin["bytes"], deadline, target) != (pin["bytes"], pin["sha256"]):
                    raise ValueError("v14_local_archive_hash")
                target.flush()
                os.fsync(target.fileno())
            with partial.open("rb") as target:
                if _measure(target, pin["bytes"], deadline) != (pin["bytes"], pin["sha256"]):
                    raise ValueError("v14_drive_archive_hash")
            if time.monotonic() >= deadline:
                raise TimeoutError("v14_copy_time_limit")
            if final.exists() or final.is_symlink() or (attempt / "receipt.json").exists() or (attempt / "receipt.json").is_symlink():
                raise FileExistsError("v14_backup_attempt_used")
            partial.rename(final)
            _atomic(attempt / "receipt.json", canonical({**pin, "state": "verified"})+b"\n")
            status.update(state="verified", drive_zip=str(final), **pin)
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error)[:512])
        _atomic(progress, canonical(status)+b"\n")
        raise
    _atomic(progress, canonical(status)+b"\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name, paths in (("prepare", ("drive-root",)), ("package", ("work", "run")),
                        ("publish", ("work", "local-zip", "attempt"))):
        command = actions.add_parser(name)
        for path in paths:
            command.add_argument("--"+path, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.drive_root)
    elif args.command == "package":
        result = package(args.work, args.run)
    else:
        result = publish(args.work, args.local_zip, args.attempt)
    print(canonical(result).decode())


if __name__ == "__main__":
    main()

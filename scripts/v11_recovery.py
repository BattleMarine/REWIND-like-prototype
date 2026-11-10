"""Package a v1.1 rerun locally and publish a verified copy to a fresh Drive attempt."""

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


RESULT_DIR = Path("REWIND-like prototype/reports/v1_branch_experiments/v11_rerun")
ARCHIVE_LIMIT = 1024**3
PACKAGE_SECONDS = 600
COPY_SECONDS = 600
CHUNK_BYTES = 1024**2
SCHEMA = "archvis_v11_rerun_recovery_v1"
ATTEMPT_MARKER = b"archvis_v11_rerun_attempt_v1\n"


def _write(stream, data):
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def prepare_attempt(drive_root):
    root = _local(drive_root, directory=True)
    for component in RESULT_DIR.parts:
        root = root / component
        root.mkdir(exist_ok=True)
        _local(root, directory=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    marker = attempt / "preflight.txt"
    with marker.open("xb") as stream:
        _write(stream, ATTEMPT_MARKER)
    if _regular_read(marker, CHUNK_BYTES) != ATTEMPT_MARKER:
        raise ValueError("v11_recovery_drive_preflight")
    return attempt


def package(work, run):
    work = _local(work, directory=True)
    run = Path(run).absolute()
    recovery = work / "recovery"
    recovery.mkdir(exist_ok=True)
    recovery = _local(recovery, directory=True)
    local = Path(tempfile.mkdtemp(prefix="local-", dir=recovery))
    if any((run / name).is_file() for name in ("last.pt", "run.json", "launch_error.json")):
        archive = local / "training.zip"
        child = subprocess.run(
            [sys.executable, "-m", "rewind_v1", "recover", "--run", str(run),
             "--output", str(archive), "--max-bytes", str(ARCHIVE_LIMIT)],
            cwd=Path(__file__).resolve().parents[1], timeout=PACKAGE_SECONDS, check=True,
            capture_output=True, text=True,
        )
        result = json.loads(child.stdout)
        pin = {key: result[key] for key in ("bytes", "sha256", "last_sha256")}
        kind = "training" if pin["last_sha256"] else "status"
    else:
        status = _regular_read(work / "notebook_training_result.json", CHUNK_BYTES)
        archive = local / "status.zip"
        with zipfile.ZipFile(archive, "x") as bundle:
            bundle.writestr("notebook_training_result.json", status)
        with archive.open("rb") as stream:
            pin = {"bytes": archive.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                   "last_sha256": None}
        kind = "status"
    receipt = {"schema": SCHEMA, "archive": archive.name, "kind": kind, **pin}
    _atomic(local / "local_receipt.json", canonical(receipt) + b"\n")
    return {**receipt, "local_zip": str(archive)}


def _measure(stream, size, deadline, destination=None):
    measured = hashlib.sha256()
    count = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("v11_recovery_copy_time_limit")
        chunk = stream.read(min(CHUNK_BYTES, size - count + 1))
        if not chunk:
            return count, measured.hexdigest()
        count += len(chunk)
        if count > size:
            raise ValueError("v11_recovery_archive_growth")
        measured.update(chunk)
        if destination is not None:
            destination.write(chunk)


def publish(work, local_zip, attempt):
    work = _local(work, directory=True)
    local_zip = _local(local_zip)
    attempt = _local(attempt, directory=True)
    if (local_zip.name not in {"training.zip", "status.zip"} or not local_zip.parent.name.startswith("local-")
            or local_zip.parent.parent != work / "recovery"):
        raise ValueError("v11_recovery_local_archive_path")
    if (not attempt.name.startswith("attempt-")
            or _regular_read(attempt / "preflight.txt", CHUNK_BYTES) != ATTEMPT_MARKER):
        raise ValueError("v11_recovery_attempt_path")
    pin = json.loads(_regular_read(local_zip.parent / "local_receipt.json", CHUNK_BYTES))
    size, expected_hash = pin["bytes"], pin["sha256"]
    if (pin["schema"] != SCHEMA or pin["archive"] != local_zip.name or not _hash(expected_hash)
            or type(size) is not int or not 0 < size <= ARCHIVE_LIMIT
            or not local_zip.is_file() or local_zip.stat().st_size != size):
        raise ValueError("v11_recovery_local_archive_pin")
    progress = local_zip.parent / (attempt.name + ".json")
    status = {"schema": SCHEMA, "state": "copying", "local_zip": str(local_zip), "drive_attempt": str(attempt)}
    with progress.open("xb") as stream:
        _write(stream, canonical(status) + b"\n")
    deadline = time.monotonic() + COPY_SECONDS
    partial, final = attempt / (local_zip.name + ".partial"), attempt / local_zip.name
    receipt_path = attempt / "receipt.json"
    try:
        with (attempt / "publish.lock").open("xb"):
            if final.exists() or final.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
                raise FileExistsError("v11_recovery_attempt_already_used")
            with local_zip.open("rb") as source, partial.open("xb") as target:
                if _measure(source, size, deadline, target) != (size, expected_hash):
                    raise ValueError("v11_recovery_local_archive_hash")
                target.flush()
                os.fsync(target.fileno())
            with partial.open("rb") as target:
                if _measure(target, size, deadline) != (size, expected_hash):
                    raise ValueError("v11_recovery_drive_archive_hash")
            if time.monotonic() >= deadline:
                raise TimeoutError("v11_recovery_copy_time_limit")
            if final.exists() or final.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
                raise FileExistsError("v11_recovery_attempt_already_used")
            partial.rename(final)
            receipt = {**pin, "state": "verified", "attempt": attempt.name}
            _atomic(receipt_path, canonical(receipt) + b"\n")
            status.update(state="verified", drive_zip=str(final), receipt=str(receipt_path), **pin)
    except Exception as exc:
        status.update(state="failed", error_type=type(exc).__name__, error=str(exc)[:512])
        _atomic(progress, canonical(status) + b"\n")
        raise
    _atomic(progress, canonical(status) + b"\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, paths in (("prepare", ("drive-root",)), ("package", ("work", "run")),
                          ("publish", ("work", "local-zip", "attempt"))):
        command = actions.add_parser(action)
        for name in paths:
            command.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = {"attempt": str(prepare_attempt(args.drive_root))}
    elif args.action == "package":
        result = package(args.work, args.run)
    else:
        result = publish(args.work, args.local_zip, args.attempt)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()

"""Recover v1.2 phase reports and publish a verified copy to a fresh Drive attempt."""

import argparse
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
from rewind_prepare.full_archvis import REPORT_LIMIT
from rewind_prepare.temporal_cache import _atomic
from scripts.v11_recovery import _measure, _write


RESULT_DIR = Path("REWIND-like prototype/reports/v1_branch_experiments/v12_recovery")
ARCHIVE_LIMIT = REPORT_LIMIT // 2
SECONDS = 600
MIB = 1024**2
SCHEMA = "archvis_v12_phase_backup_v1"
MARKER = b"archvis_v12_phase_backup_attempt_v1\n"
PHASES = ("preflight", "audit")


def context_document(path, work=None):
    context = json.loads(_regular_read(_local(path), MIB))
    pins = ("source_zip_sha256", "input_zip_sha256", "snapshot_sha256", "coverage_sha256")
    revision = context.get("source_revision")
    if (set(context) != {*pins, "source_revision", "phase", "original_work"}
            or context.get("phase") not in PHASES or any(not _hash(context.get(key)) for key in pins)
            or not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("v12_backup_context")
    original_work = _local(context["original_work"], directory=True)
    if str(original_work) != context["original_work"] or (work is not None and original_work != work):
        raise ValueError("v12_backup_original_work")
    return context


def prepare_attempt(drive_root, phase):
    if phase not in PHASES:
        raise ValueError("v12_backup_phase")
    root = _local(drive_root, directory=True)
    for component in (*RESULT_DIR.parts, phase):
        root = root / component
        root.mkdir(exist_ok=True)
        _local(root, directory=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    with (attempt / "preflight.txt").open("xb") as stream:
        _write(stream, MARKER)
    if _regular_read(attempt / "preflight.txt", MIB) != MARKER:
        raise ValueError("v12_backup_drive_preflight")
    return attempt


def package(base, work, context_path):
    base, work = _local(base, directory=True), _local(work, directory=True)
    if base.is_relative_to(work):
        raise ValueError("v12_backup_output_inside_work")
    context = context_document(context_path, work)
    used = sum(path.stat().st_size for path in base.glob("local-*/*") if path.is_file())
    if used + ARCHIVE_LIMIT + MIB > REPORT_LIMIT:
        raise ValueError("v12_backup_local_budget")
    local = Path(tempfile.mkdtemp(prefix="local-", dir=base))
    archive = local / "reports.zip"
    child = subprocess.run(
        [sys.executable, "-m", "rewind_prepare.full_archvis", "recover", "--work", str(work),
         "--output", str(archive)], cwd=Path(__file__).resolve().parents[1], timeout=SECONDS,
        check=True, capture_output=True, text=True,
    )
    pin = child.stdout.strip()
    size = archive.stat().st_size
    if not _hash(pin) or not 0 < size <= ARCHIVE_LIMIT:
        raise ValueError("v12_backup_archive_pin")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.getinfo("recovery_receipt.json").file_size > MIB:
            raise ValueError("v12_backup_recovery_receipt_size")
        audit_resume = json.loads(bundle.read("recovery_receipt.json")).get("audit_resume")
    receipt = {"schema": SCHEMA, "archive": archive.name, "kind": "phase_reports",
               "bytes": size, "sha256": pin, "context": context, "audit_resume": audit_resume}
    _atomic(local / "local_receipt.json", canonical(receipt) + b"\n")
    return {**receipt, "local_zip": str(archive)}


def publish(base, local_zip, attempt, context_path):
    base, local_zip = _local(base, directory=True), _local(local_zip)
    attempt = _local(attempt, directory=True)
    if local_zip.name != "reports.zip" or not local_zip.parent.name.startswith("local-") or local_zip.parent.parent != base:
        raise ValueError("v12_backup_local_archive_path")
    context = context_document(context_path)
    if (not attempt.name.startswith("attempt-") or attempt.parent.name != context["phase"]
            or _regular_read(attempt / "preflight.txt", MIB) != MARKER):
        raise ValueError("v12_backup_attempt")
    pin = json.loads(_regular_read(local_zip.parent / "local_receipt.json", MIB))
    size, expected = pin["bytes"], pin["sha256"]
    if (pin["schema"] != SCHEMA or pin["archive"] != local_zip.name or pin["context"] != context
            or pin["kind"] != "phase_reports" or not _hash(expected)
            or type(size) is not int or not 0 < size <= ARCHIVE_LIMIT or local_zip.stat().st_size != size):
        raise ValueError("v12_backup_local_archive_pin")
    status = {"schema": SCHEMA, "state": "copying", "context": context,
              "local_zip": str(local_zip), "drive_attempt": str(attempt)}
    progress = local_zip.parent / (attempt.name + ".json")
    with progress.open("xb") as stream:
        _write(stream, canonical(status) + b"\n")
    deadline = time.monotonic() + SECONDS
    partial, final = attempt / "reports.zip.partial", attempt / "reports.zip"
    receipt_path = attempt / "receipt.json"
    try:
        with (attempt / "publish.lock").open("xb"):
            if final.exists() or final.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
                raise FileExistsError("v12_backup_attempt_already_used")
            with local_zip.open("rb") as source, partial.open("xb") as target:
                if _measure(source, size, deadline, target) != (size, expected):
                    raise ValueError("v12_backup_local_archive_hash")
                target.flush()
                os.fsync(target.fileno())
            with partial.open("rb") as target:
                if _measure(target, size, deadline) != (size, expected):
                    raise ValueError("v12_backup_drive_archive_hash")
            if time.monotonic() >= deadline:
                raise TimeoutError("v12_backup_copy_time_limit")
            if final.exists() or final.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
                raise FileExistsError("v12_backup_attempt_already_used")
            partial.rename(final)
            _atomic(receipt_path, canonical({**pin, "state": "verified", "attempt": attempt.name}) + b"\n")
            status.update(state="verified", drive_zip=str(final), receipt=str(receipt_path), **pin)
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error)[:512])
        _atomic(progress, canonical(status) + b"\n")
        raise
    _atomic(progress, canonical(status) + b"\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--drive-root", type=Path, required=True)
    prepare.add_argument("--phase", choices=PHASES, required=True)
    for action, paths in (("package", ("base", "work", "context")),
                          ("publish", ("base", "local-zip", "attempt", "context"))):
        command = actions.add_parser(action)
        for name in paths:
            command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = {"attempt": str(prepare_attempt(args.drive_root, args.phase))}
    elif args.action == "package":
        result = package(args.base, args.work, args.context)
    else:
        result = publish(args.base, args.local_zip, args.attempt, args.context)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()

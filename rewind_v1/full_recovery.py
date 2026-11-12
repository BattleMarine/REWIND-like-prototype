"""V1.2 immutable recovery archives and finite, separately budgeted result publication."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.extracted import MIB, _local
from rewind_prepare.temporal_cache import _atomic, _check_part, _hash_file, _locked
from rewind_prepare.temporal_selection import check_pin, seal
from .full_training import FORMAT, UPDATE_MILESTONES, validate_full_config
from .runner import LAST, STAGES, _read_checkpoint
from .training import _source


GIB = 1024**3
ARCHIVE_LIMIT = 2*GIB
CHECKPOINT_LIMIT = 256*MIB
REMOTE_LIMIT = 12*GIB
REMOTE_ATTEMPTS = 24


class RemoteBudgetExhausted(TimeoutError):
    pass


def model_pins(state):
    if state["binding"]["format"] != FORMAT:
        raise ValueError("v12_recovery_format")
    progress, result = state["progress"], {}
    def add(pin, expected):
        if pin["name"] not in expected:
            raise ValueError("v12_recovery_model_name")
        entry = {key: pin[key] for key in ("name", "bytes", "sha256")}
        if result.setdefault(entry["name"], entry) != entry:
            raise ValueError("v12_recovery_model_pin_conflict")
    for role, pin in progress["best"].items():
        if role not in STAGES:
            raise ValueError("v12_recovery_stage")
        add(pin, {f"best-{role}-e{pin['epoch']:03d}.pt"})
    for role, milestones in progress["milestones"].items():
        for epoch, pin in milestones.items():
            if role not in STAGES or epoch not in {"1", "5", "10"} or pin["epoch"] != int(epoch):
                raise ValueError("v12_recovery_epoch_milestone")
            add(pin, {f"{kind}-{role}-e{int(epoch):03d}.pt" for kind in ("best", "milestone")})
    for role, milestones in progress["update_milestones"].items():
        for step, pin in milestones.items():
            if role not in STAGES or int(step) not in UPDATE_MILESTONES[role]:
                raise ValueError("v12_recovery_update_milestone")
            add(pin, {f"update-{role}-u{int(step):06d}.pt"})
    return result


def _recover_locked(run, output, expected_last=None):
    """Caller owns the run lock, including periodic calls inside Pipeline.run."""
    run, output = _local(run, directory=True), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(run):
        raise ValueError("v12_recovery_new_output")
    deadline, files = time.monotonic()+600, {}
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("v12_recovery_time_limit")
    for name in ("run.json", "launch_error.json"):
        if (run / name).exists():
            raw = _regular_read(run / name, 4*MIB)
            files[name] = {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    last_pin = None
    if (run / LAST).exists():
        last_pin = _hash_file(run / LAST, check_time=check)
        if expected_last is not None and last_pin != expected_last:
            raise ValueError("v12_recovery_last_changed")
        state = _read_checkpoint(run / LAST, last_pin, CHECKPOINT_LIMIT)
        files.update(model_pins(state))
        files[LAST] = {"name": LAST, "bytes": (run / LAST).stat().st_size, "sha256": last_pin}
        del state
    if not files:
        raise ValueError("v12_recovery_empty")
    receipt = seal({"schema": "archvis_v12_training_recovery_v1", "files": files, "last_sha256": last_pin})
    raw = canonical(receipt)+b"\n"
    upper = sum(p["bytes"] for p in files.values())+len(raw)+16384
    if upper > ARCHIVE_LIMIT or shutil.disk_usage(output.parent).free < upper:
        raise ValueError("v12_recovery_size")
    for name, pin in files.items():
        _check_part(run / name, pin, check)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".v12-recovery-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name in files:
                with (run / name).open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    while block := source.read(MIB):
                        check()
                        target.write(block)
            archive.writestr("recovery_receipt.json", raw)
        if temporary.stat().st_size > ARCHIVE_LIMIT or output.exists() or output.is_symlink():
            raise ValueError("v12_recovery_publish")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": _hash_file(output), "last_sha256": last_pin}


def recover(run, output):
    with _locked(_local(run, directory=True)):
        return _recover_locked(run, output)


def restore(archive, archive_sha, output):
    from scripts.archvis_pilot_bootstrap import extract_pinned_zip
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or shutil.disk_usage(output.parent).free < 2*ARCHIVE_LIMIT:
        raise ValueError("v12_restore_new_output_and_space")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v12-restore-") as temporary:
        staged = Path(temporary) / "training"
        extract_pinned_zip(_local(archive), archive_sha, staged, max_zip_bytes=ARCHIVE_LIMIT,
                           max_output_bytes=ARCHIVE_LIMIT, max_members=64)
        receipt = json.loads(_regular_read(staged / "recovery_receipt.json", MIB))
        check_pin(receipt, "archvis_v12_training_recovery_v1", receipt.get("content_sha256"))
        files = receipt["files"]
        if {p.name for p in staged.iterdir()} != set(files) | {"recovery_receipt.json"}:
            raise ValueError("v12_restore_members")
        for name, pin in files.items():
            if name != pin["name"]:
                raise ValueError("v12_restore_name")
            _check_part(staged / name, pin)
        last_sha = receipt["last_sha256"]
        state = _read_checkpoint(staged / LAST, last_sha, CHECKPOINT_LIMIT)
        if set(files)-{LAST, "run.json", "launch_error.json"} != set(model_pins(state)):
            raise ValueError("v12_restore_required_models")
        validate_full_config(state["binding"]["config"])
        if state["binding"]["source_code_sha256"] != _source():
            raise ValueError("v12_restore_source")
        input_identity = state["binding"]["input"]
        del state
        (staged / "recovery_receipt.json").unlink()
        if output.exists() or output.is_symlink():
            raise ValueError("v12_restore_new_output")
        staged.rename(output)
    return {"run": str(output), "last_sha256": last_sha, "input": input_identity}


class RemoteResults:
    """Finite reservations survive failed copies; a stale lock requires explicit recovery."""

    def __init__(self, drive_root, identity):
        root = _local(drive_root, directory=True)
        for name in ("REWIND-like prototype", "reports", "v1_branch_experiments", "v12_training", "run-"+digest(identity)):
            root = root / name
            root.mkdir(exist_ok=True)
            _local(root, directory=True)
        self.root, self.identity = root, identity
        self._lease = None

    def __enter__(self):
        self._lease = _locked(self.root)
        self._lease.__enter__()
        self.lease_token = None
        try:
            token = uuid.uuid4().hex
            with (self.root / "lease.json").open("xb") as stream:
                stream.write(canonical({"token": token, "identity": self.identity})+b"\n")
                stream.flush()
                __import__("os").fsync(stream.fileno())
            self.lease_token = token
            self._check_lease()
            path = self.root / "ledger.json"
            if path.exists():
                self.ledger = json.loads(_regular_read(path, MIB))
                check_pin(self.ledger, "archvis_v12_remote_result_budget_v1", self.ledger.get("content_sha256"))
                if self.ledger["identity"] != self.identity:
                    raise ValueError("v12_remote_ledger_identity")
            else:
                if any(p.name.startswith("attempt-") for p in self.root.iterdir()):
                    raise ValueError("v12_remote_ledger_missing")
                self.ledger = {"schema": "archvis_v12_remote_result_budget_v1", "identity": self.identity, "reservations": []}
                self._write()
            entries = self.ledger["reservations"]
            names = [r["attempt"] for r in entries]
            if (len(entries) > REMOTE_ATTEMPTS or len(names) != len(set(names))
                    or any(not name.startswith("attempt-") or "/" in name or "\\" in name for name in names)
                    or any(type(r["bytes"]) is not int or not 0 < r["bytes"] <= ARCHIVE_LIMIT for r in entries)
                    or sum(r["bytes"] for r in entries) > REMOTE_LIMIT
                    or {p.name for p in self.root.iterdir() if p.name.startswith("attempt-")} - set(names)):
                raise ValueError("v12_remote_ledger_scope")
            # A readback preflight occurs before any preparation/training call.
            check_pin(json.loads(_regular_read(path, MIB)), "archvis_v12_remote_result_budget_v1", self.ledger["content_sha256"])
            return self
        except BaseException:
            self._release_lease()
            self._lease.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            self._release_lease()
        finally:
            self._lease.__exit__(*args)
        return False

    def _check_lease(self):
        lease = json.loads(_regular_read(self.root / "lease.json", MIB))
        if lease != {"token": self.lease_token, "identity": self.identity}:
            raise ValueError("v12_remote_lease_changed")

    def _release_lease(self):
        if self.lease_token is not None:
            self._check_lease()
            (self.root / "lease.json").unlink()
            self.lease_token = None

    def _write(self):
        self.ledger = seal({k:v for k,v in self.ledger.items() if k != "content_sha256"})
        _atomic(self.root / "ledger.json", canonical(self.ledger)+b"\n")
        observed = json.loads(_regular_read(self.root / "ledger.json", MIB))
        if observed != self.ledger:
            raise ValueError("v12_remote_ledger_readback")

    def reserve(self, archive, *, periodic):
        self._check_lease()
        size = archive["bytes"]
        entries = self.ledger["reservations"]
        used = sum(r["bytes"] for r in entries)
        # A periodic backup must leave one full terminal archive and one attempt available.
        allowance = ARCHIVE_LIMIT if periodic else 0
        remaining_attempts = 1 if periodic else 0
        if (not 0 < size <= ARCHIVE_LIMIT or used+size+allowance > REMOTE_LIMIT
                or len(entries)+1+remaining_attempts > REMOTE_ATTEMPTS):
            raise RemoteBudgetExhausted("v12_remote_budget_terminal_reserve")
        attempt = "attempt-"+uuid.uuid4().hex
        entry = {"attempt": attempt, "bytes": size, "sha256": archive["sha256"], "periodic": periodic}
        self.ledger["reservations"].append(entry)
        self._write()  # Failed writes and copies remain charged; never reset on resume.
        folder = self.root / attempt
        folder.mkdir()
        _atomic(folder / "reservation.json", canonical(entry)+b"\n")
        return folder

    def ensure_terminal_room(self):
        self._check_lease()
        entries = self.ledger["reservations"]
        if len(entries) >= REMOTE_ATTEMPTS or sum(r["bytes"] for r in entries)+ARCHIVE_LIMIT > REMOTE_LIMIT:
            raise RemoteBudgetExhausted("v12_remote_no_terminal_room_before_work")

    def publish(self, archive, *, periodic):
        from scripts.v11_recovery import _measure
        attempt = self.reserve(archive, periodic=periodic)
        path, deadline = _local(archive["output"]), time.monotonic()+600
        partial, final = attempt / "training.zip.partial", attempt / "training.zip"
        try:
            with path.open("rb") as source, partial.open("xb") as target:
                if _measure(source, archive["bytes"], deadline, target) != (archive["bytes"], archive["sha256"]):
                    raise ValueError("v12_remote_local_hash")
                target.flush()
                __import__("os").fsync(target.fileno())
            with partial.open("rb") as source:
                if _measure(source, archive["bytes"], deadline) != (archive["bytes"], archive["sha256"]):
                    raise ValueError("v12_remote_readback_hash")
            partial.rename(final)
            receipt = seal({"schema": "archvis_v12_training_backup_v1", "state": "verified",
                            "kind": "training" if archive["last_sha256"] else "status", "archive": final.name,
                            "bytes": archive["bytes"], "sha256": archive["sha256"], "last_sha256": archive["last_sha256"],
                            "periodic": periodic, "ledger_sha256": self.ledger["content_sha256"],
                            "copy_readback_wall_seconds": 600-(deadline-time.monotonic())})
            _atomic(attempt / "receipt.json", canonical(receipt)+b"\n")
            return {**receipt, "attempt": str(attempt), "local_zip": str(path)}
        except BaseException as exc:
            _atomic(attempt / "failure.json", canonical({"state": "failed", "type": type(exc).__name__,
                                                          "reason": str(exc)[:512], "local_zip": str(path)})+b"\n")
            raise

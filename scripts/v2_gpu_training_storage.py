"""Read-only bulk cache restore and independent Drive publication for v2 GPU runs."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_prepare.temporal_selection import check_pin, seal
from rewind_v1 import v2_cache as cache, v2_retention as retention
from rewind_v1.runner import LAST, _read_checkpoint
from rewind_v1.v2_budget import Ledger, ResourceLimit
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import PACKAGE_CAP, Journal, package_path, rows_for


GIB = 1024**3
MIB = 1024**2
BOUNDARY = "dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511"
SHARDS, FRAMES = 1063, 136011
SEEDS = {"A": 42, "B": 71, "C": 103}
EVAL_SEED = 71
CHECKPOINT_SCHEMA = "rewind_v2_gpu_training_checkpoint_v1"


def read_sealed(path, schema, cap=2*MIB):
    doc = json.loads(_regular_read(path, cap))
    check_pin(doc, schema, doc.get("content_sha256"))
    return doc


def publish_json(path, schema, body, owner):
    owner()
    if path.exists() or path.is_symlink():
        raise ValueError(f"v2_gpu_publication_exists: {path.name}")
    doc = seal({"schema": schema, **body})
    raw = canonical(doc)
    _atomic(path, raw)
    if _regular_read(path, len(raw), len(raw)) != raw:
        raise ValueError(f"v2_gpu_publication_readback: {path.name}")
    owner()
    return doc


class DriveLedger(Ledger):
    """The seed's Drive directory lease protects writes across VMs."""

    def __init__(self, root, binding, cfg, owner):
        super().__init__(root, binding, cfg)
        self.owner = owner

    def persist(self):
        self.owner()
        super().persist()

    @contextmanager
    def session(self, started):
        self.owner()
        self.doc = self.read()
        for row in self.doc["invocations"]:
            if row["status"] == "running":
                row.update(status="interrupted_unknown", charged_wall_seconds=28800)
        self.started = started
        self.current = {"number": len(self.doc["invocations"]) + 1, "phase": "gpu_training", "status": "running",
                        "wall_seconds_reserved": 28800, "active_seconds_observed": 0.0,
                        **{f"{k}_bytes_{l}": 0 for k in ("read", "write") for l in ("reserved", "observed")}}
        self.doc["invocations"].append(self.current)
        self.persist()
        try:
            yield self
        except BaseException as exc:
            status = "stopped" if isinstance(exc, KeyboardInterrupt) else "paused" if isinstance(exc, TimeoutError) else "failed"
            self.current.update(status=status, error=str(exc)[:512])
            raise
        finally:
            if self.current["status"] == "running":
                self.current["status"] = "settled"
            self.persist()
            self.current = None


def final_entries(journal, ledger):
    entries = journal.scan(BOUNDARY, ledger.check)
    if (len(journal.doc["records"]) != FRAMES or len(entries) != SHARDS
            or entries[-1]["content_sha256"] != BOUNDARY or entries[-1]["frames"] != 75
            or sum(entry["frames"] for entry in entries) != FRAMES):
        raise ValueError("v2_gpu_cache_final_boundary")
    return entries


def transfer_in(source, target, pin, ledger):
    """Verify the one external read. Partial reads enter the ledger before errors propagate."""
    read, hasher = 0, hashlib.sha256()
    try:
        ledger.check()
        if source.is_symlink() or source.stat().st_size != pin["bytes"]:
            raise ValueError("v2_gpu_transfer_source_size")
        with source.open("rb") as src, target.open("xb") as dst:
            while block := src.read(min(MIB, pin["bytes"] + 1 - read)):
                read += len(block)
                ledger.check()
                if read > pin["bytes"]:
                    raise ValueError("v2_gpu_transfer_source_growth")
                hasher.update(block)
                dst.write(block)
        if read != pin["bytes"] or hasher.hexdigest() != pin["sha256"]:
            raise ValueError("v2_gpu_transfer_source_sha256")
    finally:
        ledger.observed("read", read, persist=False)
        ledger.local_io("write", read)


def restore_cache(root, journal, anchor, entries, cfg, ledger):
    root.mkdir(exist_ok=True)
    staged = [e for e in entries if not (root / f"shard-{e['index']:05d}").exists()]
    missing_bytes = sum(e["package"]["bytes"] for e in staged)
    total = sum(e["package"]["bytes"] for e in entries)
    headroom = cfg["resources"]["feature_bytes"] + cfg["resources"]["run_bytes"] + 3*GIB
    if total > cfg["resources"]["cache_bytes"] or shutil.disk_usage(root).free < missing_bytes + headroom:
        raise OSError(f"v2_gpu_local_disk_preflight required_free={missing_bytes + headroom} cache_payload={total}")
    ledger.reserve("read", missing_bytes + len(staged))
    for entry in entries:
        ledger.check()
        i, package = entry["index"], entry["package"]
        folder = root / f"shard-{i:05d}"
        bind = anchor["binding"] if not i else journal.binding
        if folder.exists():
            got = cache.receipt(root, i, bind, rows_for(journal.doc, i), ledger)
        else:
            with tempfile.TemporaryDirectory(dir=root, prefix=".shard-") as directory:
                work = Path(directory)
                archive = work / "input.zip"
                transfer_in(package_path(journal.legacy if not i else journal.root, i, package), archive, package, ledger)
                ready = work / folder.name
                extract_pinned_zip(archive, package["sha256"], ready, max_members=3,
                                   max_zip_bytes=PACKAGE_CAP, max_output_bytes=PACKAGE_CAP)
                got = cache.receipt(work, i, bind, rows_for(journal.doc, i), ledger)
                if got["content_sha256"] == package["receipt_sha256"]:
                    ready.rename(folder)
        if got["content_sha256"] != package["receipt_sha256"]:
            raise ValueError("v2_gpu_cache_shard_receipt")
        if i % 32 == 31:
            ledger.persist()
        if len(entries) > 1 and i % 128 == 127:
            print(f"Local image cache verified through shard {i+1}/{SHARDS}", flush=True)
    return {"bytes": total, "drive_read_payload_bytes": missing_bytes, "reused_shards": len(entries) - len(staged)}


class Checkpoints:
    def __init__(self, root, scratch, identity, pipeline, cfg, ledger, owner):
        self.root, self.scratch, self.identity = root, scratch, identity
        self.pipeline, self.cfg, self.ledger, self.owner = pipeline, cfg, ledger, owner
        self.root.mkdir(exist_ok=True)

    def history(self):
        result = []
        for i, path in enumerate(sorted(self.root.glob("step-*.json")), 1):
            doc = read_sealed(path, CHECKPOINT_SCHEMA)
            if (path.name != f"step-{i:06d}.json" or doc["sequence"] != i or doc["identity"] != self.identity
                    or doc["previous"] != (result[-1]["content_sha256"] if result else None)
                    or result and doc["cursor"]["updates"] < result[-1]["cursor"]["updates"]):
                raise ValueError("v2_gpu_checkpoint_history")
            result.append(doc)
        return result

    def save(self, *, first=False, completed=False):
        self.owner()
        pipeline = self.pipeline
        last = pipeline.last
        if last is None:
            return None
        history = self.history()
        if not first and history and last == history[-1]["last"] and completed == history[-1]["completed"]:
            return history[-1]
        sequence = len(history)+1
        name = f"result-{sequence:06d}-{uuid.uuid4().hex}.zip"
        local = self.scratch / name
        try:
            pin = retention.package_run(pipeline.root, local, self.cfg, self.ledger, reserve_copy=True)
            self.owner()
            retention.copy_verified(local, self.root / name, pin, self.ledger,
                                    self.ledger.clock()+min(self.cfg["copy_seconds"], self.ledger.remaining(terminal=True)),
                                    pre_reserved=True)
            cursor = {name: pipeline.durable_progress[name] for name in ("stage", "epoch", "batch", "updates")}
            body = {"identity": self.identity, "sequence": sequence, "archive": pin, "last": last,
                    "cursor": cursor, "environment": pipeline.environment, "completed": completed,
                    "previous": history[-1]["content_sha256"] if history else None}
            if first:
                proof = pipeline.durable_progress["partial_loss_sums"]
                if cursor["stage"] != "frontend" or not proof["supervised_joint_views"] or not proof["first_update_gradient_l2"] > 0:
                    raise ValueError("v2_gpu_first_update_proof")
                body["first_proof"] = proof
            result = publish_json(self.root / f"step-{sequence:06d}.json", CHECKPOINT_SCHEMA, body, self.owner)
            if first:
                self.publish_first(result)
            if len(history) >= 2:
                obsolete = history[-2]["archive"]["name"]
                first_marker = self.root / "first-update.json"
                pinned = (read_sealed(first_marker, "rewind_v2_gpu_training_first_v1")["receipt"]["archive"]["name"]
                          if first_marker.exists() else None)
                if obsolete != pinned and Path(obsolete).name == obsolete:
                    self.owner()
                    (self.root / obsolete).unlink(missing_ok=True)
            print(f"Drive checkpoint {cursor} receipt={result['content_sha256']}", flush=True)
            return result
        finally:
            local.unlink(missing_ok=True)

    def restore(self, tip):
        pin = tip["archive"]
        if Path(pin["name"]).name != pin["name"]:
            raise ValueError("v2_gpu_checkpoint_archive_name")
        got = retention.restore_run(self.root / pin["name"], pin, self.pipeline.root, self.cfg,
                                    self.ledger, self.pipeline.binding)
        if got != tip["last"]["sha256"]:
            raise ValueError("v2_gpu_checkpoint_cursor_pin")
        if "first_proof" in tip:
            state = _read_checkpoint(self.pipeline.root / LAST, got, self.cfg["resources"]["checkpoint_bytes"])
            if state["progress"]["partial_loss_sums"] != tip["first_proof"] or not state["fit"]["optimizer"]["state"]:
                raise ValueError("v2_gpu_restore_first_update_proof")
            self.publish_first(tip)
        return got

    def publish_first(self, receipt):
        marker = self.root / "first-update.json"
        body = {"identity": self.identity, "receipt": receipt, "proof": receipt["first_proof"]}
        schema = "rewind_v2_gpu_training_first_v1"
        if marker.exists():
            if read_sealed(marker, schema) != seal({"schema": schema, **body}):
                raise ValueError("v2_gpu_first_update_marker_conflict")
        else:
            publish_json(marker, schema, body, self.owner)


def verify_first(root, scratch, identity_a, ledger, cfg):
    doc = read_sealed(root / "first-update.json", "rewind_v2_gpu_training_first_v1")
    receipt, proof = doc["receipt"], doc["proof"]
    check_pin(receipt, CHECKPOINT_SCHEMA, receipt.get("content_sha256"))
    if (doc["identity"] != identity_a or receipt["identity"] != identity_a or receipt["cursor"]["stage"] != "frontend"
            or receipt["cursor"]["updates"] < 1 or proof["supervised_joint_views"] <= 0
            or not 0 < proof["first_update_gradient_l2"] < float("inf")):
        raise ValueError("v2_gpu_A_first_update_binding")
    if read_sealed(root / f"step-{receipt['sequence']:06d}.json", CHECKPOINT_SCHEMA) != receipt:
        raise ValueError("v2_gpu_A_first_update_receipt")
    pin = receipt["archive"]
    with tempfile.TemporaryDirectory(dir=scratch, prefix=".A-proof-") as directory:
        work = Path(directory)
        local = work / "first.zip"
        ledger.reserve("read", pin["bytes"] + 1)
        if Path(pin["name"]).name != pin["name"]:
            raise ValueError("v2_gpu_A_archive_path")
        transfer_in(root / pin["name"], local, pin, ledger)
        extract_pinned_zip(local, pin["sha256"], work / "verified", max_members=16,
                           max_zip_bytes=cfg["resources"]["archive_bytes"], max_output_bytes=cfg["resources"]["archive_bytes"])
        state = _read_checkpoint(work / "verified" / LAST, receipt["last"]["sha256"], cfg["resources"]["checkpoint_bytes"])
        p = state["progress"]
        if (state["binding"]["experiment"] != identity_a or not state["fit"]["optimizer"]["state"]
                or {k: p[k] for k in receipt["cursor"]} != receipt["cursor"] or p["partial_loss_sums"] != proof):
            raise ValueError("v2_gpu_A_first_update_payload")
    return doc["content_sha256"]

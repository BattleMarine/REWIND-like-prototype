"""Durable prefix of completed v1.2 pair audits, bound to one frozen source plan."""

from collections import Counter
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import time

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _hash, _regular_read
from .converter import sha
from .extracted import ROLES, ROOT, _code_identity, _local
from .temporal_cache import _atomic
from .temporal_selection import check_pin, seal


SCHEMA = "archvis_v12_completed_pair_journal_v1"
ENTRY_SCHEMA = "archvis_v12_completed_pair_v1"
MAX_BYTES = 128 * 1024**2
MAX_LINE_BYTES = 64 * 1024
HEAD_BYTES = 1024**2
HEAD = "head.json"
RECORDS = "pairs.jsonl"
REUSE_POLICY = {
    "completed_hashes": "content_checked_when_pair_was_first_audited",
    "resume_check": "pinned_path_size_and_mtime_ns",
    "completed_content_rehashed_on_resume": False,
    "limitation": "Same-size content changes preserving mtime may escape metadata checks.",
}


def producer_identity():
    return {**_code_identity(), **{name: sha((ROOT / name).read_bytes()) for name in (
        "rewind_prepare/full_archvis.py", "rewind_prepare/full_audit_journal.py")}}


def binding(plan, snapshot):
    return {"schema": SCHEMA, "source_plan_sha256": plan["content_sha256"],
            "snapshot_sha256": snapshot["content_sha256"],
            "input_zip_sha256": snapshot.get("input_zip_sha256"),
            "input_receipt_sha256": snapshot.get("input_receipt_sha256"),
            "producer": producer_identity(), "reuse_policy": REUSE_POLICY}


class PairJournal:
    def __init__(self, root, expected_binding, prior_check):
        self.root = Path(root)
        self.binding, self.prior_check = expected_binding, prior_check
        self.head = None
        self.entries = []
        self.discarded_tail_bytes = 0

    def _write_head(self, head):
        document = seal(head)
        raw = canonical(document) + b"\n"
        if len(raw) > HEAD_BYTES:
            raise ValueError("v12_journal_head_budget")
        _atomic(self.root / HEAD, raw)
        self.head = head

    def load(self, plan, motions=None, check_time=lambda: None):
        head_path = _local(self.root / HEAD)
        doc = json.loads(_regular_read(head_path, HEAD_BYTES))
        check_pin(doc, SCHEMA, doc.get("content_sha256"))
        head = {k: v for k, v in doc.items() if k != "content_sha256"}
        if head.get("binding") != self.binding:
            raise ValueError("v12_journal_binding_or_producer_changed")
        count, length = head["count"], head["bytes"]
        if (type(count) is not int or not 0 <= count <= len(plan["selection"])
                or type(length) is not int or not 0 <= length <= MAX_BYTES):
            raise ValueError("v12_journal_cursor")
        path = _local(self.root / RECORDS)
        if not path.is_file() or not length <= path.stat().st_size <= MAX_BYTES:
            raise ValueError("v12_journal_size")
        entries, previous, consumed = [], digest(self.binding), 0
        exact_payloads = {}
        with path.open("rb") as stream:
            for ordinal in range(count):
                check_time()
                raw = stream.readline(min(MAX_LINE_BYTES + 1, length - consumed + 1))
                consumed += len(raw)
                if not raw.endswith(b"\n") or consumed > length or len(raw) > MAX_LINE_BYTES:
                    raise ValueError("v12_journal_partial_committed_record")
                entry = json.loads(raw)
                check_pin(entry, ENTRY_SCHEMA, entry.get("content_sha256"))
                if (type(entry["ordinal"]) is not int or entry["ordinal"] != ordinal
                        or entry["previous_sha256"] != previous):
                    raise ValueError("v12_journal_order_or_chain")
                self._check_entry(entry, plan["selection"][ordinal], motions)
                self.prior_check(entry["record"])
                row = entry["record"]
                signature = tuple(row["files"][role]["sha256"] for role in ROLES)
                if exact_payloads.setdefault(signature, row["split"]) != row["split"]:
                    raise ValueError("v12_journal_triplet_content_split_overlap")
                previous = entry["content_sha256"]
                entries.append(entry)
        if consumed != length or previous != head["last_sha256"]:
            raise ValueError("v12_journal_head_chain")
        self.entries, self.head = entries, head
        self.discarded_tail_bytes = path.stat().st_size - length
        return entries

    @staticmethod
    def _check_entry(entry, selected, motions):
        sequence, frame, split, files = selected
        row = entry["record"]
        if ((row["sequence_id"], row["frame_index"], row["split"]) != (sequence, frame, split)
                or type(row["frame_index"]) is not int or not isinstance(row["motion_id"], str)
                or (motions is not None and row["motion_id"] != motions[sequence]["motion_id"])
                or set(row["files"]) != set(ROLES)
                or set(entry["observations"]) != set(ROLES)):
            raise ValueError("v12_journal_record_scope")
        for role, path, size in files:
            item, observed = row["files"][role], entry["observations"][role]
            if (item["path"] != path or item["size_bytes"] != size or not _hash(item["sha256"])
                    or set(observed) != {"size_bytes", "mtime_ns"} or observed["size_bytes"] != size
                    or type(observed["mtime_ns"]) is not int):
                raise ValueError("v12_journal_file_identity")
        maxima, counts = entry["geometry_max"], entry["geometry_counts"]
        if (set(maxima) != set(counts) or not counts.get("projection_px")
                or any(type(v) is not int or v <= 0 for v in counts.values())
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in maxima.values())):
            raise ValueError("v12_journal_geometry_evidence")

    def _begin(self, plan, motions, observed, budget):
        if (self.root / HEAD).exists():
            self.load(plan, motions, budget.check)
        else:
            if self.root.exists():
                raise ValueError("v12_journal_incomplete_initialization")
            self.root.mkdir()
            with (self.root / RECORDS).open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            self._write_head({"schema": SCHEMA, "binding": self.binding, "count": 0, "bytes": 0,
                              "last_sha256": digest(self.binding), "invocations": []})
        for entry in self.entries:
            budget.check()
            row = entry["record"]
            for role in ROLES:
                if observed[row["sequence_id"], row["frame_index"], role] != entry["observations"][role]:
                    raise ValueError("v12_journal_completed_source_metadata_changed")
        if self.discarded_tail_bytes:
            with (self.root / RECORDS).open("r+b") as stream:
                stream.truncate(self.head["bytes"])
                stream.flush()
                os.fsync(stream.fileno())
        head = copy.deepcopy(self.head)
        for invocation in head["invocations"]:
            if invocation["status"] == "running":
                invocation.update(status="interrupted", read_bytes_is_lower_bound=True)
        head["invocations"].append({"status": "running", "start_count": head["count"],
                                     "read_bytes": budget.bytes, "elapsed_seconds": 0.,
                                     "read_bytes_is_lower_bound": True})
        self._write_head(head)

    def _statistics(self, head, budget):
        head["invocations"][-1].update(read_bytes=budget.bytes,
                                      elapsed_seconds=round(time.monotonic()-budget.started, 3))

    def append(self, record, observed, maxima, counts, budget):
        self.prior_check(record)
        entry = seal({"schema": ENTRY_SCHEMA, "ordinal": self.head["count"],
                      "previous_sha256": self.head["last_sha256"], "record": record,
                      "observations": observed, "geometry_max": maxima, "geometry_counts": dict(counts)})
        raw = canonical(entry) + b"\n"
        if len(raw) > MAX_LINE_BYTES or self.head["bytes"] + len(raw) > MAX_BYTES:
            raise ValueError("v12_journal_record_budget")
        with (self.root / RECORDS).open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        head = copy.deepcopy(self.head)
        head.update(count=head["count"]+1, bytes=head["bytes"]+len(raw), last_sha256=entry["content_sha256"])
        self._statistics(head, budget)
        self._write_head(head)
        self.entries.append(entry)

    @contextmanager
    def session(self, plan, motions, observed, budget):
        self._begin(plan, motions, observed, budget)
        error = None
        try:
            yield self.entries.copy()
        except BaseException as exc:
            error = exc
            raise
        finally:
            head = copy.deepcopy(self.head)
            self._statistics(head, budget)
            invocation = head["invocations"][-1]
            invocation.update(status="failed" if error else "content_pass_finished", read_bytes_is_lower_bound=False)
            if error:
                invocation["failure"] = {"type": type(error).__name__, "reason": str(error)[:512]}
            try:
                self._write_head(head)
            except Exception:
                if error is None:
                    raise

    def summary(self):
        counts = Counter(entry["record"]["split"] for entry in self.entries)
        return {"schema": SCHEMA, "committed_pairs": self.head["count"], "committed_by_split": dict(counts),
                "committed_bytes": self.head["bytes"], "last_sha256": self.head["last_sha256"],
                "committed_payload_bytes": sum(item["size_bytes"] for entry in self.entries
                                               for item in entry["record"]["files"].values()),
                "invocations": self.head["invocations"], "reuse_policy": REUSE_POLICY,
                "cumulative_observed_read_bytes": sum(item["read_bytes"] for item in self.head["invocations"]),
                "cumulative_read_bytes_is_lower_bound": any(item["read_bytes_is_lower_bound"]
                                                            for item in self.head["invocations"]),
                "discarded_uncommitted_tail_bytes": self.discarded_tail_bytes}

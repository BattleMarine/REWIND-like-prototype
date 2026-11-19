"""Durable cumulative budgets for the approved v2 CPU continuation."""

import json
import math
from pathlib import Path
import re

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_prepare.temporal_selection import check_pin, seal


BASELINE_SHA = "9121e075033b9b922cb8fe93e4389f04914b70eb77cd0c496d138973a4098e55"
BASELINE_SHARDS = 4
TARGET_SHARDS = 1063
TIME_CAP = 324_000
DRIVE_CAP = 60 * 1024**3
SESSION_CAP = 28_800
CLOSURE = 90
MINIMUM = 2520 + CLOSURE
METADATA_CAP = 16 * 1024
SCHEMA = "rewind_v2_cpu_quota_v1"


def publish(path, doc, owner):
    owner()
    if path.exists() or path.is_symlink():
        raise ValueError("v2_quota_publication_conflict")
    raw = canonical(doc)
    _atomic(path, raw)
    if _regular_read(path, METADATA_CAP, len(raw)) != raw:
        raise ValueError("v2_quota_publication_readback")
    owner()


class TimeQuota:
    def __init__(self, bulk, owner, *, cap=TIME_CAP, baseline=BASELINE_SHA):
        self.root = bulk / ".cpu-quota-v1"
        self.marker = bulk / ".cpu-quota-origin-v1.json"
        self.started = bulk / ".cpu-quota-started-v1.json"
        self.owner = owner
        self.cap = cap
        self.identity = digest({"baseline": baseline, "time_cap": cap, "drive_cap": DRIVE_CAP})

    def read(self, path, expected_type):
        doc = json.loads(_regular_read(path, METADATA_CAP))
        check_pin(doc, SCHEMA, doc.get("content_sha256"))
        if doc["identity"] != self.identity or doc["kind"] != expected_type:
            raise ValueError("v2_quota_contract_pin")
        return doc

    def path(self, number, kind):
        return self.root / f"{number:06d}.{kind}.json"

    def history(self, *, initialize=False, required_tip="", enforce_tip=False):
        origin = seal({"schema": SCHEMA, "identity": self.identity, "kind": "origin"})
        if self.marker.exists() or self.marker.is_symlink():
            if self.read(self.marker, "origin") != origin or not self.root.is_dir() or self.root.is_symlink():
                raise ValueError("v2_quota_history_missing")
        elif initialize and not self.started.exists() and not self.started.is_symlink() and not self.root.is_symlink() and (
                not self.root.exists() or (self.root.is_dir() and not any(self.root.iterdir()))):
            self.root.mkdir(exist_ok=True)
            publish(self.marker, origin, self.owner)
        else:
            raise ValueError("v2_quota_origin_missing")
        tip = origin["content_sha256"]
        seen = set()
        charged = unsettled = 0
        paths = set(self.root.glob("*.json"))
        number = 0
        while self.path(number, "reserve") in paths:
            reserve_path = self.path(number, "reserve")
            reserve = self.read(reserve_path, "reserve")
            paths.remove(reserve_path)
            reserved = reserve["seconds"]
            if (reserve["number"] != number or reserve["previous"] != tip or type(reserved) is not int
                    or not MINIMUM < reserved <= SESSION_CAP or charged + reserved > self.cap):
                raise ValueError("v2_quota_reservation_chain")
            tip = reserve["content_sha256"]
            seen.add(tip)
            settle_path = self.path(number, "settle")
            if settle_path in paths:
                settle = self.read(settle_path, "settle")
                paths.remove(settle_path)
                elapsed = settle["measured_seconds"]
                if (settle["number"] != number or settle["previous"] != tip
                        or type(elapsed) is not int or elapsed < 0
                        or settle["seconds"] != min(reserved, elapsed + CLOSURE)):
                    raise ValueError("v2_quota_settlement_chain")
                tip, used = settle["content_sha256"], settle["seconds"]
                seen.add(tip)
            else:
                unsettled += 1
                used = reserved
            charged += used
            number += 1
        if self.started.exists() or self.started.is_symlink():
            witness = self.read(self.started, "started")
            if not number or witness["first"] != self.read(self.path(0, "reserve"), "reserve")["content_sha256"]:
                raise ValueError("v2_quota_history_witness")
        if paths or (required_tip and required_tip not in seen):
            raise ValueError("v2_quota_history_gap_or_rollback")
        if enforce_tip and number and not required_tip:
            raise ValueError(f"v2_quota_prior_tip_required latest={tip}")
        return {"tip": tip, "charged_seconds": charged, "unsettled": unsettled, "next": number}

    def reserve(self, history, elapsed, *, session_cap=SESSION_CAP):
        seconds = min(session_cap, self.cap - history["charged_seconds"])
        if seconds <= MINIMUM + math.ceil(elapsed):
            return None
        ticket = seal({"schema": SCHEMA, "identity": self.identity, "kind": "reserve",
                       "number": history["next"], "previous": history["tip"], "seconds": seconds})
        publish(self.path(ticket["number"], "reserve"), ticket, self.owner)
        if not self.started.exists() and not self.started.is_symlink():
            initial = self.read(self.path(0, "reserve"), "reserve")
            publish(self.started, seal({"schema": SCHEMA, "identity": self.identity, "kind": "started",
                                       "first": initial["content_sha256"]}), self.owner)
        return ticket

    def settle(self, ticket, elapsed):
        number = ticket["number"]
        measured = math.ceil(elapsed)
        status = seal({"schema": SCHEMA, "identity": self.identity, "kind": "settle",
                       "number": number, "previous": ticket["content_sha256"],
                       "measured_seconds": measured, "seconds": min(ticket["seconds"], measured + CLOSURE)})
        if self.read(self.path(number, "reserve"), "reserve") != ticket:
            raise ValueError("v2_quota_settlement_ticket")
        publish(self.path(number, "settle"), status, self.owner)
        return status


class DriveQuota:
    def __init__(self, root, baseline, *, cap=DRIVE_CAP):
        self.root = root
        self.cap = cap
        self.excluded = {e["package"]["name"] for e in baseline[1:]}
        self.excluded.update(f"boundary-{e['index']:05d}.json" for e in baseline)
        self.used = self.measure()

    def measure(self):
        counted = 0
        for path in self.root.iterdir():
            if path.name in self.excluded:
                continue
            if path.name.startswith(("shard-", "boundary-", ".publish-")):
                counted += self.file_bytes(path)
            elif path.name.startswith(".incoming-"):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("v2_quota_cache_staging_type")
                counted += sum(self.file_bytes(item) for item in path.iterdir())
        return counted

    @staticmethod
    def file_bytes(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("v2_quota_cache_payload_type")
        return path.stat().st_size

    def admit(self, package_bytes):
        return self.used + package_bytes + METADATA_CAP <= self.cap

    def committed(self, entry, *, orphan):
        self.used += len(canonical(entry)) + (0 if orphan else entry["package"]["bytes"])


def baseline_entries(root, expected=BASELINE_SHA):
    from scripts.run_v2_bulk_cache import Journal

    result = []
    for number in range(BASELINE_SHARDS):
        entry = Journal.read(Path(root) / f"boundary-{number:05d}.json")
        if (entry["index"] != number or entry["frames"] != 128
                or entry["previous"] != (result[-1]["content_sha256"] if result else None)):
            raise ValueError("v2_quota_baseline_chain")
        result.append(entry)
    if result[-1]["content_sha256"] != expected:
        raise ValueError("v2_quota_baseline_pin")
    return result

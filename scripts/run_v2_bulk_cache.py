"""Append verified v2 shards on CPU, or read a pinned boundary batch on CUDA."""

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time
import uuid

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import check_pin, seal
from rewind_v1 import v2_cache as cache, v2_contract as contract, v2_retention as retention
from rewind_v1.run_stages import frame_batch
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from rewind_v1.v2_training import CacheFrames
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, MIB, read_manifest
from scripts.run_v2_gpu_cache_probe import cuda_round_trip, load_catalog


CPU_SECONDS = 28_800
GPU_SECONDS = 1_800
PACKAGE_CAP = 64 * MIB
JOURNAL_SCHEMA = "rewind_v2_bulk_boundary_v1"


def rows_for(doc, index):
    return doc["records"][index*cache.CHUNK:(index+1)*cache.CHUNK]


def package_path(folder, index, package):
    sha = package["sha256"]
    if (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            or package["name"] != f"shard-{index:05d}-{sha}.zip" or not 0 < package["bytes"] <= PACKAGE_CAP
            or package["number"] != index):
        raise ValueError("v2_bulk_package_pin")
    path = folder / package["name"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != package["bytes"]:
        raise ValueError("v2_bulk_missing_or_changed_package")
    return path


class Journal:
    def __init__(self, root, legacy, doc, anchor, binding):
        self.root, self.legacy, self.doc, self.binding = root, legacy, doc, binding
        self.journal_id = digest({"legacy": anchor["content_sha256"], "producer": binding,
                                  "total_frames": len(doc["records"])})
        self.genesis = self.entry(0, None, anchor["packages"][0], digest(anchor["binding"]))

    def entry(self, index, previous, package, binding_sha):
        rows = rows_for(self.doc, index)
        if not rows:
            raise ValueError("v2_bulk_index_outside_manifest")
        return seal({"schema": JOURNAL_SCHEMA, "journal_id": self.journal_id, "index": index,
                     "previous": previous, "package": package, "binding_sha256": binding_sha,
                     "records_sha256": digest(rows), "frames": len(rows)})

    def path(self, index):
        return self.root / f"boundary-{index:05d}.json"

    @staticmethod
    def read(path):
        raw = _regular_read(path, 16 * 1024)
        doc = json.loads(raw)
        check_pin(doc, JOURNAL_SCHEMA, doc.get("content_sha256"))
        return doc

    def scan(self, expected, check, *, allow_absent=False):
        result = []
        for index, path in enumerate(sorted(self.root.glob("boundary-*.json"))):
            check()
            if path != self.path(index):
                raise ValueError("v2_bulk_boundary_gap")
            current = self.read(path)
            if index == 0:
                if current != self.genesis:
                    raise ValueError("v2_bulk_genesis_pin")
            elif current != self.entry(index, result[-1]["content_sha256"], current["package"], digest(self.binding)):
                raise ValueError("v2_bulk_boundary_chain")
            package_path(self.legacy if index == 0 else self.root, index, current["package"])
            result.append(current)
        published = {item["package"]["name"] for item in result[1:]}
        for path in self.root.glob("shard-*.zip"):
            if path.name not in published and not path.name.startswith(f"shard-{max(1, len(result)):05d}-"):
                raise ValueError("v2_bulk_unexpected_package")
        if expected and not any(item["content_sha256"] == expected for item in result):
            if not (allow_absent and not result and expected == self.genesis["content_sha256"]):
                raise ValueError("v2_bulk_expected_boundary_not_found")
        return result

    def publish(self, current, owner):
        owner()
        index = current["index"]
        if index and self.read(self.path(index-1))["content_sha256"] != current["previous"]:
            raise ValueError("v2_bulk_predecessor_changed")
        target = self.path(index)
        if target.exists() or target.is_symlink():
            raise ValueError("v2_bulk_boundary_already_published")
        raw = canonical(current)
        _atomic(target, raw)
        if _regular_read(target, len(raw), len(raw)) != raw:
            raise ValueError("v2_bulk_boundary_readback")
        owner()


def writer_owner(lock):
    if lock.is_symlink():
        raise ValueError("v2_bulk_writer_link")
    files = list(lock.iterdir())
    if not files:
        return "empty"
    if (len(files) != 1 or files[0].is_symlink() or not files[0].is_file()
            or not re.fullmatch(r"owner-[0-9a-f]{32}", files[0].name)):
        raise ValueError("v2_bulk_writer_invalid_owner")
    return files[0].name.removeprefix("owner-")


@contextmanager
def writer(root, recover=""):
    lock, gate = root / ".writer", root / ".writer-recovery"
    token = uuid.uuid4().hex
    file = lock / ("owner-" + token)
    if recover:
        gate.mkdir()
        try:
            if writer_owner(lock) != recover:
                raise ValueError("v2_bulk_writer_recovery_pin")
            with tempfile.TemporaryDirectory(dir=root, prefix=".writer-replacement-") as temporary:
                replacement = Path(temporary) / ".writer"
                replacement.mkdir()
                (replacement / file.name).touch(exist_ok=False)
                lock.rename(root / (".abandoned-writer-" + uuid.uuid4().hex))
                if lock.exists() or lock.is_symlink():
                    raise ValueError("v2_bulk_writer_recovery_race")
                replacement.rename(lock)
        finally:
            gate.rmdir()
    else:
        if gate.exists() or gate.is_symlink():
            raise ValueError("v2_bulk_writer_recovery_in_progress")
        try:
            lock.mkdir()
        except FileExistsError as error:
            raise ValueError(f"v2_bulk_writer_locked owner={writer_owner(lock)}") from error
        file.touch(exist_ok=False)
    print(f"Writer lock {lock} owner {token}", flush=True)

    def check():
        if writer_owner(lock) != token:
            raise ValueError("v2_bulk_writer_ownership_lost")

    try:
        check()
        if gate.exists() or gate.is_symlink():
            raise ValueError("v2_bulk_writer_recovery_in_progress")
        yield check
    finally:
        check()
        file.unlink()
        lock.rmdir()


def unpack(source, package, root, index, cfg, ledger):
    if shutil.disk_usage(root).free < 2 * PACKAGE_CAP:
        raise OSError("v2_bulk_local_workspace_low")
    path = root / f"copy-{index:05d}.zip"
    retention.copy_verified(source, path, package, ledger,
                            ledger.clock()+min(cfg["copy_seconds"], ledger.remaining()))
    extract_pinned_zip(path, package["sha256"], root / f"shard-{index:05d}",
                       max_zip_bytes=PACKAGE_CAP, max_output_bytes=PACKAGE_CAP, max_members=3)
    path.unlink()


def admit(rows, ledger, cfg, *, orphan=False):
    reserve_publish = cfg["copy_seconds"] if orphan else cfg["copy_seconds"] + cfg["package_seconds"]
    if ledger.remaining() <= reserve_publish:
        return "time"
    resources = cfg["resources"]
    source_bytes = 0 if orphan else sum(p["size_bytes"] for r in rows for p in r["files"].values())
    read = (2 if orphan else 3)*PACKAGE_CAP + source_bytes
    write = PACKAGE_CAP + (0 if orphan else cache.shard_write_reservation(rows))
    for kind, extra in (("read", read), ("write", write)):
        if ledger.current[kind+"_bytes_reserved"] + extra > resources[kind+"_bytes"]:
            return kind
    return None


def add_shards(journal, expected, target, work, spec, cfg, rgb, pose, ledger, report, *,
               frame_builder=None, recover_writer="", owner_check=None, minimum_shards=0, drive_quota=None,
               prefetch_workers=None):
    metrics = None
    if prefetch_workers is not None:
        from scripts import v2_ordered_prefetch as prefetch

        if frame_builder is not None:
            raise ValueError("v2_prefetch_uses_production_decoder")
        metrics = report.setdefault("performance", {})

    def measure(key):
        return nullcontext() if metrics is None else prefetch.elapsed(metrics, key)

    journal.root.mkdir(parents=True, exist_ok=True)
    with (writer(journal.root, recover_writer) if owner_check is None else nullcontext(owner_check)) as owner:
        with measure("resume_scan_seconds"):
            entries = journal.scan(expected, ledger.check, allow_absent=True)
        if len(entries) < minimum_shards:
            raise ValueError("v2_bulk_baseline_required")
        if not entries:
            package_path(journal.legacy, 0, journal.genesis["package"])
            journal.publish(journal.genesis, owner)
            entries = [journal.genesis]
        report["starting_shards"] = len(entries)
        report["added_shards"] = 0
        report["boundary_sha256"] = entries[-1]["content_sha256"]
        report["completed_shards"] = len(entries)
        report["completed_frames"] = sum(e["frames"] for e in entries)
        while len(entries) < target:
            ledger.check()
            owner()
            index = len(entries)
            rows = rows_for(journal.doc, index)
            orphans = list(journal.root.glob(f"shard-{index:05d}-*.zip"))
            if len(orphans) > 1:
                raise ValueError("v2_bulk_conflicting_unpublished_packages")
            if drive_quota is not None and not drive_quota.admit(0 if orphans else PACKAGE_CAP):
                report["stop_reason"] = "drive_cumulative_budget"
                return
            reason = admit(rows, ledger, ledger.cfg, orphan=bool(orphans))
            if reason:
                report["stop_reason"] = reason + "_budget"
                return
            if shutil.disk_usage(work).free < 3 * PACKAGE_CAP:
                report["stop_reason"] = "local_disk_budget"
                return
            with tempfile.TemporaryDirectory(dir=work, prefix="shard-") as tmp:
                stage = Path(tmp)
                if orphans:
                    orphan = orphans[0]
                    pin = {"number": index, "name": orphan.name, "bytes": orphan.stat().st_size,
                           "sha256": orphan.stem.rsplit("-", 1)[-1]}
                    with measure("orphan_recovery_seconds"):
                        unpack(package_path(journal.root, index, pin), pin, stage, index, cfg, ledger)
                        item = cache.receipt(stage, index, journal.binding, rows, ledger)
                else:
                    ledger.cfg = {**ledger.cfg, "terminal_reserve_seconds":
                                  ledger.cfg["terminal_reserve_seconds"] + cfg["package_seconds"] + cfg["copy_seconds"]}
                    try:
                        with measure("shard_prepare_seconds"):
                            if prefetch_workers is None:
                                item = cache.publish_shard(stage, index, journal.binding, rows, spec, rgb, pose, ledger,
                                                           frame_builder=frame_builder)
                            else:
                                item = prefetch.publish_shard(stage, index, journal.binding, rows, spec, rgb, pose,
                                                              ledger, workers=prefetch_workers, metrics=metrics)
                    finally:
                        ledger.cfg = {**ledger.cfg, "terminal_reserve_seconds":
                                      ledger.cfg["terminal_reserve_seconds"] - cfg["package_seconds"] - cfg["copy_seconds"]}
                    with measure("receipt_verify_seconds"):
                        if cache.receipt(stage, index, journal.binding, rows, ledger) != item:
                            raise ValueError("v2_bulk_new_receipt_pin")
                    local_zip = stage / "package.zip"
                    with measure("package_seconds"):
                        pin = retention.pack_shard(stage / f"shard-{index:05d}", local_zip, ledger)
                    pin.update(number=index, name=f"shard-{index:05d}-{pin['sha256']}.zip")
                    final = journal.root / pin["name"]
                    if drive_quota is not None and not drive_quota.admit(pin["bytes"]):
                        report["stop_reason"] = "drive_cumulative_budget"
                        return
                    with measure("publish_zip_seconds"):
                        with tempfile.TemporaryDirectory(dir=journal.root, prefix=".incoming-") as incoming:
                            draft = Path(incoming) / pin["name"]
                            retention.copy_verified(local_zip, draft, pin, ledger,
                                                    ledger.clock()+min(cfg["copy_seconds"], ledger.remaining()))
                            owner()
                            if list(journal.root.glob(f"shard-{index:05d}-*.zip")):
                                raise ValueError("v2_bulk_concurrent_package")
                            draft.rename(final)
                        ledger.reserve("read", pin["bytes"])
                        if _hash_file(final, expected_size=pin["bytes"], check_time=ledger.check) != pin["sha256"]:
                            raise ValueError("v2_bulk_published_package_sha")
                        ledger.observed("read", pin["bytes"])
                if item["frames"] != len(rows):
                    raise ValueError("v2_bulk_receipt_frames")
                pin["receipt_sha256"] = item["content_sha256"]
                current = journal.entry(index, entries[-1]["content_sha256"], pin, digest(journal.binding))
                with measure("publish_boundary_seconds"):
                    journal.publish(current, owner)
                if drive_quota is not None:
                    drive_quota.committed(current, orphan=bool(orphans))
            entries.append(current)
            report["added_shards"] += 1
            report["completed_shards"] = len(entries)
            report["completed_frames"] += len(rows)
            report["boundary_sha256"] = current["content_sha256"]
            if metrics is not None:
                key = "recovered_frames" if orphans else "newly_encoded_frames"
                metrics[key] = metrics.get(key, 0) + len(rows)
            ledger.persist()
            print(f"Retained {len(entries)} shards; boundary {current['content_sha256']}", flush=True)
        report["stop_reason"] = "target_reached"


def read_batch(journal, expected, right, legacy_binding, work, spec, cfg, ledger, report):
    entries = journal.scan(expected, ledger.check)
    selected = next((e["index"] for e in entries if e["content_sha256"] == expected), -1)
    if not 1 <= right <= selected:
        raise ValueError("v2_bulk_gpu_window_not_pinned")
    rows = [*rows_for(journal.doc, right-1)[-1:], *rows_for(journal.doc, right)[:1]]
    if len(rows) != 2 or any(row["split"] != "train" for row in rows):
        raise ValueError("v2_bulk_gpu_train_boundary_required")
    verified = {}
    for current in entries[right-1:right+1]:
        index, package = current["index"], current["package"]
        folder = journal.legacy if index == 0 else journal.root
        unpack(package_path(folder, index, package), package, work, index, cfg, ledger)
        verified[index] = (legacy_binding if index == 0 else journal.binding, package["receipt_sha256"])
    data = CacheFrames()
    data.cache = cache.Cache.open_read_only_window(work, journal.doc, spec, ledger, verified_shards=verified)
    keys = [(row["sequence_id"], row["frame_index"]) for row in rows]
    try:
        values = frame_batch(data, keys, "train", [128, 128], ledger, {})
    finally:
        data.cache.close()
    batch = dict(zip(("image", "xy", "xy_valid"), values, strict=True))
    report["batch"] = {"keys": keys, "shards": [right-1, right], "valid_counts": batch["xy_valid"].sum(-1).tolist()}
    report["cuda"] = cuda_round_trip(batch)
    if not report["cuda"]["all_exact"]:
        raise ValueError("v2_bulk_gpu_roundtrip")


def run(args, pins):
    started = time.monotonic()
    report = {"status": "failed", "mode": args.mode, "stage": "preflight"}
    ledger = None
    try:
        torch.set_num_threads(1)
        if digest(contract.source()) != pins["consumer_source_identity"]:
            raise ValueError("v2_bulk_source_identity")
        cfg_raw = _regular_read(contract.ROOT / "configs/rewind_v2.json", MIB)
        if contract.sha(cfg_raw) != pins["config_sha256"]:
            raise ValueError("v2_bulk_config_pin")
        if available_ram() < 4 * 1024 * MIB or shutil.disk_usage(args.work).free < 384 * MIB:
            raise OSError("v2_bulk_disk_or_ram_preflight")
        cfg = json.loads(cfg_raw)
        spec_root = contract.ROOT
        extract_pinned_zip(spec_root / pins["embedded_manifest"], pins["archive_sha256"], args.work / "manifest",
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        doc, spec = read_manifest(args.work / "manifest" / MANIFEST_NAME, pins, spec_root / "reference")
        anchor = load_catalog(_regular_read(args.legacy / "catalog.json", 64*1024, pins["catalog_bytes"]), pins)
        legacy_binding = anchor["binding"]
        current = cache.binding(doc, spec, cfg)
        cpu_binding = {**current, "torch": "2.14.1+cpu"}
        policy = cache.ReadOnlyFirstShard(pins["producer_binding_sha256"], pins["consumer_source_identity"],
                                          pins["package"]["receipt_sha256"])
        policy.verify(legacy_binding, {**current, "torch": "2.14.1+cu130"})
        report["runtime"] = {key: current[key] for key in ("torch", "numpy", "pillow")}
        report["runtime"]["python"] = platform.python_version()
        report["runtime"]["torch_cuda_build"] = torch.version.cuda
        resources = cfg["resources"]
        if args.mode == "gpu":
            if current["torch"] != "2.14.1+cu130" or torch.version.cuda != "13.0" or not torch.cuda.is_available():
                raise RuntimeError("v2_bulk_observed_cuda_build_required")
            resources = {**resources, "read_bytes": 4*PACKAGE_CAP, "write_bytes": 2*PACKAGE_CAP}
        elif current["torch"] != "2.14.1+cpu":
            raise RuntimeError("v2_bulk_observed_cpu_build_required")
        policy_cfg = {**cfg, "resources": resources, "seconds_per_invocation": args.seconds,
                      "external_timeout_seconds": args.seconds,
                      "terminal_reserve_seconds": 1320 if args.mode == "cpu" else 30}
        ledger = Ledger(args.work / "usage", {"bulk_binding": digest(cpu_binding), "config": digest(cfg)}, policy_cfg)
        journal = Journal(args.bulk, args.legacy, doc, anchor, cpu_binding)
        report["genesis_sha256"] = journal.genesis["content_sha256"]
        report["total_shards"] = (len(doc["records"]) + cache.CHUNK - 1) // cache.CHUNK
        try:
            with ledger.invocation("v2_bulk_" + args.mode):
                ledger.started = started
                if args.mode == "cpu":
                    report["stage"] = "append"
                    if not 1 <= args.target <= report["total_shards"]:
                        raise ValueError("v2_bulk_target_outside_manifest")
                    add_shards(journal, args.boundary, args.target, args.work, spec, cfg,
                               args.rgb, args.pose, ledger, report, recover_writer=args.recover_writer)
                else:
                    report["stage"] = "read_boundary"
                    read_batch(journal, args.boundary, args.right, legacy_binding, args.work, spec, cfg, ledger, report)
        except TimeoutError as error:
            if args.mode != "cpu" or "boundary_sha256" not in report:
                raise
            report["stop_reason"] = str(error)[:256] or "timeout"
        if args.mode == "cpu" and report["stop_reason"] != "target_reached":
            report.update(status="paused", stage="boundary_retained")
        else:
            report.update(status="passed", stage="complete")
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)[:512]}
        raise
    finally:
        if ledger is not None and ledger.path.is_file():
            report["usage"] = ledger.snapshot()["invocations"][-1]
        report["worker_seconds"] = time.monotonic() - started
        _atomic(args.work / "result.json", canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    for name in ("pins", "work", "legacy", "bulk"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--boundary", default="")
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--target", type=int)
    parser.add_argument("--recover-writer", default="")
    parser.add_argument("--right", type=int)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--pose", type=Path)
    args = parser.parse_args(argv)
    common = (args.work, args.legacy, args.bulk)
    for i, path in enumerate(common):
        for other in common[:i]:
            a, b = path.resolve(), other.resolve()
            if a.is_relative_to(b) or b.is_relative_to(a):
                parser.error("work, legacy and bulk directories must be disjoint")
        if path.is_symlink():
            parser.error("work, legacy and bulk directory links are unsupported")
    if args.mode == "cpu":
        if not (args.target and args.rgb and args.pose and 2520 < args.seconds <= CPU_SECONDS):
            parser.error("CPU requires target, RGB, pose and the bounded CPU time budget")
        if args.recover_writer and not re.fullmatch(r"empty|[0-9a-f]{32}", args.recover_writer):
            parser.error("recovery requires the exact stopped writer owner token or empty")
        for raw in (args.rgb, args.pose):
            for path in common:
                if raw.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(raw.resolve()):
                    parser.error("RGB and pose roots must be disjoint from all write paths")
    elif args.recover_writer or not (args.right and args.boundary and 30 < args.seconds <= GPU_SECONDS):
        parser.error("GPU requires right shard index, boundary SHA and the bounded GPU time budget")
    run(args, json.loads(_regular_read(args.pins, 32*1024)))


if __name__ == "__main__":
    main()

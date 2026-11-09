"""Bounded, resumable raw RGB/pose packs for an approved temporal selection."""

from collections import defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import zipfile

from PIL import Image

from rewind_archive.contracts import canonical, digest
from rewind_archive.matching import identify
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _no_link, _regular_read, decode_rgb_payload
from .converter import sha, validate_pose_geometry
from .extracted import (
    IMAGE_SIZE, MAX_FILE_BYTES, MAX_MANIFEST_BYTES, MIB, ROLES, _code_identity, _local, _payload_path,
    _pose_targets, _raw_pose,
)
from .spec import validate_spec
from .temporal_selection import CLIP, SCHEMA as SELECTION_SCHEMA, TRAIN_COUNT, VALIDATION_COUNT, check_pin, seal


SCHEMA = "archvis_temporal_raw_cache_v1"
JOURNAL = "progress.json"
MANIFEST = "cache_manifest.json"
PART_FRAMES = 50


def producer_identity():
    return {**_code_identity(), **{
        "rewind_prepare/" + name: _hash_file(Path(__file__).with_name(name))
        for name in ("temporal_cache.py", "temporal_selection.py")
    }}


@dataclass(frozen=True)
class Limits:
    max_read_bytes: int
    max_disk_bytes: int
    max_buffer_bytes: int
    max_seconds: float
    workers: int = 1

    def __post_init__(self):
        if (any(type(value) is not int or value <= 0 for value in
                (self.max_read_bytes, self.max_disk_bytes, self.max_buffer_bytes))
                or type(self.workers) is not int or self.workers not in (1, 2, 4, 8)
                or type(self.max_seconds) not in (int, float) or not 0 < self.max_seconds <= 86400):
            raise ValueError("temporal_cache_limits")


@contextmanager
def _locked(directory):
    # An OS lock is released automatically on process exit, including a hard stop.
    with (directory / ".lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path, expected_size=None, check_time=lambda: None):
    with path.open("rb") as stream:
        if expected_size is None:
            return hashlib.file_digest(stream, "sha256").hexdigest()
        hasher = hashlib.sha256()
        remaining = expected_size
        while remaining:
            check_time()
            raw = stream.read(min(MIB, remaining))
            if not raw:
                raise ValueError("temporal_cache_part_short_read")
            hasher.update(raw)
            remaining -= len(raw)
        if stream.read(1):
            raise ValueError("temporal_cache_part_growth")
        return hasher.hexdigest()


def _check_part(path, part, check_time=lambda: None):
    info = _no_link(path)
    if (not stat.S_ISREG(info.st_mode) or info.st_size != part["bytes"]
            or _hash_file(path, part["bytes"], check_time) != part["sha256"]):
        raise ValueError("temporal_cache_part_pin")


def _write_doc(path, doc):
    _atomic(path, canonical(doc) + b"\n")


def _checked_plan(plan, expected):
    check_pin(plan, SELECTION_SCHEMA, expected)
    rows = plan["records"]
    train = [row["key"] for row in rows if row["split"] == "train"]
    val = [row["key"] for row in rows if row["split"] == "validation"]
    expected_train = [[sequence, frame] for sequence, start in plan["train_clip_starts"]
                      for frame in range(start, start + CLIP)]
    if (plan["status"] != "metadata_only" or plan["clip_frames"] != CLIP
            or len(train) != TRAIN_COUNT or train != expected_train or len(val) != VALIDATION_COUNT
            or digest(val) != plan["validation_keys_sha256"]
            or len(rows) != TRAIN_COUNT + VALIDATION_COUNT or len({tuple(row["key"]) for row in rows}) != len(rows)
            or plan["temporal_ready"] is not False or plan["train_ready"] is not False):
        raise ValueError("temporal_cache_plan_not_eligible")
    for row in rows:
        if set(row["files"]) != set(ROLES) or any(
            identify(row["files"][role]["path"], "pose" if role == "pose" else "rgb") != (*row["key"], role)
            for role in ROLES
        ):
            raise ValueError("temporal_cache_plan_file_role")


def _inspect(rows, roots, pose_limit):
    batches = []
    for row in rows:
        entries = {}
        for role in ROLES:
            pin = row["files"][role]
            path = _payload_path(roots["pose" if role == "pose" else "rgb"], pin["path"])
            info = _no_link(path)
            limit = pose_limit if role == "pose" else MAX_FILE_BYTES
            if (not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= limit
                    or pin.get("size_bytes", info.st_size) != info.st_size):
                raise ValueError("temporal_cache_source_size")
            entries[role] = (path, info.st_size, pin)
        batches.append((row, entries))
    return batches


def _read_pair(task, spec, calibration, check_time):
    row, entries = task
    stages = dict(read=0.0, hash=0.0, rgb_decode=0.0, pose_validation=0.0)
    result = {}
    for role, (path, size, pin) in entries.items():
        check_time()
        start = time.monotonic()
        raw = _regular_read(path, size, size)
        stages["read"] += time.monotonic() - start
        start = time.monotonic()
        fingerprint = sha(raw)
        if pin.get("sha256", fingerprint) != fingerprint:
            raise ValueError("temporal_cache_source_hash")
        stages["hash"] += time.monotonic() - start
        start = time.monotonic()
        if role == "pose":
            doc = _raw_pose(raw, row["key"][1])
            residuals = defaultdict(list)
            validate_pose_geometry(doc, spec, calibration, residuals)
            if not residuals["projection_px"]:
                raise ValueError("temporal_cache_pose_no_reprojection")
            _pose_targets(doc, spec)
            stages["pose_validation"] += time.monotonic() - start
        else:
            decode_rgb_payload(raw, IMAGE_SIZE, IMAGE_SIZE[0] * IMAGE_SIZE[1], Image)
            stages["rgb_decode"] += time.monotonic() - start
        result[role] = (raw, {"path": pin["path"], "size_bytes": size, "sha256": fingerprint})
    return result, stages


def _pack(path, tasks, spec, calibration, limits, check_time):
    import psutil

    timing = defaultdict(float)
    rows = []
    start = time.monotonic()
    process = psutil.Process()
    timing["rss_before_bytes"] = process.memory_info().rss
    timing["peak_observed_rss_bytes"] = timing["rss_before_bytes"]
    with ThreadPoolExecutor(max_workers=limits.workers) as pool, zipfile.ZipFile(path, "w") as pack:
        for offset in range(0, len(tasks), limits.workers):
            check_time()
            batch = tasks[offset:offset + limits.workers]
            # A batch owns all queued/running/completed payloads until the ordered write.
            # Encoded/read-copy slack and decoded RGB/pose working storage are counted separately.
            buffer_bound = sum(3 * sum(size for _, size, _ in entries.values()) + 24 * MIB
                               for _, entries in batch)
            if buffer_bound > limits.max_buffer_bytes:
                raise ValueError("temporal_cache_buffer_limit")
            futures = [pool.submit(_read_pair, task, spec, calibration, check_time) for task in batch]
            try:
                wait(futures, return_when=FIRST_EXCEPTION)
                for future in futures:
                    if future.done() and future.exception() is not None:
                        future.result()
                for (row, _), future in zip(batch, futures):
                    payloads, stats = future.result()
                    started = time.monotonic()
                    for raw, pin in payloads.values():
                        pack.writestr(zipfile.ZipInfo(pin["path"]), raw)
                    timing["pack"] += time.monotonic() - started
                    rows.append({**row, "files": {role: pin for role, (_, pin) in payloads.items()}})
                    for stage, seconds in stats.items():
                        timing[stage] += seconds
                    timing["peak_observed_rss_bytes"] = max(timing["peak_observed_rss_bytes"],
                                                           process.memory_info().rss)
                    del payloads, raw
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                futures.clear()
                if "future" in locals():
                    del future
            timing["peak_planned_buffer_bytes"] = max(timing["peak_planned_buffer_bytes"], buffer_bound)
    timing["part_wall_seconds"] = time.monotonic() - start
    return rows, dict(timing)


def prepare(plan, *, expected_sha256, rgb_root, pose_root, spec_path, calibration_path, output, limits, max_parts=None):
    """Build deterministic ZIP_STORED parts. Partial runs publish only durable part receipts."""
    _checked_plan(plan, expected_sha256)
    if max_parts is not None and (type(max_parts) is not int or max_parts < 1):
        raise ValueError("temporal_cache_max_parts")
    roots = {"rgb": _local(rgb_root, directory=True), "pose": _local(pose_root, directory=True)}
    output = Path(output).absolute()
    if any(output.is_relative_to(root) for root in roots.values()):
        raise ValueError("temporal_cache_output_inside_source")
    output.mkdir(exist_ok=True)
    output = _local(output, directory=True)
    spec_raw = _regular_read(_local(spec_path), 256 * 1024)
    calibration_raw = _regular_read(_local(calibration_path), MAX_FILE_BYTES)
    if (sha(spec_raw), sha(calibration_raw)) != (plan["spec_sha256"], plan["calibration_sha256"]):
        raise ValueError("temporal_cache_geometry_pin")
    spec, calibration = validate_spec(read_json(spec_raw)), read_json(calibration_raw)
    with _locked(output):
        return _prepare(plan, roots, spec, calibration, output, limits, max_parts)


def _prepare(plan, roots, spec, calibration, output, limits, max_parts):
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started > limits.max_seconds:
            raise TimeoutError("temporal_cache_time_limit")

    state_path = output / JOURNAL
    binding = {"selection_sha256": plan["content_sha256"], "roots": {kind: str(path) for kind, path in roots.items()},
               "limits": asdict(limits), "producer_code_sha256": producer_identity()}
    state = {"binding": binding, "parts": [], "read_bytes_reserved": 0, "status": "started"}
    if state_path.exists():
        state = read_json(_regular_read(state_path, MAX_MANIFEST_BYTES))
        if state["binding"] != binding:
            raise ValueError("temporal_cache_resume_binding")
        for index, part in enumerate(state["parts"]):
            check_time()
            if part["name"] != f"part-{index:04d}.zip":
                raise ValueError("temporal_cache_resume_part_name")
            _check_part(output / part["name"], part, check_time)
    rows = plan["records"]
    ranges = list(range(0, len(rows), PART_FRAMES))
    completed = len(state["parts"])
    end = len(ranges) if max_parts is None else min(len(ranges), completed + max_parts)
    temporary = output / ".pending-pack.zip"
    temporary.unlink(missing_ok=True)
    try:
        for part_number in range(completed, end):
            check_time()
            begin = ranges[part_number]
            started_stat = time.monotonic()
            tasks = _inspect(rows[begin:begin + PART_FRAMES], roots, spec["limits"]["max_file_bytes"])
            stat_seconds = time.monotonic() - started_stat
            payload = sum(size for _, entries in tasks for _, size, _ in entries.values())
            reserve = payload + len(tasks) * len(ROLES)
            # ZIP_STORED overhead and room to atomically replace metadata.
            pack_upper = payload + sum(128 + 2 * len(pin["path"].encode("utf-8"))
                                       for _, entries in tasks for _, _, pin in entries.values()) + 4096
            metadata_upper = 2 * len(canonical(state)) + 4 * MIB
            used = sum(item.stat().st_size for item in output.iterdir() if item.is_file())
            if state["read_bytes_reserved"] + reserve > limits.max_read_bytes:
                raise TimeoutError("temporal_cache_read_limit")
            if (used + pack_upper + metadata_upper > limits.max_disk_bytes
                    or shutil.disk_usage(output).free < pack_upper + metadata_upper):
                raise TimeoutError("temporal_cache_disk_limit")
            state.update(status="running", read_bytes_reserved=state["read_bytes_reserved"] + reserve,
                         pending={"part": part_number, "bytes_reserved": reserve})
            _write_doc(state_path, state)
            packed, timing = _pack(temporary, tasks, spec, calibration, limits, check_time)
            check_time()
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            if temporary.stat().st_size > pack_upper:
                raise ValueError("temporal_cache_zip_size")
            started_hash = time.monotonic()
            name = f"part-{part_number:04d}.zip"
            pack_size = temporary.stat().st_size
            descriptor = {"name": name, "bytes": pack_size, "sha256": _hash_file(temporary, pack_size, check_time),
                          "records": packed}
            timing.update(stat_seconds=stat_seconds, pack_hash_seconds=time.monotonic() - started_hash)
            temporary.replace(output / name)
            state["parts"].append(descriptor)
            state.setdefault("timing", []).append(timing)
            state.pop("pending")
            _write_doc(state_path, state)
        state.update(status="completed" if end == len(ranges) else "partial")
        state.pop("failure", None)
        if state["status"] == "completed":
            manifest = seal({"schema": SCHEMA, "source_kind": "native_rgb_png_and_pose_json",
                             "preprocessing": "none_raw_source_bytes_v1", "binding": {
                                 key: plan[key] for key in ("spec_sha256", "calibration_sha256", "dataset_version",
                                                            "inventory_sha256", "validation_keys_sha256")},
                             "selection_sha256": plan["content_sha256"], "parts": state["parts"],
                             "train_ready": False, "temporal_ready": False, "blockers": plan["blockers"],
                             "producer_code_sha256": binding["producer_code_sha256"]})
            _write_doc(output / MANIFEST, manifest)
            state["cache_sha256"] = manifest["content_sha256"]
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "partial_failure",
                     failure={"type": type(exc).__name__, "reason": str(exc)[:512]},
                     last_invocation_wall_seconds=time.monotonic() - started)
        with suppress(OSError):
            _write_doc(state_path, state)
        raise
    else:
        state["last_invocation_wall_seconds"] = time.monotonic() - started
        _write_doc(state_path, state)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return state


def restore(cache, *, expected_sha256, output, max_disk_bytes, max_seconds):
    """Copy whole immutable parts locally. Only a complete verified copy is published."""
    started = time.monotonic()
    if (type(max_disk_bytes) is not int or max_disk_bytes <= 0
            or type(max_seconds) not in (int, float) or not 0 < max_seconds <= 86400):
        raise ValueError("temporal_restore_limits")
    cache = _local(cache, directory=True)
    raw = _regular_read(cache / MANIFEST, MAX_MANIFEST_BYTES)
    manifest = read_json(raw)
    check_pin(manifest, SCHEMA, expected_sha256)
    parts = manifest["parts"]
    if (not parts or len(parts) > (TRAIN_COUNT + VALIDATION_COUNT + PART_FRAMES - 1) // PART_FRAMES
            or any(part["name"] != f"part-{index:04d}.zip" or type(part["bytes"]) is not int
                   or not 0 < part["bytes"] <= PART_FRAMES * (3 * MAX_FILE_BYTES + 4096)
                   for index, part in enumerate(parts))
            or sum(part["bytes"] for part in parts) + len(raw) > max_disk_bytes):
        raise ValueError("temporal_restore_part_limits")
    output = Path(output).absolute()
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("temporal_restore_output_location")
    output.mkdir(exist_ok=True)
    output = _local(output, directory=True)
    copied = reused = 0

    def check_time():
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("temporal_restore_time_limit")

    with _locked(output):
        marker = output / ".restore"
        if marker.exists() and _regular_read(marker, 128).decode("ascii") != expected_sha256:
            raise ValueError("temporal_restore_output_binding")
        if not marker.exists():
            if any(entry.name != ".lock" for entry in output.iterdir()):
                raise ValueError("temporal_restore_output_not_empty")
            _atomic(marker, expected_sha256.encode("ascii"))
        temp = output / ".pending-pack.zip"
        temp.unlink(missing_ok=True)
        for part in parts:
            check_time()
            target = output / part["name"]
            if target.exists():
                _check_part(target, part, check_time)
                reused += 1
                continue
            source = cache / part["name"]
            info = _no_link(source)
            if not stat.S_ISREG(info.st_mode) or info.st_size != part["bytes"]:
                raise ValueError("temporal_restore_source_size")
            if shutil.disk_usage(output).free < part["bytes"] + len(raw):
                raise TimeoutError("temporal_restore_disk_limit")
            remaining = part["bytes"]
            hasher = hashlib.sha256()
            try:
                with source.open("rb") as reader, temp.open("xb") as writer:
                    while remaining:
                        check_time()
                        block = reader.read(min(remaining, MIB))
                        if not block:
                            raise ValueError("temporal_restore_short_read")
                        remaining -= len(block)
                        hasher.update(block)
                        writer.write(block)
                    if reader.read(1) or hasher.hexdigest() != part["sha256"]:
                        raise ValueError("temporal_restore_source_hash")
                    writer.flush()
                    os.fsync(writer.fileno())
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
            copied += part["bytes"]
        _atomic(output / MANIFEST, raw)
    return {"status": "completed", "cache_sha256": expected_sha256, "copied_bytes": copied,
            "reused_parts": reused, "elapsed_seconds": time.monotonic() - started,
            "train_ready": False, "temporal_ready": False}


class RawCache:
    """Pair reads from pinned local pack members, with no eager image tensors."""

    def __init__(self, cache, expected_sha256):
        self.root = _local(cache, directory=True)
        self.manifest = read_json(_regular_read(self.root / MANIFEST, MAX_MANIFEST_BYTES))
        check_pin(self.manifest, SCHEMA, expected_sha256)
        self.index = {}
        for index, part in enumerate(self.manifest["parts"]):
            if part["name"] != f"part-{index:04d}.zip":
                raise ValueError("temporal_cache_part_name")
            for row in part["records"]:
                key = tuple(row["key"])
                if key in self.index or row["split"] not in ("train", "validation"):
                    raise ValueError("temporal_cache_pair_identity")
                self.index[key] = (part["name"], row)

    def read_pair(self, sequence, frame, *, split):
        name, row = self.index[sequence, frame]
        if row["split"] != split:
            raise ValueError("temporal_cache_pair_split")
        result = {}
        with zipfile.ZipFile(_payload_path(self.root, name)) as archive:
            for role in ROLES:
                pin = row["files"][role]
                info = archive.getinfo(pin["path"])
                if (info.file_size != pin["size_bytes"] or not 1 <= info.file_size <= MAX_FILE_BYTES
                        or info.compress_type != zipfile.ZIP_STORED):
                    raise ValueError("temporal_cache_member_size")
                with archive.open(info) as stream:
                    raw = stream.read(info.file_size + 1)
                if len(raw) != pin["size_bytes"] or sha(raw) != pin["sha256"]:
                    raise ValueError("temporal_cache_member_pin")
                result[role] = raw
        return result

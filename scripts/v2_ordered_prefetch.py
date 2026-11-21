"""A bounded source reader for the unchanged v2 shard format."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import tempfile
from threading import Event, Lock
import time

import numpy as np
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _no_link, _signature
from rewind_prepare.coordinates import camera_transform
from rewind_prepare.extracted import _payload_path, _raw_pose
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_cache as cache
from rewind_v1 import v2_contract as contract


WORKERS = (1, 4, 8)
MIB = 1024**2
WINDOW_BYTES = 64 * MIB


@contextmanager
def elapsed(metrics, key):
    start = time.monotonic()
    try:
        yield
    finally:
        metrics[key] = metrics.get(key, 0.0) + time.monotonic() - start


def read_payload(path, expected, check, observed):
    """Use the pinned size and same-handle checks, with exact chunk I/O accounting."""
    info = _no_link(path)
    if not stat.S_ISREG(info.st_mode) or not 0 < expected <= 8*MIB or info.st_size != expected:
        raise ValueError("v2_prefetch_source_size")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(os.open(path, flags), "rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        if _signature(before) != _signature(info):
            raise ValueError("v2_prefetch_source_changed")
        data = bytearray(expected)
        buffer = memoryview(data)
        offset = 0
        while offset < expected:
            check()
            n = handle.readinto(buffer[offset:offset+MIB])
            if not n:
                raise ValueError("v2_prefetch_source_short")
            observed(n)
            offset += n
        if (_signature(os.fstat(handle.fileno())) != _signature(before)
                or _signature(_no_link(path)) != _signature(before)):
            raise ValueError("v2_prefetch_source_changed")
    return data


@contextmanager
def ordered_sources(rows, rgb, pose, workers, ledger, metrics):
    """Yield one ordered stream. The coordinator alone updates the persisted ledger."""
    if workers not in WORKERS:
        raise ValueError("v2_prefetch_worker_count")
    sizes = [sum(pin["size_bytes"] for pin in row["files"].values()) for row in rows]
    if any(not 0 < size <= WINDOW_BYTES for size in sizes):
        raise ValueError("v2_prefetch_frame_bytes")
    ledger.reserve("read", sum(sizes))
    cancelled = Event()
    guard = Lock()
    deadline = ledger.clock() + ledger.remaining()
    counters = {"source_bytes": 0, "read_and_sha_worker_seconds": 0.0}

    def check():
        if cancelled.is_set() or ledger.clock() >= deadline:
            raise TimeoutError("v2_prefetch_cancelled_or_deadline")

    def observed(amount):
        with guard:
            counters["source_bytes"] += amount

    def load(row):
        start = time.monotonic()
        try:
            values = {}
            for role, pin in row["files"].items():
                check()
                path = _payload_path(Path(pose if role == "pose" else rgb), pin["path"])
                value = read_payload(path, pin["size_bytes"], check, observed)
                if contract.sha(value) != pin["sha256"]:
                    raise ValueError("v2_source_sha")
                check()
                values[role] = value
            return values
        finally:
            with guard:
                counters["read_and_sha_worker_seconds"] += time.monotonic() - start

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v2-source") if workers > 1 else None
    pending = deque()
    window_size = submitted = 0

    def stream():
        nonlocal window_size, submitted
        for sequence, row in enumerate(rows):
            ledger.check()
            if pool is None:
                metrics["peak_window_bytes"] = max(metrics.get("peak_window_bytes", 0), sizes[sequence])
                metrics["peak_window_frames"] = 1
                with elapsed(metrics, "read_wait_seconds"):
                    yield_value = load(row)
                yield yield_value
                del yield_value
                continue
            while (submitted < len(rows) and len(pending) < workers
                   and window_size + sizes[submitted] <= WINDOW_BYTES):
                pending.append(pool.submit(load, rows[submitted]))
                window_size += sizes[submitted]
                submitted += 1
                metrics["peak_window_bytes"] = max(metrics.get("peak_window_bytes", 0), window_size)
                metrics["peak_window_frames"] = max(metrics.get("peak_window_frames", 0), len(pending))
            future = pending[0]
            with elapsed(metrics, "read_wait_seconds"):
                while True:
                    ledger.check()
                    try:
                        raw = future.result(timeout=0.2)
                        break
                    except FutureTimeout:
                        if future.done():
                            raw = future.result()
                            break
            yield raw
            pending.popleft()
            window_size -= sizes[sequence]
            del raw, future

    stream_iterator = stream()
    try:
        yield stream_iterator
    finally:
        cancelled.set()
        stream_iterator.close()
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            with guard:
                for key, value in counters.items():
                    metrics[key] = metrics.get(key, 0) + value
                ledger.observed("read", counters["source_bytes"])


def decode(raw, row, spec):
    # Keep Pillow warning contexts and Torch decoding on the coordinator thread.
    image, target = cache._frame(raw, row["frame_index"], spec)
    pose = _raw_pose(raw["pose"], row["frame_index"])
    coord = spec["coordinates"]
    cameras = [camera_transform(pose["camera_"+side], coord["mm_per_raw_unit"], coord["raw_consistency_tolerance"])[0]
               for side in ("left", "right")]
    target["rotation64"] = torch.tensor([camera["rotation_matrix"] for camera in cameras], dtype=torch.float64)
    target["translation64"] = torch.tensor([camera["translation"] for camera in cameras], dtype=torch.float64)
    return image, target


def publish_shard(root, number, binding, rows, spec, rgb, pose, ledger, *, workers, metrics):
    """Use the original v2 encoder, tensor order, serialization and receipt contract."""
    ledger.reserve("write", cache.shard_write_reservation(rows))
    with tempfile.TemporaryDirectory(dir=root, prefix=".pending-") as tmp:
        staged = Path(tmp) / f"shard-{number:05d}"
        staged.mkdir()
        images = np.lib.format.open_memmap(staged/"images.npy", mode="w+", dtype=np.float32,
                                           shape=(len(rows), 2, 3, 128, 128))
        targets = []
        try:
            with ordered_sources(rows, rgb, pose, workers, ledger, metrics) as stream:
                for i, (row, raw) in enumerate(zip(rows, stream, strict=True)):
                    with elapsed(metrics, "decode_seconds"):
                        image, target = decode(raw, row, spec)
                    images[i] = image.numpy()
                    targets.append(target)
                    del raw
            images.flush()
        finally:
            del images
        with elapsed(metrics, "serialize_seconds"):
            ledger.check()
            with (staged/"images.npy").open("r+b") as handle:
                os.fsync(handle.fileno())
            tensors = {name: torch.stack([target[name] for target in targets]) for name in targets[0]}
            if not all(torch.isfinite(value).all() for value in tensors.values()):
                raise ValueError("v2_cache_nonfinite")
            with (staged/"targets.pt").open("xb") as handle:
                torch.save(tensors, handle)
                handle.flush()
                os.fsync(handle.fileno())
            sizes = {name: (staged/name).stat().st_size for name in ("images.npy", "targets.pt")}
            files = {name: {"bytes": size, "sha256": _hash_file(staged/name, expected_size=size, check_time=ledger.check)}
                     for name, size in sizes.items()}
            receipt = seal({"schema": "rewind_v2_shard_v1", "number": number, "binding": binding,
                            "records_sha256": digest(rows), "frames": len(rows), "files": files})
            _atomic(staged/"receipt.json", canonical(receipt))
            ledger.check()
            staged.rename(root/staged.name)
            ledger.observed("write", sum(sizes.values()))
    return receipt

"""Bounded Drive reads and local staging for the verified 256px input cache."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import stat
from threading import Event
import time
import uuid

from rewind_m1.local_rgb import _no_link, _signature
from rewind_v1.v2_budget import ResourceLimit
from scripts.run_v2_cache256 import PACKAGE_CAP
from scripts.v2_gpu_training_cache256 import (CHUNK, FINAL_IO_RESERVE, FRAME_BYTES, INDEX_SCHEMA, DiskCache,
                                              RangeCache256, build_index, sha)
from scripts.v2_gpu_training_storage import MIB, publish_json


WORKERS = 4
WINDOW = 8
WINDOW_BYTES = 44*MIB


@dataclass(frozen=True)
class Span:
    shard: int
    offset: int
    size: int
    checksum: str
    prefix: str


@dataclass
class Fetch:
    path: Path
    reserved: int
    refs: int = 1
    future: Future = field(default_factory=Future)
    observed: int = 0
    local_written: int = 0
    local_read: int = 0
    transfer_seconds: float = 0.
    index_seconds: float = 0.
    accounted: bool = False
    consumed: bool = False


class StagingDiskCache(DiskCache):
    """Keep outstanding temporary files in the same disk quota as LRU entries."""

    def __init__(self, previous):
        self.path, self.cap = previous.path, previous.cap
        self.entries, self.used = previous.entries, previous.used
        self.staged = 0

    def make_space(self, amount):
        super().make_space(amount+self.staged)


class IndexMeter:
    def __init__(self, job, check):
        self.job, self.check = job, check

    def local_io(self, kind, size):
        if kind != "read":
            raise ValueError("v2_gpu256_prefetch_index_io")
        self.job.local_read += size


@contextmanager
def verified_source(source, package_bytes):
    before = _no_link(source)
    if not stat.S_ISREG(before.st_mode) or before.st_size != package_bytes:
        raise ValueError("v2_gpu256_prefetch_package_size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(source, flags), "rb", buffering=0) as handle:
        if _signature(os.fstat(handle.fileno())) != _signature(before):
            raise ValueError("v2_gpu256_prefetch_package_changed")
        yield handle
        if (_signature(os.fstat(handle.fileno())) != _signature(before)
                or _signature(_no_link(source)) != _signature(before)):
            raise ValueError("v2_gpu256_prefetch_package_changed")


def stage_range(job, source, entry, span, check):
    size = entry["package"]["bytes"]
    if not 0 <= span.offset < span.offset+span.size <= size:
        raise ValueError("v2_gpu256_prefetch_range_size")
    raw = bytearray(span.size)
    view = memoryview(raw)
    started = time.monotonic()
    try:
        check()
        with verified_source(source, size) as handle:
            handle.seek(span.offset)
            while job.observed < span.size:
                check()
                begin = job.observed
                read = handle.readinto(view[begin:begin+MIB])
                if not read:
                    raise ValueError("v2_gpu256_prefetch_range_truncated")
                job.observed += read
        if sha(raw) != span.checksum:
            raise ValueError("v2_gpu256_frame_or_target_range_sha256")
    finally:
        job.transfer_seconds = time.monotonic()-started
    check()
    with job.path.open("xb") as out:
        job.local_written += out.write(raw)
        if job.local_written != span.size:
            raise OSError("v2_gpu256_prefetch_short_local_write")
        out.flush()
        os.fsync(out.fileno())
    return raw


def stage_package(job, source, entry, binding, check):
    pin = entry["package"]
    started = time.monotonic()
    hasher = hashlib.sha256()
    try:
        check()
        with verified_source(source, pin["bytes"]) as handle, job.path.open("xb") as out:
            while job.observed < pin["bytes"]:
                check()
                block = handle.read(min(MIB, pin["bytes"]-job.observed))
                if not block:
                    raise ValueError("v2_gpu256_prefetch_zip_truncated")
                job.observed += len(block)
                hasher.update(block)
                written = out.write(block)
                job.local_written += written
                if written != len(block):
                    raise OSError("v2_gpu256_prefetch_short_local_write")
            check()
            extra = handle.read(1)
            job.observed += len(extra)
            if extra or hasher.hexdigest() != pin["sha256"]:
                raise ValueError("v2_gpu256_prefetch_zip_sha256")
            out.flush()
            os.fsync(out.fileno())
    finally:
        job.transfer_seconds = time.monotonic()-started
    started = time.monotonic()
    try:
        return build_index(job.path, entry, binding, IndexMeter(job, check))
    finally:
        job.index_seconds = time.monotonic()-started


def fetch_job(job, task, source, entry, detail, cancelled, deadline, clock):
    def check():
        if cancelled.is_set() or clock() >= deadline:
            raise TimeoutError("v2_gpu256_prefetch_cancelled_or_deadline")

    if job.future.set_running_or_notify_cancel():
        try:
            result = task(job, source, entry, detail, check)
            check()
        except BaseException as exc:
            job.future.set_exception(exc)
        else:
            job.future.set_result(result)


class ParallelRangeCache256(RangeCache256):
    """Workers prepare immutable bytes. The caller owns the ledger, LRU and receipts."""

    def __init__(self, local, *args, **kwargs):
        local = Path(local)
        local.mkdir(exist_ok=True)
        if local.is_symlink():
            raise ValueError("v2_gpu256_local_cache_link")
        self.stage = local/".parallel-input"
        if self.stage.is_symlink():
            raise ValueError("v2_gpu256_prefetch_staging_link")
        self.stage.mkdir(exist_ok=True)
        for path in self.stage.iterdir():
            if not re.fullmatch(r"[0-9a-f]{32}\.part", path.name) or path.is_symlink() or not path.is_file():
                raise ValueError("v2_gpu256_prefetch_staging_entry")
            path.unlink()
        super().__init__(local, *args, **kwargs)
        self.disk = StagingDiskCache(self.disk)
        self.pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="v2-gpu256-read")
        self.jobs, self.window = {}, deque()
        self.cancelled = Event()
        self.order = iter(())
        self.split = None
        self.metrics.update(prefetch_workers=WORKERS, prefetch_window_frames=WINDOW, prefetch_window_byte_cap=WINDOW_BYTES,
                            prefetch_submitted_spans=0, prefetch_consumed_spans=0, prefetch_discarded_spans=0,
                            prefetch_discarded_bytes=0, prefetch_wait_seconds=0., prefetch_drain_seconds=0.,
                            prefetch_peak_pending_bytes=0, prefetch_peak_pending_spans=0, prefetch_peak_staging_bytes=0,
                            prefetch_cold_submitted=0, prefetch_cold_published=0, prefetch_cold_fallback=0,
                            prefetch_peak_cold_jobs=0)

    def set_order(self, keys, split):
        self.drain()
        self.order = iter(keys)
        self.split = split

    def _account(self, key, job):
        if job.accounted:
            return
        self.ledger.observed("read", job.observed, persist=False)
        self.ledger.local_io("write", job.local_written)
        self.ledger.local_io("read", job.local_read)
        label = "range" if isinstance(key, Span) else "zip"
        self.metrics[f"drive_{label}_bytes"] += job.observed
        self.metrics[f"drive_{label}_seconds"] += job.transfer_seconds
        self.metrics["index_seconds"] += job.index_seconds
        job.accounted = True

    def _release_stage(self, job):
        job.path.unlink(missing_ok=True)
        self.disk.staged -= job.reserved
        job.reserved = 0

    def _retire(self, key, job):
        self._account(key, job)
        if not job.consumed:
            self.metrics["prefetch_discarded_spans"] += 1
            self.metrics["prefetch_discarded_bytes"] += job.observed
        self._release_stage(job)

    def _drop_jobs(self):
        if not self.jobs:
            return
        started = time.monotonic()
        self.cancelled.set()
        for job in self.jobs.values():
            job.future.cancel()
        # A cancelled future need not be dequeued by the pool to release the
        # reservation. Running workers must finish before accounting/cleanup.
        pending = [job.future for job in self.jobs.values() if not job.future.cancelled()]
        interrupted = None
        while pending:
            try:
                _, pending = wait(pending, timeout=.2)
            except KeyboardInterrupt as exc:
                interrupted = exc
        for key, job in self.jobs.items():
            self._account(key, job)
        cleanup_error = None
        for key, job in self.jobs.items():
            try:
                self._retire(key, job)
            except OSError as exc:
                self.disk.staged -= job.reserved
                cleanup_error = cleanup_error or exc
        self.jobs.clear()
        self.cancelled = Event()
        self.metrics["prefetch_drain_seconds"] += time.monotonic()-started
        if interrupted is not None:
            raise interrupted
        if cleanup_error is not None:
            raise cleanup_error

    def drain(self):
        self.order = iter(())
        self.split = None
        self.window.clear()
        self._drop_jobs()

    def _reap(self):
        for key, job in list(self.jobs.items()):
            if job.future.done():
                self._account(key, job)
                if not job.refs:
                    self._retire(key, job)
                    del self.jobs[key]

    def _submit(self, key, *, current):
        if key in self.jobs:
            self.jobs[key].refs += 1
            return True
        if len(self.jobs) >= 2*WINDOW:
            return False
        warm = isinstance(key, Span)
        shard = key.shard if warm else key
        entry = self.entries[shard]
        package = entry["package"]
        name = key.prefix+key.checksum if warm else "z-"+package["sha256"]
        if name in self.disk.entries or (warm and "z-"+package["sha256"] in self.disk.entries):
            return False
        warm_bytes = sum(span.size for span in self.jobs if isinstance(span, Span))
        if warm and warm_bytes+key.size > WINDOW_BYTES:
            return False
        cold_jobs = sum(isinstance(item, int) and not job.consumed for item, job in self.jobs.items())
        if not warm and cold_jobs >= WINDOW:
            return False
        read_bytes = key.size if warm else package["bytes"]+1
        if not current:
            available = (self.budget.credit + self.ledger.cfg["resources"]["read_bytes"] - FINAL_IO_RESERVE
                         - self.ledger.current["read_bytes_reserved"])
            if available < read_bytes+PACKAGE_CAP+1:
                return False
        local_bytes = key.size if warm else package["bytes"]
        try:
            self.disk.make_space(local_bytes)
            self.budget.reserve(read_bytes)
        except ResourceLimit:
            return False
        job = Fetch(self.stage/(uuid.uuid4().hex+".part"), local_bytes)
        deadline = self.ledger.clock()+self.ledger.remaining()
        self.disk.staged += local_bytes
        self.jobs[key] = job
        task, detail = (stage_range, key) if warm else (stage_package, self.binding)
        try:
            self.pool.submit(fetch_job, job, task, self.store/package["name"], entry, detail, self.cancelled,
                             deadline, self.ledger.clock)
        except BaseException:
            # This also covers an interrupt racing with submission: cancellation
            # succeeds for a never-started future, or drain joins a running one.
            job.future.cancel()
            raise
        if warm:
            self.metrics["prefetch_submitted_spans"] += 1
        else:
            self.metrics["prefetch_cold_submitted"] += 1
            self.metrics["prefetch_peak_cold_jobs"] = max(self.metrics["prefetch_peak_cold_jobs"], cold_jobs+1)
        self.metrics["prefetch_peak_pending_spans"] = max(self.metrics["prefetch_peak_pending_spans"], len(self.jobs))
        self.metrics["prefetch_peak_pending_bytes"] = max(self.metrics["prefetch_peak_pending_bytes"],
                                                          warm_bytes+(key.size if warm else 0))
        self.metrics["prefetch_peak_staging_bytes"] = max(self.metrics["prefetch_peak_staging_bytes"], self.disk.staged)
        return True

    def _fill(self):
        while len(self.window) < WINDOW:
            key = next(self.order, None)
            if key is None:
                return
            number, slot = divmod(self.mapping[tuple(key)], CHUNK)
            current = not self.window
            marker = self.index_root/f"shard-{number:05d}.json"
            if number not in self.indices and not marker.is_file():
                tasks = (number,)
            else:
                try:
                    info = self.indices.get(number) or super()._index(number)
                except (OSError, ValueError):
                    if current:
                        raise
                    self.window.append((tuple(key), ()))
                    continue
                image = Span(number, info["image_offset"]+slot*FRAME_BYTES, FRAME_BYTES, info["frame_sha256"][slot], "f-")
                target = Span(number, info["target_offset"], info["target_bytes"], info["target_sha256"], "t-")
                tasks = (image,) if number in self.targets else (target, image)
            self.window.append((tuple(key), tuple(item for item in tasks if self._submit(item, current=current))))

    def _consume(self, key, job, cache_name):
        started = time.monotonic()
        try:
            while True:
                self.ledger.check()
                try:
                    result = job.future.result(timeout=.2)
                    break
                except FutureTimeout:
                    if job.future.done():
                        result = job.future.result()
                        break
        finally:
            self.metrics["prefetch_wait_seconds"] += time.monotonic()-started
        self._account(key, job)
        if not job.consumed:
            if cache_name not in self.disk.entries:
                if job.path.stat().st_size != job.reserved:
                    raise ValueError("v2_gpu256_staging_size")
                self.disk.publish(cache_name, job.path)
            self._release_stage(job)
            job.consumed = True
        return result

    def _index(self, number):
        if number in self.indices:
            return self.indices[number]
        marker = self.index_root/f"shard-{number:05d}.json"
        if marker.is_file():
            return super()._index(number)
        job = self.jobs.get(number)
        if job is None:
            self._drop_jobs()
            self.metrics["prefetch_cold_fallback"] += 1
            return super()._index(number)
        entry = self.entries[number]
        package = entry["package"]
        index = self._consume(number, job, "z-"+package["sha256"])
        binding = {"cache": self.identity, "package": package, "records_sha256": entry["records_sha256"], "index": number}
        publish_json(marker, INDEX_SCHEMA, {"binding": binding, **index}, self.owner)
        result = super()._index(number)
        self.metrics["reused_index_shards"] -= 1
        self.metrics["indexed_shards"] += 1
        self.metrics["prefetch_cold_published"] += 1
        return result

    def _span(self, number, offset, count, expected, prefix):
        span = Span(number, offset, count, expected, prefix)
        name = prefix+expected
        job = self.jobs.get(span)
        if job is not None:
            first = not job.consumed
            raw = self._consume(span, job, name)
            self.metrics["prefetch_consumed_spans"] += int(first)
            return raw
        if (name not in self.disk.entries and
                "z-"+self.entries[number]["package"]["sha256"] not in self.disk.entries):
            self._drop_jobs()
        return super()._span(number, offset, count, expected, prefix)

    def frame(self, key, split):
        try:
            if self.split is None:
                return super().frame(key, split)
            self._reap()
            self._fill()
            if not self.window or (tuple(key), split) != (self.window[0][0], self.split):
                raise ValueError("v2_gpu256_prefetch_order_mismatch")
            _, tasks = self.window.popleft()
            result = super().frame(key, split)
            for item in tasks:
                if item in self.jobs:
                    self.jobs[item].refs -= 1
            self._reap()
            return result
        except BaseException:
            self.drain()
            raise

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            super().close()

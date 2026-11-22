"""Bounded, byte-exact random access to ZIP_STORED stereo frames at 256px."""

from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import zipfile

import numpy as np
import torch

from rewind_archive.contracts import digest
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import check_pin
from rewind_v1.v2_budget import ResourceLimit
from scripts.run_v2_cache256 import RECEIPT_SCHEMA, package_file
from scripts.v2_gpu_training_storage import GIB, MIB, publish_json, read_sealed, transfer_in


SIDE = 256
CHUNK = 128
FRAME_BYTES = 2*3*SIDE*SIDE*4
INDEX_SCHEMA = "rewind_v2_gpu_256_frame_index_v1"
DISK_HEADROOM = 7*GIB
FINAL_IO_RESERVE = 8*GIB
TARGET_RAM_CAP = 384*MIB
HASH = re.compile(r"[0-9a-f]{64}")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class ReadBudget:
    """Persist a read reservation before transfer, with independent terminal backup headroom."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.credit = 0

    def reserve(self, needed):
        self.ledger.check()
        if needed > self.credit:
            resources = self.ledger.cfg["resources"]
            available = resources["read_bytes"] - FINAL_IO_RESERVE - self.ledger.current["read_bytes_reserved"]
            increment = min(max(128*MIB, needed-self.credit), available)
            if increment < needed-self.credit:
                raise ResourceLimit("v2_gpu256_input_read_reserve_for_terminal_backup")
            self.ledger.reserve("read", increment)
            self.credit += increment
        self.credit -= needed


class DiskCache:
    """Slot-local LRU for whole verified ZIPs, frames and target blobs."""

    def __init__(self, path, cap):
        self.path, self.cap = Path(path), cap
        self.path.mkdir(exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("v2_gpu256_local_cache_link")
        files = [entry for entry in self.path.iterdir() if re.fullmatch(r"[fzt]-[0-9a-f]{64}", entry.name)]
        if any(entry.is_symlink() or not entry.is_file() for entry in files):
            raise ValueError("v2_gpu256_local_cache_entry")
        self.entries = OrderedDict((p.name, p.stat().st_size) for p in sorted(files, key=lambda p: p.stat().st_mtime_ns))
        self.used = sum(self.entries.values())
        self.make_space(0)

    def make_space(self, amount):
        free = shutil.disk_usage(self.path).free
        while self.entries and (self.used+amount > self.cap or free < DISK_HEADROOM+amount):
            name, size = self.entries.popitem(last=False)
            (self.path/name).unlink()
            self.used -= size
            free = shutil.disk_usage(self.path).free
        if self.used+amount > self.cap or free < DISK_HEADROOM+amount:
            raise ResourceLimit("v2_gpu256_local_disk_headroom")

    def get(self, name):
        if name in self.entries:
            self.entries.move_to_end(name)
            file = self.path/name
            os.utime(file)
            return file
        return None

    def publish(self, name, path):
        target = self.path/name
        path.replace(target)
        self.entries[name] = target.stat().st_size
        self.used += self.entries[name]
        return target

    def put(self, name, raw):
        self.make_space(len(raw))
        target = self.path/name
        _atomic(target, raw)
        self.entries[name] = len(raw)
        self.used += len(raw)


def member_start(handle, info):
    if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size or info.flag_bits & 1:
        raise ValueError("v2_gpu256_zip_stored_required")
    handle.seek(info.header_offset)
    raw = handle.read(30)
    if len(raw) != 30:
        raise ValueError("v2_gpu256_zip_local_header")
    signature, name, extra = struct.unpack("<4s22xHH", raw)
    if signature != b"PK\x03\x04" or handle.read(name) != info.filename.encode("ascii"):
        raise ValueError("v2_gpu256_zip_local_header")
    return info.header_offset + 30 + name + extra


def build_index(path, entry, binding, ledger):
    """Derive offset and frame SHA only after the caller pins the complete package bytes."""
    package = entry["package"]
    with zipfile.ZipFile(path) as bundle, path.open("rb") as handle:
        if sorted(bundle.namelist()) != ["images.npy", "receipt.json", "targets.pt"]:
            raise ValueError("v2_gpu256_zip_members")
        receipt_info = bundle.getinfo("receipt.json")
        if not 0 < receipt_info.file_size <= MIB:
            raise ValueError("v2_gpu256_receipt_size")
        receipt = json.loads(bundle.read(receipt_info))
        check_pin(receipt, RECEIPT_SCHEMA, package["receipt_sha256"])
        frames = entry["frames"]
        expected_shape = (frames, 2, 3, SIDE, SIDE)
        if (receipt["index"] != entry["index"] or receipt["binding"] != binding or receipt["size"] != SIDE
                or receipt["records_sha256"] != entry["records_sha256"] or tuple(receipt["image_shape"]) != expected_shape
                or receipt["image_dtype"] != "float32" or set(receipt["files"]) != {"images.npy", "targets.pt"}):
            raise ValueError("v2_gpu256_shard_binding")
        image, target = bundle.getinfo("images.npy"), bundle.getinfo("targets.pt")
        image_start, target_start = member_start(handle, image), member_start(handle, target)
        if not 0 < target.file_size <= 4*MIB or any(x.file_size != receipt["files"][x.filename]["bytes"] for x in (image, target)):
            raise ValueError("v2_gpu256_member_sizes")
        target_raw = bundle.read(target)
        if sha(target_raw) != receipt["files"]["targets.pt"]["sha256"]:
            raise ValueError("v2_gpu256_target_sha256")
        with bundle.open(image) as stream:
            version = np.lib.format.read_magic(stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
            if version not in readers:
                raise ValueError("v2_gpu256_npy_version")
            shape, fortran, dtype = readers[version](stream)
            header = stream.tell()
            if shape != expected_shape or fortran or dtype.str != "<f4" or header + frames*FRAME_BYTES != image.file_size:
                raise ValueError("v2_gpu256_npy_contract")
            stream.seek(0)
            verify = hashlib.sha256(stream.read(header))
            hashes = []
            for _ in range(frames):
                ledger.check()
                raw = stream.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES:
                    raise ValueError("v2_gpu256_short_frame")
                verify.update(raw)
                hashes.append(sha(raw))
            if verify.hexdigest() != receipt["files"]["images.npy"]["sha256"]:
                raise ValueError("v2_gpu256_image_member_sha256")
        ledger.local_io("read", image.file_size + target.file_size)
        return {"image_offset": image_start+header, "frame_sha256": hashes, "target_offset": target_start,
                "target_bytes": target.file_size, "target_sha256": sha(target_raw)}


def load_targets(raw, frames, joints):
    tensors = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    shapes = {"xy": (frames,2,joints,2), "xy_valid": (frames,2,joints), "target_mm": (frames,joints,3),
              "target_valid": (frames,joints), "camera": (frames,2,9), "rotation64": (frames,2,3,3),
              "translation64": (frames,2,3)}
    if set(tensors) != set(shapes):
        raise ValueError("v2_gpu256_target_names")
    for key, shape in shapes.items():
        tensor = tensors[key]
        dtype = torch.bool if key.endswith("valid") else torch.float64 if key.endswith("64") else torch.float32
        if tensor.shape != shape or tensor.dtype != dtype or not torch.isfinite(tensor).all():
            raise ValueError("v2_gpu256_target_shape_dtype")
    return tensors


class RangeCache256:
    def __init__(self, local, index_root, store, entries, doc, spec, binding, cfg, ledger, owner):
        self.store, self.entries, self.records = store, entries, doc["records"]
        self.spec, self.binding, self.ledger, self.owner = spec, binding, ledger, owner
        self.indices, self.targets, self.handles = {}, OrderedDict(), OrderedDict()
        self.target_memory = 0
        self.index_root = index_root
        self.index_root.mkdir(exist_ok=True)
        if self.index_root.is_symlink():
            raise ValueError("v2_gpu256_index_link")
        self.disk = DiskCache(local, cfg["resources"]["cache_bytes"])
        self.budget = ReadBudget(ledger)
        self.identity = digest({"journal": binding, "final": entries[-1]["content_sha256"], "resolution": SIDE})
        self.mapping = {(r["sequence_id"], r["frame_index"]): i for i, r in enumerate(self.records)}
        self.metrics = {"indexed_shards": 0, "reused_index_shards": 0, "frame_hits": 0, "drive_zip_bytes": 0,
                        "drive_range_bytes": 0, "drive_zip_seconds": 0., "drive_range_seconds": 0., "index_seconds": 0.,
                        "local_cache_payload_bytes": self.disk.used}
        self.last_persist = time.monotonic()

    def prepare_frontend(self):
        """Use the original shuffled order, staging a shard only on its first access."""

    def _source(self, entry):
        return package_file(self.store, entry["index"], entry["package"])

    def _index(self, number):
        if number in self.indices:
            return self.indices[number]
        entry = self.entries[number]
        package = entry["package"]
        marker = self.index_root/f"shard-{number:05d}.json"
        binding = {"cache": self.identity, "package": package, "records_sha256": entry["records_sha256"], "index": number}
        if marker.exists():
            doc = read_sealed(marker, INDEX_SCHEMA, 64*1024)
            self.metrics["reused_index_shards"] += 1
        else:
            cache_name = "z-"+package["sha256"]
            archive = self.disk.get(cache_name)
            if archive is None:
                self.disk.make_space(package["bytes"])
                self.budget.reserve(package["bytes"]+1)
                started = time.monotonic()
                with tempfile.TemporaryDirectory(dir=self.disk.path, prefix=".stage-") as temporary:
                    staged = Path(temporary)/"source.zip"
                    transfer_in(self._source(entry), staged, package, self.ledger)
                    archive = self.disk.publish(cache_name, staged)
                self.metrics["drive_zip_bytes"] += package["bytes"]
                self.metrics["drive_zip_seconds"] += time.monotonic()-started
            elif _hash_file(archive, expected_size=package["bytes"], check_time=self.ledger.check) != package["sha256"]:
                raise ValueError("v2_gpu256_reused_full_zip_sha256")
            started = time.monotonic()
            index = build_index(archive, entry, self.binding, self.ledger)
            doc = publish_json(marker, INDEX_SCHEMA, {"binding": binding, **index}, self.owner)
            self.metrics["indexed_shards"] += 1
            self.metrics["index_seconds"] += time.monotonic()-started
        end_images = doc["image_offset"] + entry["frames"]*FRAME_BYTES
        if (doc["binding"] != binding or len(doc["frame_sha256"]) != entry["frames"]
                or not all(HASH.fullmatch(h) for h in [*doc["frame_sha256"], doc["target_sha256"]])
                or not 0 < doc["image_offset"] < end_images <= doc["target_offset"] < package["bytes"]
                or not 0 < doc["target_bytes"] <= min(4*MIB, package["bytes"]-doc["target_offset"])):
            raise ValueError("v2_gpu256_range_index_binding")
        self.indices[number] = doc
        return doc

    def _span(self, number, offset, count, expected, prefix):
        entry = self.entries[number]
        name = prefix+expected
        local = self.disk.get(name)
        archive = self.disk.get("z-"+entry["package"]["sha256"]) if local is None else None
        if local is not None:
            self.ledger.local_io("read", count)
            raw = local.read_bytes() if local.stat().st_size == count else b""
            if prefix == "f-":
                self.metrics["frame_hits"] += 1
        elif archive is not None:
            with archive.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(count)
            self.ledger.local_io("read", len(raw))
        else:
            self.budget.reserve(count)
            started = time.monotonic()
            if number not in self.handles:
                self.handles[number] = self._source(entry).open("rb", buffering=0)
                if len(self.handles) > 16:
                    self.handles.popitem(last=False)[1].close()
            handle = self.handles[number]
            self.handles.move_to_end(number)
            handle.seek(offset)
            acquired = bytearray()
            try:
                while len(acquired) < count:
                    self.ledger.check()
                    block = handle.read(count-len(acquired))
                    if not block:
                        break
                    acquired.extend(block)
            finally:
                self.ledger.observed("read", len(acquired), persist=False)
                self.metrics["drive_range_bytes"] += len(acquired)
                self.metrics["drive_range_seconds"] += time.monotonic()-started
            raw = acquired
        if len(raw) != count or sha(raw) != expected:
            raise ValueError("v2_gpu256_frame_or_target_range_sha256")
        if local is None and (archive is None or prefix == "t-"):
            self.disk.put(name, raw)
            self.ledger.local_io("write", count)
        return raw

    def frame(self, key, split):
        self.ledger.check()
        i = self.mapping[tuple(key)]
        if self.records[i]["split"] != split:
            raise ValueError("v2_gpu256_frame_split")
        number, slot = divmod(i, CHUNK)
        info = self._index(number)
        if number not in self.targets:
            raw = self._span(number, info["target_offset"], info["target_bytes"], info["target_sha256"], "t-")
            targets = load_targets(raw, self.entries[number]["frames"], len(self.spec["joint_names"]))
            size = sum(t.nbytes for t in targets.values())
            while self.targets and self.target_memory+size > TARGET_RAM_CAP:
                self.target_memory -= self.targets.popitem(last=False)[1][0]
            self.targets[number] = (size, targets)
            self.target_memory += size
        self.targets.move_to_end(number)
        raw = self._span(number, info["image_offset"]+slot*FRAME_BYTES, FRAME_BYTES, info["frame_sha256"][slot], "f-")
        image = np.frombuffer(raw, dtype="<f4").reshape(2, 3, SIDE, SIDE)
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("v2_gpu256_image_range")
        if time.monotonic()-self.last_persist >= 60:
            self.ledger.persist()
            self.last_persist = time.monotonic()
        self.metrics["local_cache_payload_bytes"] = self.disk.used
        return torch.from_numpy(image.copy()), {key: t[slot].clone() for key, t in self.targets[number][1].items()}

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.targets.clear()
        self.target_memory = 0

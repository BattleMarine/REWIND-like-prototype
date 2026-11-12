"""Resumable exact-float32 frame preparation; no raw-cache or Drive uploads."""

from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha
from rewind_prepare.coordinates import camera_transform
from rewind_prepare.extracted import IMAGE_SIZE, MIB, ROLES, _local, _payload_path, _pose_targets, _raw_pose
from rewind_prepare.temporal_cache import _atomic, _check_part, _hash_file, _locked
from rewind_prepare.temporal_selection import check_pin, seal
from .data import _field, frame_image
from .full_contract import AUDIT_ZIP_SHA, MANIFEST_SHA, PREPROCESSING
from .training import _source


GIB = 1024**3
CHUNK = 128
CACHE_LIMIT = 12*GIB
SCHEMA = "archvis_v12_frame_chunk_v1"


class PreparationBudget:
    def __init__(self, *, seconds=7200, read_bytes=8*GIB, reserve=120):
        if not 0 <= reserve < seconds <= 7200 or not 1 <= read_bytes <= 8*GIB:
            raise ValueError("v12_prepare_budget")
        self.started, self.seconds, self.reserve = time.monotonic(), seconds, reserve
        self.limit, self.source_bytes, self.cache_verified_bytes = read_bytes, 0, 0

    def check(self):
        if time.monotonic()-self.started >= self.seconds-self.reserve:
            raise TimeoutError("v12_prepare_time_limit")

    def read_source(self, root, entry):
        self.check()
        if self.source_bytes + entry["size_bytes"] > self.limit:
            raise TimeoutError("v12_prepare_read_limit")
        raw = _regular_read(_payload_path(root, entry["path"]), 8*MIB, entry["size_bytes"])
        self.source_bytes += len(raw)
        if sha(raw) != entry["sha256"]:
            raise ValueError("v12_prepare_source_sha")
        self.check()
        return raw


def cache_binding(manifest, spec_raw):
    return {"audit_zip_sha256": AUDIT_ZIP_SHA, "manifest_sha256": MANIFEST_SHA,
            "spec_sha256": sha(spec_raw), "source_code_sha256": _source(),
            "preprocessing": PREPROCESSING, "size": [128, 128], "dtype": "float32",
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__), "pillow_version": __import__("PIL").__version__,
            "chunk_frames": CHUNK, "target_scale_mm": 100., "total_frames": len(manifest["records"])}


def _chunk_rows(records, number):
    return records[number*CHUNK:(number+1)*CHUNK]


def _read_receipt(folder, binding, rows, number, budget):
    doc = json.loads(_regular_read(folder / "receipt.json", MIB))
    check_pin(doc, SCHEMA, doc.get("content_sha256"))
    if (doc["binding"] != binding or doc["number"] != number
            or doc["records_sha256"] != digest(rows) or doc["frames"] != len(rows)
            or set(doc["files"]) != {"images.npy", "targets.pt"}
            or {p.name for p in folder.iterdir()} != {"images.npy", "targets.pt", "receipt.json"}):
        raise ValueError("v12_frame_chunk_binding")
    for name, pin in doc["files"].items():
        _check_part(folder / name, {"name": name, **pin}, budget.check)
        budget.cache_verified_bytes = getattr(budget, "cache_verified_bytes", 0)+pin["bytes"]
    return doc


def _frame(raw, frame, spec):
    doc = _raw_pose(raw["pose"], frame)
    fields = _pose_targets(doc, spec)
    xy, valid = zip(*(_field(field, 2) for field in fields[:2]))
    xyz, xyz_valid = _field(fields[2], 3)
    coords = spec["coordinates"]
    cameras = {side: camera_transform(doc["camera_"+side], coords["mm_per_raw_unit"],
                                      coords["raw_consistency_tolerance"])[0] for side in ("left", "right")}
    return frame_image(raw, [128, 128]), {
        "xy": scale_xy(torch.stack(xy), IMAGE_SIZE, [128, 128]), "xy_valid": torch.stack(valid),
        "target_mm": xyz, "target_valid": xyz_valid, "camera": encode_camera(cameras, 100.)}


def _publish_chunk(root, rows, number, binding, spec, rgb_root, pose_root, budget):
    folder = root / f"chunk-{number:05d}"
    if folder.exists() or folder.is_symlink():
        raise ValueError("v12_prepare_chunk_exists")
    with tempfile.TemporaryDirectory(dir=root, prefix=".pending-") as temporary:
        staged = Path(temporary) / folder.name
        staged.mkdir()
        images = np.lib.format.open_memmap(staged / "images.npy", mode="w+", dtype=np.float32,
                                          shape=(len(rows), 2, 3, 128, 128))
        metadata = []
        try:
            for i, row in enumerate(rows):
                raw = {role: budget.read_source(pose_root if role == "pose" else rgb_root, row["files"][role])
                       for role in ROLES}
                image, target = _frame(raw, row["frame_index"], spec)
                images[i] = image.numpy()
                metadata.append(target)
            images.flush()
        finally:
            del images
        with (staged / "images.npy").open("r+b") as stream:
            os.fsync(stream.fileno())
        budget.check()
        tensors = {name: torch.stack([item[name] for item in metadata]) for name in metadata[0]}
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("v12_prepare_tensor_nonfinite")
        with (staged / "targets.pt").open("xb") as stream:
            torch.save(tensors, stream)
            stream.flush()
            os.fsync(stream.fileno())
        files = {name: {"bytes": (staged / name).stat().st_size, "sha256": _hash_file(staged / name, check_time=budget.check)}
                 for name in ("images.npy", "targets.pt")}
        receipt = seal({"schema": SCHEMA, "number": number, "binding": binding,
                        "records_sha256": digest(rows), "frames": len(rows), "files": files})
        _atomic(staged / "receipt.json", canonical(receipt)+b"\n")
        # One directory publication is the commit boundary; no head can precede its files.
        staged.rename(folder)
    return receipt


def prepare_frames(manifest, spec_raw, rgb_root, pose_root, output, *, budget=None, max_chunks=None):
    budget = budget or PreparationBudget()
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("v12_prepare_max_chunks")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    output = _local(output, directory=True)
    rgb_root, pose_root = _local(rgb_root, directory=True), _local(pose_root, directory=True)
    binding, records, spec = cache_binding(manifest, spec_raw), manifest["records"], json.loads(spec_raw)
    receipts, created, state = [], 0, "paused"
    with _locked(output):
        # Only our unpublished staging directories may be removed after an interrupted process.
        for pending in output.glob(".pending-*"):
            resolved = _local(pending, directory=True)
            if not resolved.is_relative_to(output) or resolved == output:
                raise ValueError("v12_prepare_staging_path")
            shutil.rmtree(resolved)
        if any(p.name not in {".lock", "binding.json", "preparation.json"} and not p.name.startswith("chunk-")
               for p in output.iterdir()):
            raise ValueError("v12_prepare_unknown_output")
        binding_path = output / "binding.json"
        if binding_path.exists():
            if json.loads(_regular_read(binding_path, MIB)) != binding:
                raise ValueError("v12_prepare_binding")
        else:
            if any(p.name.startswith("chunk-") for p in output.iterdir()):
                raise ValueError("v12_prepare_missing_binding")
            _atomic(binding_path, canonical(binding)+b"\n")
        failure = None
        try:
            numbers = (len(records)+CHUNK-1)//CHUNK
            known = {f"chunk-{number:05d}" for number in range(numbers)}
            if any(p.name.startswith("chunk-") and p.name not in known for p in output.iterdir()):
                raise ValueError("v12_prepare_extra_chunks")
            for number in range(numbers):
                budget.check()
                rows = _chunk_rows(records, number)
                folder = output / f"chunk-{number:05d}"
                if folder.exists():
                    receipt = _read_receipt(_local(folder, directory=True), binding, rows, number, budget)
                else:
                    if any((output / f"chunk-{later:05d}").exists() for later in range(number+1, numbers)):
                        raise ValueError("v12_prepare_chunk_gap")
                    if max_chunks is not None and created >= max_chunks:
                        break
                    needed = sum(f["size_bytes"] for row in rows for f in row["files"].values())
                    if budget.source_bytes+needed > budget.limit:
                        raise TimeoutError("v12_prepare_read_limit")
                    # Include completed data, the active temporary chunk, metadata and JSON overhead.
                    used = sum(pin["bytes"] for r in receipts for pin in r["files"].values())
                    reserve = len(rows)*(2*3*128*128*4 + len(spec["joint_names"])*48 + 1024)+2*MIB
                    if used+reserve > CACHE_LIMIT or shutil.disk_usage(output).free < reserve+2*GIB:
                        raise TimeoutError("v12_prepare_disk_limit")
                    receipt = _publish_chunk(output, rows, number, binding, spec, rgb_root, pose_root, budget)
                    created += 1
                receipts.append(receipt)
            if len(receipts) == numbers:
                state = "completed"
        except TimeoutError as exc:
            reason = str(exc)
        except BaseException as exc:
            failure, state, reason = exc, "stopped" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc)[:512]
        else:
            reason = None
        report = seal({"schema": "archvis_v12_frame_preparation_v1", "binding": binding,
                       "status": state, "reason": reason, "completed_frames": sum(r["frames"] for r in receipts),
                       "chunks": [{"number": r["number"], "receipt_sha256": r["content_sha256"]} for r in receipts],
                       "source_bytes_this_invocation": budget.source_bytes,
                       "cache_verified_bytes_this_invocation": budget.cache_verified_bytes,
                       "elapsed_seconds": time.monotonic()-budget.started})
        _atomic(output / "preparation.json", canonical(report)+b"\n")
        if failure is not None:
            raise failure
    return report


class FrameCache:
    """At most two small target chunks and two read-only image mappings are retained."""

    def __init__(self, root, manifest, spec_raw, budget):
        self.root, self.records = _local(root, directory=True), manifest["records"]
        self.joints = len(json.loads(spec_raw)["joint_names"])
        self.binding, self.budget = cache_binding(manifest, spec_raw), budget
        self.index = {(r["sequence_id"], r["frame_index"]): i for i, r in enumerate(self.records)}
        self.chunks, self.receipts = OrderedDict(), []
        for number in range((len(self.records)+CHUNK-1)//CHUNK):
            self.receipts.append(_read_receipt(self.root / f"chunk-{number:05d}", self.binding,
                                              _chunk_rows(self.records, number), number, budget))
        self.identity = digest({"binding": self.binding, "chunks": [r["content_sha256"] for r in self.receipts]})

    def _load(self, number):
        if number not in self.chunks:
            folder = self.root / f"chunk-{number:05d}"
            images = np.load(folder / "images.npy", mmap_mode="r", allow_pickle=False)
            count, joints = self.receipts[number]["frames"], self.joints
            raw = _regular_read(folder / "targets.pt", 4*MIB)
            targets = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if images.shape != (count, 2, 3, 128, 128) or images.dtype != np.float32:
                raise ValueError("v12_frame_cache_image_shape")
            if set(targets) != {"xy", "xy_valid", "target_mm", "target_valid", "camera"}:
                raise ValueError("v12_frame_cache_target_fields")
            shapes = {"xy": (count, 2, joints, 2), "xy_valid": (count, 2, joints),
                      "target_mm": (count, joints, 3), "target_valid": (count, joints), "camera": (count, 2, 9)}
            if any(targets[k].shape != shape or targets[k].dtype != (torch.bool if k.endswith("valid") else torch.float32)
                   or not torch.isfinite(targets[k]).all() for k, shape in shapes.items()):
                raise ValueError("v12_frame_cache_target_shape")
            self.chunks[number] = images, targets
            if len(self.chunks) > 2:
                _, (previous, _) = self.chunks.popitem(last=False)
                previous._mmap.close()
        self.chunks.move_to_end(number)
        return self.chunks[number]

    def close(self):
        for images, _ in self.chunks.values():
            images._mmap.close()
        self.chunks.clear()

    def frame(self, key, split):
        index = self.index[tuple(key)]
        if self.records[index]["split"] != split:
            raise ValueError("v12_frame_cache_split")
        images, targets = self._load(index//CHUNK)
        slot = index % CHUNK
        image = torch.from_numpy(np.array(images[slot], copy=True))
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("v12_frame_cache_image_finite")
        return image, {name: value[slot].clone() for name, value in targets.items()}

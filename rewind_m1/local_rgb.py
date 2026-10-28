"""추출된 PNG의 명시적 manifest를 읽는 로컬 lazy stereo 입력.

이미지/torch/Pillow는 sample을 요청할 때만 읽거나 import한다.
학습 계약 발급, resize, augmentation, 이미지 캐시는 제공하지 않는다.
2026-10-27 구현: 실행 검증은 하지 않았다.
"""

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat
import struct
import warnings
import zlib

from rewind_archive.contracts import PreparationError, digest, safe_path, validate_manifest
from rewind_archive.matching import identify
from rewind_data.audit import read_json


class RGBContractError(ValueError):
    """Manifest, 경로, 파일 또는 decoder 입력 계약을 위반했다."""


@dataclass(frozen=True)
class RGBLimits:
    max_manifest_bytes: int = 4 * 1024**2
    max_records: int = 10000
    max_sources: int = 32
    max_file_bytes: int = 8 * 1024**2
    max_pixels_per_view: int = 4 * 1024**2
    max_batch_size: int = 4
    max_batch_tensor_bytes: int = 128 * 1024**2

    def __post_init__(self):
        ceilings = (16 * 1024**2, 100000, 32, 64 * 1024**2,
                    16 * 1024**2, 16, 512 * 1024**2)
        for (name, value), ceiling in zip(asdict(self).items(), ceilings):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise RGBContractError(f"rgb_limit:{name}")


def _fields(value, names, reason):
    if not isinstance(value, dict) or set(value) != set(names):
        raise RGBContractError(reason)


def _text(value, maximum=2048):
    return (isinstance(value, str) and bool(value.strip()) and
            len(value.encode("utf-8")) <= maximum and
            not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value):
    try:
        if safe_path(value) != value or any(c in value for c in "*?[]"):
            raise RGBContractError("rgb_canonical_literal_path")
    except PreparationError as exc:
        raise RGBContractError("rgb_unsafe_path") from exc
    return value


def _no_link(path):
    info = path.lstat()
    # Windows junction을 포함한 reparse point도 거부한다.
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RGBContractError("rgb_link_or_reparse_point")
    return info


def _signature(info):
    common = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    # Windows ctime은 생성시각으로, 경로와 handle 조회값의 차이를 변경으로 단정하지 않는다.
    return common if os.name == "nt" else common + (info.st_ctime_ns,)


def _regular_read(path, maximum, expected_size=None):
    """같은 handle에서 상한 있는 읽기와 변경 감지. 내용 hash는 호출자가 검사한다."""
    info = _no_link(path)
    if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum:
        raise RGBContractError("rgb_file_type_or_byte_limit")
    if expected_size is not None and info.st_size != expected_size:
        raise RGBContractError("rgb_file_size_mismatch")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) |
             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = os.fstat(handle.fileno())
        if _signature(before) != _signature(info):
            raise RGBContractError("rgb_file_changed")
        raw = handle.read(info.st_size + 1)
        after = os.fstat(handle.fileno())
    if len(raw) != info.st_size or _signature(before) != _signature(after):
        raise RGBContractError("rgb_file_changed")
    if _signature(_no_link(path)) != _signature(after):
        raise RGBContractError("rgb_file_changed")
    return raw


class LocalStereoRGB:
    """Map-style API: len, metadata(index), [index], load_batch([indices]).

    생성자는 manifest와 파일metadata만 검사한다. PNG bytes/hash/decode 검사는
    요청된 sample에만 적용하며, 반환 tensor/metadata를 내부에 캐시하지 않는다.
    """

    def __init__(self, manifest_path, root, *, limits=None):
        self._limits = RGBLimits() if limits is None else limits
        if not isinstance(self._limits, RGBLimits):
            raise RGBContractError("rgb_limits_type")
        try:
            base = Path(root).absolute()
            for component in (base, *base.parents):
                _no_link(component)
            if not base.is_dir():
                raise RGBContractError("rgb_root_directory")
            self._root = base.resolve(strict=True)
            manifest_path = Path(manifest_path).absolute()
            for component in manifest_path.parents:
                _no_link(component)
            raw = _regular_read(manifest_path, self._limits.max_manifest_bytes)
            document = read_json(raw)
            self._validate(document)
        except RGBContractError:
            raise
        except PreparationError as exc:
            raise RGBContractError(f"rgb_archive_contract:{exc.reason}") from exc
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise RGBContractError("rgb_manifest_or_local_file_invalid") from exc
        self._manifest = copy.deepcopy(document)
        self._records = tuple(self._manifest["records"])
        self._manifest_sha256 = document["content_sha256"]
        self._width, self._height = document["image_size"]

    def _file(self, relative):
        relative = _path(relative)
        if self._root.resolve(strict=True) != self._root:
            raise RGBContractError("rgb_root_changed")
        _no_link(self._root)
        path = self._root
        for part in relative.split("/"):
            path = path / part
            _no_link(path)
        if not path.resolve(strict=True).is_relative_to(self._root):
            raise RGBContractError("rgb_path_outside_root")
        if not path.is_file():
            raise RGBContractError("rgb_not_regular_file")
        return path

    def _validate(self, doc):
        _fields(doc, ("schema_version", "id", "dataset_version", "data_kind",
                      "image_size", "sources", "records", "content_sha256"), "rgb_manifest_fields")
        if doc["schema_version"] != "local_stereo_rgb_v1":
            raise RGBContractError("rgb_manifest_schema")
        if not _text(doc["id"], 128) or not _text(doc["dataset_version"], 128):
            raise RGBContractError("rgb_manifest_identity")
        if doc["data_kind"] not in ("dataset_sample", "synthetic_fixture"):
            raise RGBContractError("rgb_data_kind")
        if not _hash(doc["content_sha256"]) or digest({k: v for k, v in doc.items() if k != "content_sha256"}) != doc["content_sha256"]:
            raise RGBContractError("rgb_manifest_hash")
        size = doc["image_size"]
        if (not isinstance(size, list) or len(size) != 2 or
                any(type(n) is not int or n < 4 for n in size) or
                size[0] * size[1] > self._limits.max_pixels_per_view):
            raise RGBContractError("rgb_image_size_or_pixel_limit")
        if size[0] * size[1] * 2 * 3 * 4 > self._limits.max_batch_tensor_bytes:
            raise RGBContractError("rgb_single_tensor_byte_limit")
        sources = doc["sources"]
        if not isinstance(sources, dict) or not 1 <= len(sources) <= self._limits.max_sources:
            raise RGBContractError("rgb_sources_limit")
        for key, source in sources.items():
            if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key) is None:
                raise RGBContractError("rgb_source_id")
            _fields(source, ("archive_manifest", "extraction_manifest_sha256"), "rgb_source_fields")
            manifest = validate_manifest(source["archive_manifest"])
            if manifest["kind"] != "rgb" or manifest["dataset_version"] != doc["dataset_version"] or not _hash(source["extraction_manifest_sha256"]):
                raise RGBContractError("rgb_source_kind_version_or_hash")
            synthetic_source = manifest["source_scope"] == "synthetic_fixture"
            if synthetic_source != (doc["data_kind"] == "synthetic_fixture"):
                raise RGBContractError("rgb_source_data_kind_mismatch")
        rows = doc["records"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= self._limits.max_records:
            raise RGBContractError("rgb_record_limit")
        identities, files, sequence_splits = set(), set(), {}
        for row in rows:
            _fields(row, ("sequence_id", "frame_index", "split", "split_evidence", "views"), "rgb_record_fields")
            seq, frame = _path(row["sequence_id"]), row["frame_index"]
            if type(frame) is not int or not 0 <= frame < 10**12:
                raise RGBContractError("rgb_frame_index")
            if (seq, frame) in identities:
                raise RGBContractError("rgb_duplicate_sequence_frame")
            identities.add((seq, frame))
            if row["split"] not in ("train", "validation", "test", "unknown") or not _text(row["split_evidence"]):
                raise RGBContractError("rgb_split_declaration")
            if sequence_splits.setdefault(seq, row["split"]) != row["split"]:
                raise RGBContractError("rgb_sequence_split_overlap")
            _fields(row["views"], ("left", "right"), "rgb_missing_or_extra_view")
            for side in ("left", "right"):
                view = row["views"][side]
                _fields(view, ("path", "size_bytes", "sha256", "image_size", "source_id",
                               "member", "tar_complete"), "rgb_view_fields")
                relative = _path(view["path"])
                if relative.casefold() in files:
                    raise RGBContractError("rgb_duplicate_file_path")
                files.add(relative.casefold())
                if not isinstance(view["source_id"], str) or view["source_id"] not in sources:
                    raise RGBContractError("rgb_missing_source_reference")
                if identify(_path(view["member"]), "rgb") != (seq, frame, side):
                    raise RGBContractError("rgb_member_identity_mismatch")
                if (view["tar_complete"] is not True or view["image_size"] != size or
                        not isinstance(view["image_size"], list) or
                        any(type(n) is not int for n in view["image_size"])):
                    raise RGBContractError("rgb_incomplete_or_mixed_resolution")
                if type(view["size_bytes"]) is not int or not 1 <= view["size_bytes"] <= self._limits.max_file_bytes or not _hash(view["sha256"]):
                    raise RGBContractError("rgb_view_size_or_hash")
                info = self._file(relative).stat()
                if info.st_size != view["size_bytes"]:
                    raise RGBContractError("rgb_file_size_mismatch")

    @property
    def manifest_sha256(self):
        return self._manifest_sha256

    def __len__(self):
        return len(self._records)

    def _index(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise RGBContractError("rgb_index_out_of_range")
        return index

    def metadata(self, index):
        row = self._records[self._index(index)]
        return {"record": copy.deepcopy(row), "manifest_sha256": self.manifest_sha256,
                "data_kind": self._manifest["data_kind"],
                "dataset_version": self._manifest["dataset_version"],
                "sources": {key: copy.deepcopy(self._manifest["sources"][key])
                            for key in {v["source_id"] for v in row["views"].values()}}}

    @staticmethod
    def _dependencies():
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RGBContractError("rgb_optional_dependencies_required: requirements-rgb.txt") from exc
        return torch, Image

    def _decode_into(self, view, destination, torch, Image):
        try:
            raw = _regular_read(self._file(view["path"]), self._limits.max_file_bytes, view["size_bytes"])
            if hashlib.sha256(raw).hexdigest() != view["sha256"]:
                raise RGBContractError("rgb_file_hash_mismatch")
            # Pillow에 넘기기 전에 IHDR의 원래 bit depth·channel·픽셀 수를 확인한다.
            if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\0\0\0\rIHDR":
                raise RGBContractError("rgb_png_header")
            w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", raw[16:29])
            if [w, h] != view["image_size"] or w * h > self._limits.max_pixels_per_view:
                raise RGBContractError("rgb_decoded_size_or_pixel_limit")
            if depth != 8 or color not in (2, 6) or compression or filtering or interlace not in (0, 1):
                raise RGBContractError("rgb_png_requires_rgb_or_rgba8")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw), formats=["PNG"]) as image:
                    if image.size != (w, h) or getattr(image, "n_frames", 1) != 1:
                        raise RGBContractError("rgb_png_size_or_multiframe")
                    image.verify()
                with Image.open(io.BytesIO(raw), formats=["PNG"]) as image:
                    image.load()
                    if image.mode not in ("RGB", "RGBA"):
                        raise RGBContractError("rgb_png_mode")
                    if "transparency" in image.info:
                        raise RGBContractError("rgb_transparent_pixels_not_supported")
                    if image.mode == "RGBA" and image.getextrema()[3] != (255, 255):
                        raise RGBContractError("rgb_transparent_pixels_not_supported")
                    with image.convert("RGB") as rgb:
                        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
                        values = pixels.reshape(h, w, 3).permute(2, 0, 1).to(torch.float32).div_(255.0)
                        destination.copy_(values)
        except RGBContractError:
            raise
        except (OSError, ValueError, EOFError, SyntaxError, struct.error, zlib.error,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise RGBContractError("rgb_png_decode_or_file_failure") from exc

    def _fill(self, index, destination, torch, Image):
        row = self._records[index]
        for position, side in enumerate(("left", "right")):
            self._decode_into(row["views"][side], destination[position], torch, Image)

    def __getitem__(self, index):
        index = self._index(index)
        torch, Image = self._dependencies()
        images = torch.empty((2, 3, self._height, self._width), dtype=torch.float32, device="cpu")
        self._fill(index, images, torch, Image)
        return {"images": images, "metadata": self.metadata(index)}

    def load_batch(self, indices):
        """호출자가 고른 작은 batch를 상한 확인 후 확보한다."""
        if not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= self._limits.max_batch_size:
            raise RGBContractError("rgb_batch_size_limit")
        indices = tuple(self._index(i) for i in indices)
        if len({self._records[i]["split"] for i in indices}) != 1:
            raise RGBContractError("rgb_batch_mixed_split")
        tensor_bytes = len(indices) * 2 * 3 * self._height * self._width * 4
        if tensor_bytes > self._limits.max_batch_tensor_bytes:
            raise RGBContractError("rgb_batch_tensor_byte_limit")
        torch, Image = self._dependencies()
        images = torch.empty((len(indices), 2, 3, self._height, self._width), dtype=torch.float32, device="cpu")
        metadata = []
        for position, index in enumerate(indices):
            self._fill(index, images[position], torch, Image)
            metadata.append(self.metadata(index))
        return {"images": images, "metadata": metadata}

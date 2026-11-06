"""UnrealEgo2 로컬 분할 원본의 전체 검증과 선택 staging. 업로드/삭제/학습 없음.

split ZIP의 sequence는 환경/시간대/인물/안경/번호/motion 6개 요소다.
분할 archive는 공식 registry 순서대로 연결해 읽으며 결합 파일을 만들지 않는다.
선택 밖 payload도 소비하고 gzip EOF/CRC와 tar 종료를 확인한다.
로컬 SHA는 provenance이며 공급자의 공식 archive SHA 검증은 아니다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
from typing import Any
import urllib.request
import zipfile

OFFICIAL_SPLIT_URL = "https://unrealego.mpi-inf.mpg.de/data/data_split/UnrealEgoData2.zip"
CATEGORIES = tuple("UnrealEgoData2_" + s for s in
                   ("train_val_rgb", "train_val_pose", "test_rgb", "test_pose"))
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "configs/fleet_inventory_observed.json"


class SourceScanError(ValueError):
    """실패 시 report 속성과 stage_dir/source_report.json을 참조한다."""

    def __init__(self, reason: str, report: dict | None = None):
        self.reason, self.report = reason, report
        super().__init__(reason)


@dataclass(frozen=True)
class SourceLimits:
    max_compressed_bytes: int = 32 * 1024**3
    max_output_bytes: int = 2 * 1024**3
    max_decompressed_bytes: int = 32 * 1024**3
    max_members: int = 100_000
    max_file_bytes: int = 16 * 1024**2
    max_manifest_bytes: int = 64 * 1024**2
    max_metadata_bytes: int = 1024**2
    max_split_bytes: int = 4 * 1024**2
    max_split_decoded_bytes: int = 16 * 1024**2
    reserve_free_bytes: int = 512 * 1024**2
    chunk_bytes: int = 1024**2
    request_timeout: float = 30.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "request_timeout":
                if not isinstance(value, (int, float)) or not 0 < value <= 300:
                    raise SourceScanError("invalid_request_timeout")
            elif type(value) is not int or value < (0 if name == "reserve_free_bytes" else 1):
                raise SourceScanError("invalid_limit:" + name)
        if not 512 <= self.chunk_bytes <= 8 * 1024**2:
            raise SourceScanError("invalid_chunk_bytes")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")


def _safe(name):
    # 기존 rewind_archive.contracts.safe_path와 같은 portable path 정책.
    if (not isinstance(name, str) or not name or len(name.encode("utf-8")) > 1024
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or "\\" in name or ":" in name):
        raise SourceScanError("unsafe_path")
    path = PurePosixPath(name)
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if (path.is_absolute() or ".." in path.parts or not path.parts
            or any(p.endswith((".", " ")) or reserved.match(p) for p in path.parts)):
        raise SourceScanError("unsafe_path")
    return str(path)


def _pax_fields(data):
    # 기존 제한형 PAX parser의 지원 집합을 standalone으로 유지한다.
    fields, pos = {}, 0
    while pos < len(data):
        space = data.find(b" ", pos, min(len(data), pos + 12))
        if space < 0 or not data[pos:space].isdigit():
            raise SourceScanError("invalid_pax")
        end = pos + int(data[pos:space])
        if end > len(data) or end <= space + 1 or data[end - 1:end] != b"\n":
            raise SourceScanError("invalid_pax")
        entry = data[space + 1:end - 1]
        if b"=" not in entry:
            raise SourceScanError("invalid_pax")
        key, value = (s.decode("utf-8") for s in entry.split(b"=", 1))
        if key in fields or key not in ("path", "size", "mtime", "atime", "ctime",
                                        "uid", "gid", "uname", "gname"):
            raise SourceScanError("unsupported_pax")
        fields[key] = value
        pos = end
    return fields


def _identity(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _load_split(split_zip, limits):
    source = str(split_zip)
    if source.startswith(("http://", "https://")):
        if source != OFFICIAL_SPLIT_URL:
            raise SourceScanError("split_url_not_official")
        with urllib.request.urlopen(source, timeout=limits.request_timeout) as response:
            if response.geturl() != OFFICIAL_SPLIT_URL:
                raise SourceScanError("split_redirect_not_official")
            data = response.read(limits.max_split_bytes + 1)
    else:
        path = Path(split_zip)
        if path.stat().st_size > limits.max_split_bytes:
            raise SourceScanError("split_zip_bytes_limit")
        data = path.read_bytes()
    if len(data) > limits.max_split_bytes:
        raise SourceScanError("split_zip_bytes_limit")
    sequences, hashes, names = {}, {}, set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.infolist()) > 32:
            raise SourceScanError("split_zip_entries_limit")
        decoded = 0
        found = {}
        for info in archive.infolist():
            path = _safe(info.filename)
            if path.casefold() in names:
                raise SourceScanError("split_zip_duplicate_path")
            names.add(path.casefold())
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise SourceScanError("split_zip_symlink")
            decoded += info.file_size
            if decoded > limits.max_split_decoded_bytes:
                raise SourceScanError("split_decoded_bytes_limit")
            stem = Path(path).name
            if stem not in ("train.txt", "validation.txt", "test.txt"):
                continue
            split = stem[:-4]
            if split in found:
                raise SourceScanError("split_zip_duplicate_split")
            payload = archive.read(info)  # ZIP CRC checked by zipfile.
            found[split] = path
            hashes[split] = hashlib.sha256(payload).hexdigest()
            for line in payload.decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                seq = _safe(line.strip().rstrip("/"))
                if len(seq.split("/")) != 6:
                    raise SourceScanError("split_sequence_shape")
                if seq in sequences:
                    raise SourceScanError("split_duplicate_or_overlap")
                sequences[seq] = split
        if set(found) != {"train", "validation", "test"}:
            raise SourceScanError("split_txt_missing")
    return sequences, {"source": source, "sha256": hashlib.sha256(data).hexdigest(),
                       "txt_sha256": hashes, "members": found,
                       "trust": "local_bytes_not_independently_authenticated"}


def _parts(parts_dir, category, environment, registry_path):
    registry_path = Path(registry_path or DEFAULT_REGISTRY)
    data = registry_path.read_bytes()
    registry = json.loads(data)
    matches = [a for a in registry["archives"]
               if a["category"] == category and a["name"] == environment]
    if len(matches) != 1:
        raise SourceScanError("registry_environment_missing_or_duplicate")
    entry = matches[0]
    count = entry["official_part_count"]
    if type(count) is not int or not 1 <= count <= 1000:
        raise SourceScanError("registry_part_count_invalid")
    expected = [f"{environment}.tar.gz{i:03d}" for i in range(count)]
    declared = sorted(entry["parts"], key=lambda part: part["index"])
    if ([p["index"] for p in declared] != list(range(count)) or
            [p["name"] for p in declared] != expected):
        raise SourceScanError("registry_part_order_invalid")
    folder = Path(parts_dir).resolve(strict=True)
    observed = {p.name for p in folder.glob(environment + ".tar.gz*")}
    if observed != set(expected):
        raise SourceScanError("part_set_mismatch")
    paths = [folder / name for name in expected]
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise SourceScanError("part_not_regular_or_empty")
    # Observed Drive byte sizes are not authoritative official archive hashes.
    # Deliberately do not require local sizes to equal that old remote inventory.
    return paths, {"path": str(registry_path), "sha256": hashlib.sha256(data).hexdigest(),
                   "official_count_source": entry.get("official_count_source"),
                   "expected_part_names": expected,
                   "authenticity": "names_and_order_only_no_official_content_digest"}


class _SplitReader:
    def __init__(self, paths, max_compressed_bytes, progress_callback=None,
                 progress_bytes=64 * 1024**2, progress_seconds=5.0):
        self.paths = paths
        self.identities = [_identity(p) for p in paths]
        self.total_compressed_bytes = sum(s[2] for s in self.identities)
        self.max_compressed_bytes = max_compressed_bytes
        if self.total_compressed_bytes > max_compressed_bytes:
            raise SourceScanError("compressed_bytes_limit")
        self.compressed_bytes = 0
        self.hashes = [hashlib.sha256() for _ in paths]
        self.sizes = [0] * len(paths)
        self.index = 0
        self.handle = None
        self.progress_callback = progress_callback
        self.progress_bytes, self.progress_seconds = progress_bytes, progress_seconds
        self.progress_started = self.progress_last_time = time.monotonic()
        self.progress_last_bytes = 0
        self.progress(force=True)

    def progress(self, force=False, phase="reading"):
        if self.progress_callback is None:
            return
        now = time.monotonic()
        if (force or self.compressed_bytes - self.progress_last_bytes >= self.progress_bytes
                or now - self.progress_last_time >= self.progress_seconds):
            self.progress_callback({"phase": phase, "compressed_bytes": self.compressed_bytes,
                                    "total_compressed_bytes": self.total_compressed_bytes,
                                    "elapsed_seconds": now - self.progress_started})
            self.progress_last_bytes, self.progress_last_time = self.compressed_bytes, now

    def read(self, size):
        if size < 0:
            raise SourceScanError("unbounded_compressed_read")
        chunks = []
        while size and self.index < len(self.paths):
            if self.handle is None:
                if _identity(self.paths[self.index]) != self.identities[self.index]:
                    raise SourceScanError("source_changed")
                self.handle = self.paths[self.index].open("rb")
            part_remaining = self.identities[self.index][2] - self.sizes[self.index]
            if not part_remaining:
                if _identity(self.paths[self.index]) != self.identities[self.index]:
                    raise SourceScanError("source_changed")
                self.handle.close()
                self.handle = None
                self.index += 1
                continue
            remaining = self.max_compressed_bytes - self.compressed_bytes
            if remaining <= 0:
                raise SourceScanError("compressed_bytes_limit")
            chunk = self.handle.read(min(size, part_remaining, remaining))
            if chunk:
                self.hashes[self.index].update(chunk)
                self.sizes[self.index] += len(chunk)
                self.compressed_bytes += len(chunk)
                chunks.append(chunk)
                size -= len(chunk)
                self.progress()
            else:
                self.handle.close()
                self.handle = None
                self.index += 1
        return b"".join(chunks)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def verify(self):
        for i, path in enumerate(self.paths):
            if _identity(path) != self.identities[i] or self.sizes[i] != self.identities[i][2]:
                raise SourceScanError("source_changed_or_not_fully_read")

    def report(self):
        return [{"name": p.name, "size": self.identities[i][2], "bytes_read": self.sizes[i],
                 "sha256": self.hashes[i].hexdigest()} for i, p in enumerate(self.paths)]


class _DecodedReader:
    def __init__(self, handle, limits):
        self.handle, self.limits, self.count = handle, limits, 0

    def read(self, size):
        data = self.handle.read(min(size, self.limits.max_decompressed_bytes - self.count + 1))
        self.count += len(data)
        if self.count > self.limits.max_decompressed_bytes:
            raise SourceScanError("decompressed_bytes_limit")
        return data

    def exact(self, size):
        data = self.read(size)
        if len(data) != size:
            raise SourceScanError("incomplete_tar")
        return data


def _member_info(path, kind, sequences):
    parts = path.split("/")
    seq = "/".join(parts[:6]) if len(parts) >= 7 else None
    split = sequences.get(seq)
    view, frame = None, None
    if kind == "pose" and len(parts) == 8 and parts[6] == "json":
        match = re.fullmatch(r"frame_(\d+)\.json", parts[7])
    elif (kind == "rgb" and len(parts) == 9 and parts[6] == "fisheye_final_image"
          and parts[7] in ("camera_left", "camera_right")):
        view = parts[7][7:]
        match = re.fullmatch(r"final_(\d+)\.png", parts[8])
    else:
        match = None
    if match:
        frame = int(match[1])
    return {"sequence": seq, "split": split, "view": view, "frame": frame,
            "matched": split is not None and frame is not None}


def scan_source(parts_dir, category, environment, split_zip, stage_dir, *,
                selected_sequences=None, sequence_prefixes=None, limits=None,
                registry_path=None, progress_callback=None,
                progress_bytes=64 * 1024**2, progress_seconds=5.0,
                **limit_overrides) -> dict[str, Any]:
    """환경 하나를 검증해 stage_dir/<environment>/... 및 manifest.json 생성.

    parts_dir는 해당 환경 .tar.gz000.. 조각이 바로 있는 폴더다.
    선택은 공식 sequence 전체 이름 또는 경로 요소 경계의 prefix다(서로 배타적).
    누락·미매칭은 보고하고 upload_ready=False, 구조/CRC/예산 오류는
    SourceScanError로 중단한다. 부분 결과는 보존하며 새 stage_dir로만 실행한다.
    max_output_bytes는 파일 payload, max_manifest_bytes는 보고서 예약 공간이다.
    max_compressed_bytes는 실행 직전 stat 합과 실제 read 양쪽에 적용한다.
    progress_callback(dict)는 선택 사항이며 기본 출력은 없다. 지정하면 시작·끝과
    progress_bytes 또는 progress_seconds 간격에 읽기 진행을 전달한다. 콜백 예외는
    전체 실행 실패로 처리하므로 KeyboardInterrupt로 안전 중단을 요청할 수 있다.
    RAM은 max_members/경로길이에 비례하며 hard RSS guard는 제공하지 않는다.
    """
    if category not in CATEGORIES or _safe(environment) != environment or "/" in environment:
        raise SourceScanError("invalid_category_or_environment")
    if selected_sequences is not None and sequence_prefixes is not None:
        raise SourceScanError("selection_conflict")
    if (progress_callback is not None and not callable(progress_callback)):
        raise SourceScanError("invalid_progress_callback")
    if (type(progress_bytes) is not int or progress_bytes <= 0
            or type(progress_seconds) not in (int, float)
            or not math.isfinite(progress_seconds) or progress_seconds <= 0):
        raise SourceScanError("invalid_progress_interval")
    if limits is None:
        limits = SourceLimits()
    elif isinstance(limits, dict):
        limits = SourceLimits(**limits)
    if not isinstance(limits, SourceLimits):
        raise SourceScanError("invalid_limits")
    limits = replace(limits, **limit_overrides)
    stage = Path(stage_dir)
    stage.parent.mkdir(parents=True, exist_ok=True)
    if stage.exists() or stage.is_symlink():
        raise SourceScanError("stage_already_exists")
    stage.mkdir()
    stage = stage.resolve()
    report = {"schema": "unrealego2_upload_source_v1", "category": category,
              "environment": environment, "stage_dir": str(stage), "status": "scanning",
              "upload_ready": False, "gzip_crc_valid": False, "tar_complete": False,
              "official_archive_authenticity_verified": False,
              "limits": asdict(limits), "files": [], "output_bytes": 0}
    started = time.monotonic()
    source, reader = None, None
    reserved = limits.max_manifest_bytes
    record_bytes = 0

    def space(payload=0):
        if shutil.disk_usage(stage).free < limits.reserve_free_bytes + reserved + payload:
            raise SourceScanError("staging_free_space_limit")

    def store_report(name):
        payload = _json_bytes(report)
        if len(payload) > limits.max_manifest_bytes:
            raise SourceScanError("manifest_bytes_limit")
        temporary = stage / (name + ".partial")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, stage / name)

    try:
        space()
        sequences, split_provenance = _load_split(split_zip, limits)
        allowed_splits = {"test"} if "_test_" in category else {"train", "validation"}
        expected = {seq for seq, split in sequences.items()
                    if seq.startswith(environment + "/") and split in allowed_splits}
        if not expected:
            raise SourceScanError("no_official_sequences_for_environment")
        if selected_sequences is not None:
            if not isinstance(selected_sequences, (list, tuple)) or not selected_sequences:
                raise SourceScanError("invalid_selected_sequences")
            selected = {_safe(seq.rstrip("/")) for seq in selected_sequences}
            if len(selected) != len(selected_sequences) or not selected <= expected:
                raise SourceScanError("selection_not_official_or_duplicate")
        elif sequence_prefixes is not None:
            if not isinstance(sequence_prefixes, (list, tuple)) or not sequence_prefixes:
                raise SourceScanError("invalid_sequence_prefixes")
            prefixes = [_safe(p.rstrip("/")) for p in sequence_prefixes]
            if any(not any(seq == p or seq.startswith(p + "/") for seq in expected)
                   for p in prefixes):
                raise SourceScanError("prefix_without_official_sequence")
            selected = {seq for seq in expected
                        if any(seq == p or seq.startswith(p + "/") for p in prefixes)}
        else:
            selected = expected.copy()
        paths, registry = _parts(parts_dir, category, environment, registry_path)
        report.update(split_zip=split_provenance, registry=registry,
                      selected_sequences=sorted(selected), expected_sequence_count=len(expected))
        report["total_compressed_bytes"] = sum(path.stat().st_size for path in paths)
        if report["total_compressed_bytes"] > limits.max_compressed_bytes:
            raise SourceScanError("compressed_bytes_limit")
        source = _SplitReader(paths, limits.max_compressed_bytes, progress_callback,
                              progress_bytes, progress_seconds)
        report["total_compressed_bytes"] = source.total_compressed_bytes
        seen, file_keys, directories = set(), set(), {}
        frame_keys, observed, unexpected, unmatched, views = set(), set(), set(), [], {}
        files = report["files"]
        kind = category.rsplit("_", 1)[1]
        headers, global_pax, local_pax, longname = 0, {}, {}, None
        with gzip.GzipFile(fileobj=source, mode="rb") as decoded:
            reader = _DecodedReader(decoded, limits)
            while True:
                block = reader.exact(512)
                if not any(block):
                    if local_pax or longname:
                        raise SourceScanError("orphan_tar_metadata")
                    if any(reader.exact(512)):
                        raise SourceScanError("tar_missing_second_end_block")
                    while True:
                        tail = reader.read(limits.chunk_bytes)
                        if not tail:
                            break
                        if any(tail):
                            raise SourceScanError("nonzero_data_after_tar_end")
                    if reader.count % 512:
                        raise SourceScanError("tar_padding_not_block_aligned")
                    report["tar_complete"] = True
                    report["gzip_crc_valid"] = True
                    break
                headers += 1
                if headers > limits.max_members:
                    raise SourceScanError("members_limit")
                info = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
                size = info.size
                if size < 0:
                    raise SourceScanError("negative_tar_size")
                if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
                    if size > limits.max_metadata_bytes:
                        raise SourceScanError("tar_metadata_bytes_limit")
                    payload = reader.exact(size)
                    if any(reader.exact((-size) % 512)):
                        raise SourceScanError("nonzero_tar_padding")
                    if info.type == tarfile.GNUTYPE_LONGNAME:
                        if longname is not None:
                            raise SourceScanError("duplicate_longname")
                        longname = payload.rstrip(b"\x00").decode("utf-8")
                    else:
                        attrs = _pax_fields(payload)
                        if info.type == tarfile.XGLTYPE:
                            if "path" in attrs or "size" in attrs:
                                raise SourceScanError("unsupported_global_pax_path_or_size")
                            global_pax.update(attrs)
                        else:
                            if local_pax:
                                raise SourceScanError("duplicate_local_pax")
                            local_pax = attrs
                    continue
                attrs = {**global_pax, **local_pax}
                name = attrs.get("path", longname or info.name)
                if "size" in attrs:
                    if not re.fullmatch(r"[0-9]+", attrs["size"]):
                        raise SourceScanError("invalid_pax_size")
                    size = int(attrs["size"])
                local_pax, longname = {}, None
                path = _safe(name)
                # Official streams contain environment/...; tolerate only explicit ./ normalization.
                if path.split("/")[0] != environment:
                    raise SourceScanError("tar_environment_mismatch")
                key = path.casefold()
                if key in seen:
                    raise SourceScanError("duplicate_tar_path")
                components = path.split("/")
                for i in range(1, len(components)):
                    parent = "/".join(components[:i])
                    parent_key = parent.casefold()
                    if parent_key in file_keys:
                        raise SourceScanError("tar_file_directory_collision")
                    if parent_key in directories and directories[parent_key] != parent:
                        raise SourceScanError("tar_case_alias")
                    directories[parent_key] = parent
                seen.add(key)
                if info.isdir():
                    if size:
                        raise SourceScanError("nonempty_tar_directory")
                    if key in directories and directories[key] != path:
                        raise SourceScanError("tar_case_alias")
                    directories[key] = path
                    continue
                if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or info.issparse():
                    raise SourceScanError("unsupported_tar_member_type")
                if key in directories:
                    raise SourceScanError("tar_file_directory_collision")
                file_keys.add(key)
                if size > limits.max_file_bytes:
                    raise SourceScanError("file_bytes_limit")
                meta = _member_info(path, kind, sequences)
                seq = meta["sequence"]
                if meta["matched"]:
                    frame_key = (seq, meta["view"], meta["frame"])
                    if frame_key in frame_keys:
                        raise SourceScanError("duplicate_sequence_view_frame")
                    frame_keys.add(frame_key)
                if seq in expected and meta["matched"]:
                    observed.add(seq)
                elif seq and seq not in expected:
                    unexpected.add(seq)
                if not meta["matched"] or meta["split"] not in allowed_splits:
                    unmatched.append(path)
                # All actual archive paths are audited, including filtered-out files.
                record_bytes += len(path.encode("utf-8")) + 512
                if record_bytes > limits.max_manifest_bytes // 2:
                    raise SourceScanError("manifest_record_budget_limit")
                chosen = seq in selected
                if chosen and report["output_bytes"] + size > limits.max_output_bytes:
                    raise SourceScanError("output_bytes_limit")
                handle = None
                if chosen:
                    space(size)
                    destination = stage / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    handle = destination.open("xb")
                digest, remaining = hashlib.sha256(), size
                try:
                    while remaining:
                        chunk = reader.exact(min(remaining, limits.chunk_bytes))
                        if chosen:
                            space(len(chunk))
                            handle.write(chunk)
                            digest.update(chunk)
                        remaining -= len(chunk)
                finally:
                    if handle:
                        handle.close()
                if any(reader.exact((-size) % 512)):
                    raise SourceScanError("nonzero_tar_padding")
                if chosen:
                    files.append({"relative_path": path, "size": size,
                                  "sha256": digest.hexdigest(), **meta})
                    report["output_bytes"] += size
                    if meta["matched"] and kind == "rgb":
                        views.setdefault((seq, meta["frame"]), set()).add(meta["view"])
        source.verify()
        source.progress(force=True, phase="read_complete")
        report.update(parts=source.report(), decompressed_bytes=reader.count,
                      compressed_bytes=source.compressed_bytes,
                      tar_headers=headers, observed_sequence_count=len(observed),
                      missing_official_sequences=sorted(expected - observed),
                      missing_selected_sequences=sorted(selected - observed),
                      unexpected_sequences=sorted(unexpected), unmatched_files=unmatched,
                      missing_stereo_views=[{"sequence": s, "frame": f,
                                            "missing": sorted({"left", "right"} - v)}
                                           for (s, f), v in sorted(views.items()) if len(v) != 2],
                      status="complete", elapsed_seconds=time.monotonic() - started)
        report["upload_ready"] = bool(files) and not any(report[k] for k in
            ("missing_official_sequences", "missing_selected_sequences", "unexpected_sequences",
             "unmatched_files", "missing_stereo_views"))
        store_report("manifest.json")
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", upload_ready=False,
                      failure=exc.reason if isinstance(exc, SourceScanError) else type(exc).__name__,
                      elapsed_seconds=time.monotonic() - started)
        if source:
            report["parts"] = source.report()
            report["compressed_bytes"] = source.compressed_bytes
        if reader:
            report["decompressed_bytes"] = reader.count
        # If disk is exhausted retain original exception and in-memory report.
        try:
            store_report("source_report.json")
        except (OSError, SourceScanError):
            pass
        raise SourceScanError(report["failure"], report) from exc
    finally:
        if source:
            source.close()

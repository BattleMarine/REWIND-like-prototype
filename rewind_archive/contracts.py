"""자격 증명 없는 archive 입력 계약과 작업 예산."""

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit


class PreparationError(ValueError):
    def __init__(self, reason, diagnostic=None):
        self.reason = reason
        self.diagnostic = diagnostic
        super().__init__(reason)


def canonical(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(doc):
    return hashlib.sha256(canonical(doc)).hexdigest()


def validate_includes(includes):
    """인증·metadata 전에 공유하는 선택 preflight. 중복도 입력 오류다."""
    if not isinstance(includes,list) or not 1 <= len(includes) <= 32:
        raise PreparationError("explicit_selection_required")
    for pattern in includes:
        if not isinstance(pattern,str) or not pattern or len(pattern.encode("utf-8"))>1024:
            raise PreparationError("explicit_selection_required")
        safe_path(pattern)
    if len(set(includes)) != len(includes):
        raise PreparationError("duplicate_selection")
    return list(includes)


def validate_exact_targets(targets, includes, limits, mode):
    if targets is None:
        return []
    if not isinstance(targets,list) or len(targets)>32:
        raise PreparationError("exact_targets_list")
    if not targets:
        return []
    if mode!="extract" or len(targets)>limits.max_files:
        raise PreparationError("exact_targets_mode_or_file_limit")
    for target in targets:
        if not isinstance(target,str) or any(c in target for c in "*?[]") or safe_path(target,limits.max_path_bytes)!=target:
            raise PreparationError("exact_targets_canonical_literal_path")
    if len({t.casefold() for t in targets})!=len(targets):
        raise PreparationError("exact_targets_duplicate")
    validate_includes(includes)
    if set(includes)!=set(targets):
        raise PreparationError("selection_exact_targets_mismatch")
    return list(targets)


def target_summary(targets, files, usable):
    stored_names={entry["member"] for entry in files}
    stored=[t for t in targets if t in stored_names]
    completed=stored if usable else []
    missing=[t for t in targets if t not in completed]
    complete=bool(targets) and not missing
    return {"exact_targets":list(targets),"stored_targets":stored,"completed_targets":completed,
        "missing_targets":missing,"targets_complete":complete,
        "targets_status":"not_requested" if not targets else "unusable" if not usable else "complete" if complete else "incomplete"}


def safe_path(name, max_bytes=1024):
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > max_bytes:
        raise PreparationError("invalid_or_long_path")
    if any(ord(c) < 32 or ord(c) == 127 for c in name) or "\\" in name or ":" in name:
        raise PreparationError("unsafe_path")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise PreparationError("unsafe_path")
    parts = [part for part in p.parts if part != "."]
    # Windows에서도 reserved name/ADS/끝의 dot·공백을 만들지 않는다.
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if not parts or any(part.endswith((".", " ")) or reserved.match(part) for part in parts):
        raise PreparationError("unsafe_path")
    return "/".join(parts)


def validate_manifest(doc):
    if not isinstance(doc, dict) or set(doc) != {"schema_version", "archive_id", "dataset_version", "kind", "source_scope", "total_parts", "parts"}:
        raise PreparationError("manifest_fields")
    if doc["schema_version"] != "archive_manifest_v1" or doc["kind"] not in ("rgb", "pose"):
        raise PreparationError("manifest_schema_kind")
    if doc["source_scope"] not in ("synthetic_fixture", "official_server", "official_prefix_local", "user_drive", "local_unverified"):
        raise PreparationError("source_scope")
    for key in ("archive_id", "dataset_version"):
        if not isinstance(doc[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", doc[key]):
            raise PreparationError("manifest_identity")
    parts = doc["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 1000 or type(doc["total_parts"]) is not int or not len(parts) <= doc["total_parts"] <= 1000:
        raise PreparationError("manifest_part_count")
    names = set()
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != {"index", "name", "size_bytes", "version", "source", "identity"}:
            raise PreparationError("manifest_part_fields")
        if type(part["index"]) is not int or part["index"] != index:
            raise PreparationError("noncontiguous_prefix")
        name = safe_path(part["name"])
        if "/" in name or not name.endswith(f"{index:03d}") or name in names:
            raise PreparationError("part_name_index")
        names.add(name)
        if type(part["size_bytes"]) is not int or part["size_bytes"] <= 0:
            raise PreparationError("part_size")
        if not isinstance(part["version"], str) or not 1 <= len(part["version"]) <= 128 or any(ord(c)<32 for c in part["version"]):
            raise PreparationError("part_version")
        source, identity = part["source"], part["identity"]
        if not isinstance(source, dict) or not isinstance(identity, dict):
            raise PreparationError("source_identity")
        kind = source.get("type")
        if kind == "local":
            if set(source) != {"type", "path"} or not isinstance(source["path"], str) or not source["path"]:
                raise PreparationError("local_source")
            if set(identity) != {"mtime_ns"} or type(identity["mtime_ns"]) is not int or identity["mtime_ns"] < 0:
                raise PreparationError("local_identity")
        elif kind == "http":
            if set(source) != {"type", "url"} or not isinstance(source["url"], str):
                raise PreparationError("http_source")
            u = urlsplit(source["url"])
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment or any(ord(c)<32 for c in source["url"]):
                raise PreparationError("credential_or_invalid_url")
            if not identity or set(identity) - {"etag", "last_modified"}:
                raise PreparationError("http_identity_required")
            for key, value in identity.items():
                if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(ord(c)<32 for c in value):
                    raise PreparationError("http_validator")
                if key == "etag" and (value.startswith("W/") or not re.fullmatch(r'"[^"\r\n]+"', value)):
                    raise PreparationError("strong_etag_required")
        elif kind == "drive":
            if set(source) != {"type", "file_id"} or not isinstance(source["file_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", source["file_id"]):
                raise PreparationError("drive_source")
            if set(identity) != {"version", "modified_time"} or any(not isinstance(v, str) or not 1 <= len(v) <= 128 or any(ord(c)<32 for c in v) for v in identity.values()):
                raise PreparationError("drive_identity_required")
        else:
            raise PreparationError("unsupported_source")
    if len(canonical(doc)) > 128*1024:
        raise PreparationError("manifest_bytes_limit")
    return doc


@dataclass(frozen=True)
class Limits:
    max_transfer_bytes: int = 8*1024**2
    max_elapsed_seconds: float = 60.0
    request_timeout_seconds: float = 5.0
    io_chunk_bytes: int = 64*1024
    media_range_bytes: int = 256*1024
    max_decompressed_bytes: int = 32*1024**2
    max_file_bytes: int = 4*1024**2
    max_disk_bytes: int = 16*1024**2
    max_files: int = 32
    max_members: int = 1000
    max_metadata_bytes: int = 64*1024
    max_path_bytes: int = 1024
    max_ranges: int = 128

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("seconds"):
                if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                    raise PreparationError("invalid_limits")
            elif type(value) is not int or value < 1:
                raise PreparationError("invalid_limits")
        if not 512 <= self.io_chunk_bytes <= 1024**2 or not 512 <= self.media_range_bytes <= 1024**2 or self.max_path_bytes > 4096 or self.max_members > 100000 or self.max_files > 10000 or self.max_ranges > 10000:
            raise PreparationError("unsafe_limits")
        if self.request_timeout_seconds > min(30, self.max_elapsed_seconds):
            raise PreparationError("request_timeout_limit")

    @classmethod
    def from_dict(cls, doc):
        fields = set(asdict(cls()))
        if not isinstance(doc, dict) or set(doc) not in (fields, fields-{"media_range_bytes"}):
            raise PreparationError("limits_fields_required")
        return cls(**doc)


class Budget:
    def __init__(self, limits, clock=time.monotonic):
        self.limits, self.clock, self.started = limits, clock, clock()
        self.transfer_bytes = self.decompressed_bytes = self.payload_disk_bytes = 0
        self.members = self.metadata_bytes = self.files = self.ranges = 0
        self.media_requests = self.metadata_requests = self.local_reads = 0
        self.peak_output_chunk_bytes = self.peak_input_chunk_bytes = 0
        self.peak_payload_disk_bytes = 0

    def check_time(self):
        remaining = self.limits.max_elapsed_seconds - (self.clock()-self.started)
        if remaining <= 0:
            raise PreparationError("elapsed_limit")
        return remaining

    def timeout(self):
        return min(self.limits.request_timeout_seconds, self.check_time())

    def add(self, field, amount, maximum):
        self.check_time()
        if getattr(self,field) + amount > maximum:
            raise PreparationError(field+"_limit")
        setattr(self, field, getattr(self,field)+amount)

    def stats(self):
        return {key:value for key,value in vars(self).items() if key not in ("limits", "clock", "started")} | {"elapsed_seconds":self.clock()-self.started,
            "http_requests_total":self.media_requests+self.metadata_requests}

    def request(self, kind, maximum=None):
        self.add("ranges", 1, self.limits.max_ranges if maximum is None else maximum)
        field = {"media":"media_requests", "metadata":"metadata_requests", "local":"local_reads"}[kind]
        setattr(self, field, getattr(self, field)+1)

    def reserve_requests(self, count):
        self.check_time()
        if self.ranges+count > self.limits.max_ranges:
            raise PreparationError("ranges_limit")

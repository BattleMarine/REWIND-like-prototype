"""Portable v2 128/A bundle layout and bounded member I/O (standard library only)."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


MIB = 1024**2
GIB = 1024**3
A = "reports/v2_gpu_training_128_v1/A-seed-42"
BULK = "reports/cache/v2_bulk_cache_041837Z"
LEGACY = "reports/cache/v2_cpu_cache_041837Z"
DIAGNOSTIC = f"{A}/diagnostics/invocation-roh3t3jf.json"
STEP = f"{A}/checkpoints/step-000009.json"
COMPARISON = f"{A}/comparison.json"
FEATURE_IDENTITY = "9588afef53e4802089baf7c51235f18068d110d02b1cac8ca6371b8d7aa5e276"
FEATURES = f"{A}/features/{FEATURE_IDENTITY}"
SOURCE = "source/v2_gpu_training_128_v1.zip"
SOURCE_RECEIPT = "source/receipt.json"
INDEX = "BUNDLE_INDEX.json"
SCHEMA = "rewind_v2_gpu128_a_portable_v1"
SOURCE_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
SOURCE_RECEIPT_SHA = "7a699521ec937b4481ec544f9d1e01fc7a7f7abb58d20442afab86c85e73d43d"
DIAGNOSTIC_SHA = "2d6e9026e6f12df945fed105722ca668db799c843fb6100ab52c3984abb8f08c"
COMPARISON_SHA = "0ea729ef9a8c14b98b81c704669cd910f2e741acb2371c8789007c0fda6174d4"
FINAL_RECEIPT = "7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4"
FINAL_BOUNDARY = "dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511"
TRAINER_REVISION = "b7a2d00f3bc755808a451ed4ead873e6513d094b"
GIT_URL = "https://github.com/BattleMarine/REWIND-like-prototype"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def name_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name or any(c in name for c in "\\:\x00"):
        raise ValueError(f"bundle_unsafe_member: {name}")
    return path


def check_json_pin(doc, schema, expected=None):
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    inferred = sha(canonical(body))
    if doc.get("schema") != schema or doc.get("content_sha256") != inferred or expected is not None and inferred != expected:
        raise ValueError(f"bundle_json_pin: {schema}")


def hash_stream(source, pin, check=lambda: None, destination=None):
    hasher, count = hashlib.sha256(), 0
    while block := source.read(2*MIB):
        check()
        count += len(block)
        if count > pin["bytes"]:
            raise ValueError("bundle_member_growth")
        hasher.update(block)
        if destination is not None and destination.write(block) != len(block):
            raise OSError("bundle_short_write")
    if count != pin["bytes"] or hasher.hexdigest() != pin["sha256"]:
        raise ValueError("bundle_member_sha256_or_size")
    return count


class BundleView:
    def __init__(self, path):
        if path.is_symlink():
            raise ValueError("bundle_input_link")
        self.path = path.resolve(strict=True)
        self.archive = None if self.path.is_dir() else zipfile.ZipFile(self.path)
        self.members = {}
        try:
            if self.archive:
                for info in self.archive.infolist():
                    if info.is_dir():
                        continue
                    name = name_path(info.filename).as_posix()
                    mode = stat.S_IFMT(info.external_attr >> 16)
                    if name in self.members or mode not in (0, stat.S_IFREG) or info.flag_bits & 1:
                        raise ValueError(f"bundle_invalid_member: {name}")
                    self.members[name] = info
                if len(self.members) > 4000:
                    raise ValueError("bundle_member_count")
        except BaseException:
            self.close()
            raise

    def file(self, name):
        path = self.path.joinpath(*name_path(name).parts)
        if not path.is_file() or path.is_symlink() or path.resolve() != path:
            raise ValueError(f"bundle_missing_file: {name}")
        return path

    @contextmanager
    def open(self, name, size=None):
        name_path(name)
        if self.archive:
            info = self.members.get(name)
            if info is None or size is not None and info.file_size != size:
                raise ValueError(f"bundle_member_size: {name}")
            with self.archive.open(info) as stream:
                yield stream
        else:
            path = self.file(name)
            with path.open("rb") as stream:
                if size is not None and path.stat().st_size != size:
                    raise ValueError(f"bundle_member_size: {name}")
                yield stream

    def read(self, name, cap, pin=None):
        with self.open(name, pin["bytes"] if pin else None) as stream:
            data = stream.read(cap+1)
        if not 0 < len(data) <= cap or pin and (len(data) != pin["bytes"] or sha(data) != pin["sha256"]):
            raise ValueError(f"bundle_member_pin: {name}")
        return data

    def copy(self, name, path, pin, check=lambda: None):
        created = False
        try:
            with self.open(name, pin["bytes"]) as stream, path.open("xb") as output:
                created = True
                hash_stream(stream, pin, check, output)
        except BaseException:
            if created:
                path.unlink()
            raise
        return path

    def index(self):
        doc = json.loads(self.read(INDEX, 2*MIB))
        check_json_pin(doc, SCHEMA)
        pins = doc["members"]
        if not 1 <= len(pins) <= 4000 or INDEX in pins:
            raise ValueError("bundle_index_member_count")
        for name, pin in pins.items():
            name_path(name)
            if type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= GIB or not re.fullmatch("[a-f0-9]{64}", pin["sha256"]):
                raise ValueError(f"bundle_index_pin: {name}")
        if self.archive and self.members.keys() != (pins.keys() | {INDEX}):
            raise ValueError("bundle_index_listing")
        return doc

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def metadata(view, pins):
    def read(name, cap, expected=None):
        if expected is not None and pins[name]["sha256"] != expected:
            raise ValueError(f"bundle_fixed_evidence_pin: {name}")
        return json.loads(view.read(name, cap, pins[name]))

    source = read(SOURCE_RECEIPT, 64*1024, SOURCE_RECEIPT_SHA)["pins"]
    final = read(DIAGNOSTIC, 192*1024, DIAGNOSTIC_SHA)
    step = read(STEP, 64*1024)
    comparison = read(COMPARISON, 64*1024, COMPARISON_SHA)
    check_json_pin(final, "rewind_v2_gpu_invocation_v1")
    check_json_pin(step, "rewind_v2_gpu_training_checkpoint_v1", FINAL_RECEIPT)
    check_json_pin(comparison, "rewind_v2_gpu_comparison_v1")
    identity = final["identity"]
    if (source["source_sha256"] != SOURCE_SHA or pins[SOURCE]["sha256"] != SOURCE_SHA
            or final["status"] != "completed" or step != final["checkpoint"] or comparison != final["comparison"]
            or identity != step["identity"] or identity != comparison["binding"]["experiment"]
            or identity["source"] != SOURCE_SHA or identity["cache_boundary"] != FINAL_BOUNDARY
            or not step["completed"] or step["cursor"] != {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}):
        raise ValueError("bundle_final_128_a_identity")
    for name, item in ((f"{A}/checkpoints", step["archive"]), (A, comparison["archive"])):
        if name_path(item["name"]).name != item["name"]:
            raise ValueError("bundle_final_archive_name")
        if pins[f"{name}/{item['name']}"] != {key: item[key] for key in ("bytes", "sha256")}:
            raise ValueError("bundle_final_archive_pin")
    return source, final, step, comparison


def feature_groups(view, pins=None):
    names = [f"{split}-{i:05d}" for split, count in (("train", 3097), ("validation", 421)) for i in range(count)]
    groups = {}
    for start in range(0, len(names), 24):
        tag = f"group-{start//24:05d}"
        name = f"{FEATURES}/{tag}.json"
        raw = view.read(name, 64*1024, pins[name] if pins is not None else None)
        record = json.loads(raw)
        check_json_pin(record, "rewind_v2_gpu_features_group_v1")
        archive = record["archive"]
        if (record["identity"] != FEATURE_IDENTITY or record["folders"] != names[start:start+24]
                or not re.fullmatch(rf"{tag}-[0-9a-f]{{32}}\.zip", archive["name"])
                or not 0 < archive["bytes"] <= 32*MIB or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])):
            raise ValueError("bundle_frozen_feature_group_scope")
        archive_name = f"{FEATURES}/{archive['name']}"
        archive_pin = {key: archive[key] for key in ("bytes", "sha256")}
        if pins is not None and pins[archive_name] != archive_pin:
            raise ValueError("bundle_frozen_feature_group_pin")
        groups[name] = (raw, archive_name, archive_pin, record["folders"])
    return groups

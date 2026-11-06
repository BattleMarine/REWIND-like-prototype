"""Drive API 기반 추출본 감사와 명시적 복구. 원본 삭제/덮어쓰기를 하지 않는다.

stage_dir/environment/... 를 root_id/category/environment/... 와 비교한다.
bundle 모드는 외부에서 ZIP으로 읽는 별도 산출물이며 loose 파일 복구가 아니다.
Drive API는 compare-and-swap을 지원하지 않으므로 사용자 업로드를 멈춘 상태에서
실행해야 한다. 관측 가능한 동시 변경은 중단하지만 API 호출 사이 경쟁을 잠글 수 없다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
CATEGORIES = frozenset(f"UnrealEgoData2_{split}_{kind}" for split in ("train_val", "test") for kind in ("rgb", "pose"))
FIELDS = "id,name,mimeType,size,md5Checksum,modifiedTime,version,parents,appProperties,trashed"


class DriveSafetyError(RuntimeError):
    """불명확한 대상, 동시 변경 또는 예산 초과로 안전 중단."""


class _Budget:
    def __init__(self, requests, seconds, writes=0, upload_bytes=0):
        for label, value in (("requests", requests), ("seconds", seconds), ("writes", writes), ("upload_bytes", upload_bytes)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise DriveSafetyError(f"invalid budget: {label}")
            if label != "seconds" and int(value) != value:
                raise DriveSafetyError(f"non-integral budget: {label}")
        self.max_requests, self.max_seconds = requests, seconds
        self.max_writes, self.max_bytes = writes, upload_bytes
        self.requests = self.writes = self.upload_bytes = 0
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started > self.max_seconds:
            raise DriveSafetyError("elapsed time limit")

    def call(self, request, *, write=False, size=0):
        self.check()
        if self.requests >= self.max_requests or self.writes + int(write) > self.max_writes or self.upload_bytes + size > self.max_bytes:
            raise DriveSafetyError("request/write/upload byte limit")
        self.requests += 1
        self.writes += int(write)
        self.upload_bytes += size  # 실패한 업로드도 예약량으로 계산한다.
        result = request.execute()
        self.check()
        return result

    def stats(self):
        return {"requests": self.requests, "writes": self.writes, "upload_bytes_reserved": self.upload_bytes,
                "elapsed_seconds": time.monotonic() - self.started}

    def upload(self, request, size):
        if not getattr(request, "resumable", None):
            return self.call(request, write=True, size=size)
        self.check()
        if self.writes + 1 > self.max_writes or self.upload_bytes + size > self.max_bytes:
            raise DriveSafetyError("write/upload byte limit")
        self.writes += 1
        self.upload_bytes += size
        response, first = None, True
        while response is None:
            self.check()
            # First next_chunk performs session initiation plus one media request.
            cost = 2 if first else 1
            if self.requests + cost > self.max_requests:
                raise DriveSafetyError("request limit during resumable upload")
            self.requests += cost
            _, response = request.next_chunk(num_retries=0)
            first = False
        self.check()
        return response


def _digest(path):
    md5 = hashlib.md5()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise DriveSafetyError(f"local file changed: {path}")
    return before.st_size, md5.hexdigest()


def _name(name):
    if not isinstance(name, str) or name in ("", ".", "..") or any(c in name for c in "/\\\x00"):
        raise DriveSafetyError(f"unsafe path component: {name!r}")
    return name


def _children(service, parent, budget, names=None):
    result, token, seen = [], None, set()
    def escape(value):
        return value.replace("\\", "\\\\").replace("'", "\\'")
    query = f"'{escape(parent)}' in parents and trashed = false"
    if names:
        query += " and (" + " or ".join(f"name = '{escape(name)}'" for name in names) + ")"
    while True:
        page = budget.call(service.files().list(q=query,
            fields=f"nextPageToken,incompleteSearch,files({FIELDS})", pageSize=1000, pageToken=token,
            supportsAllDrives=True, includeItemsFromAllDrives=True))
        if page.get("incompleteSearch"):
            raise DriveSafetyError("Drive listing incompleteSearch; no safe completeness claim")
        result.extend(page.get("files", []))
        token = page.get("nextPageToken")
        if not token:
            return result
        if token in seen:
            raise DriveSafetyError("repeated pagination token")
        seen.add(token)


def _get(service, file_id, budget):
    return budget.call(service.files().get(fileId=file_id, fields=FIELDS, supportsAllDrives=True))


def _signature(item):
    return json.dumps(item, sort_keys=True, separators=(",", ":"))


def _scan_local(stage_dir, environment, max_files, budget):
    base = Path(stage_dir).resolve()
    start = base / environment
    if not start.is_dir() or start.is_symlink():
        raise DriveSafetyError("staged environment directory missing or symlink")
    expected = []
    for directory, dirs, files in os.walk(start, followlinks=False):
        budget.check()
        for name in dirs + files:
            path = Path(directory) / _name(name)
            if path.is_symlink() or not path.resolve().is_relative_to(base):
                raise DriveSafetyError("symlink/path escape in staged tree")
        for name in sorted(files):
            if len(expected) >= max_files:
                raise DriveSafetyError("local file count limit")
            path = Path(directory) / name
            size, md5 = _digest(path)
            expected.append({"path": path.relative_to(start).as_posix(), "local_path": str(path), "size": size, "md5": md5})
    if not expected:
        raise DriveSafetyError("empty staged tree")
    return sorted(expected, key=lambda e: e["path"])


def _audit(service, root_id, category, environment, stage_dir, max_files, budget):
    if category not in CATEGORIES:
        raise DriveSafetyError("category not allowlisted (depth is excluded)")
    _name(environment)
    root = _get(service, root_id, budget)
    if root.get("mimeType") != FOLDER or root.get("trashed"):
        raise DriveSafetyError("destination root is not a live folder")
    expected = _scan_local(stage_dir, environment, max_files, budget)
    report = {"schema": 1, "root_id": root_id, "category": category, "environment": environment,
              "stage_dir": str(Path(stage_dir).resolve()), "max_files": max_files, "expected": expected,
              "verified": [], "missing": [], "corrupt": [], "pending": [], "extras": [], "conflicts": [],
              "unscanned_folders": [], "scope": "staged expected parent folders only",
              "folders": {"": root_id}, "remote_snapshot": {"@root": _signature(root)}}
    remote = {}

    required_dirs = {category, f"{category}/{environment}"}
    for expected_item in expected:
        parts = f"{category}/{environment}/{expected_item['path']}".split("/")
        required_dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))

    def visit(parent, prefix, only=None):
        children = _children(service, parent, budget, names=[only] if only else None)
        names = {}
        for item in children:
            if only is not None and item["name"] != only:
                continue
            name = _name(item["name"])
            names.setdefault(name, []).append(item)
        for name, items in names.items():
            path = f"{prefix}/{name}" if prefix else name
            for item in items:
                report["remote_snapshot"][item["id"]] = _signature(item)
            if len(items) != 1 or items[0].get("mimeType") == SHORTCUT:
                report["conflicts"].append({"path": path, "ids": [x["id"] for x in items], "reason": "duplicate name or shortcut"})
                continue
            item = items[0]
            if item.get("mimeType") == FOLDER:
                report["folders"][path] = item["id"]
                if path in required_dirs:
                    visit(item["id"], path, environment if path == category else None)
                else:
                    report["unscanned_folders"].append(path)
            else:
                remote[path] = item

    visit(root_id, "", category)
    prefix = f"{category}/{environment}/"
    expected_paths = set()
    for path in sorted(required_dirs.intersection(remote)):
        report["conflicts"].append({"path": path, "reason": "file at expected folder path"})
    blocked_paths = {conflict["path"] for conflict in report["conflicts"]}
    for item in expected:
        path = prefix + item["path"]
        expected_paths.add(path)
        existing = remote.get(path)
        parts = path.split("/")
        if any("/".join(parts[:i]) in blocked_paths for i in range(1, len(parts) + 1)):
            report["pending"].append({"path": item["path"], "reason": "ambiguous destination path"})
        elif path in report["folders"]:
            report["conflicts"].append({"path": path, "reason": "folder at expected file path"})
        elif existing is None:
            report["missing"].append(item["path"])
        elif _matches(existing, item):
            report["verified"].append(item["path"])
        elif not existing.get("md5Checksum") or existing.get("size") is None:
            report["pending"].append({"path": item["path"], "remote": existing, "reason": "size/checksum unavailable; upload may still be running"})
        else:
            report["corrupt"].append({"path": item["path"], "remote": existing})
    report["extras"] = [{"path": p, "remote": v} for p, v in remote.items() if p not in expected_paths]
    report["stats"] = budget.stats()
    return report


def _matches(remote, expected):
    return (remote.get("mimeType") not in (FOLDER, SHORTCUT) and not remote.get("trashed")
            and str(remote.get("size")) == str(expected["size"]) and remote.get("md5Checksum") == expected["md5"])


def audit(service, root_id, category, environment, stage_dir, *, max_files=1_000_000,
          max_requests=100_000, max_seconds=3600):
    """읽기 전용 감사. MD5/크기 미제공 파일은 판정 보류(pending)한다."""
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
        raise DriveSafetyError("invalid max_files")
    return _audit(service, root_id, category, environment, stage_dir, max_files, _Budget(max_requests, max_seconds))


def _default_media(path):
    from googleapiclient.http import MediaFileUpload
    # Small PNG/JSON use one multipart create; large files retain bounded chunk RAM.
    return MediaFileUpload(str(path), chunksize=8 * 1024**2, resumable=Path(path).stat().st_size > 8 * 1024**2)


def repair(service, audit_report, *, enabled=False, mode="loose", max_writes=1000,
           max_upload_bytes=1024**3, max_requests=100_000, max_seconds=3600, media_factory=None,
           max_staging_bytes=1024**3, staging_dir=None):
    """명시 opt-in 복구. 호출 전 사용자의 Drive 업로드를 정지해야 한다.

    max_writes는 폴더 생성·업로드·이름 변경을 모두 포함한다. 네트워크 timeout은
    전달된 service transport에서 설정해야 하며 시간 상한은 API 호출 사이 협력 검사다.
    예외 시 이미 검증한 pending 파일/보존본은 남겨 다음 audit+repair로 회수한다.
    """
    if not enabled:
        raise DriveSafetyError("repair requires enabled=True")
    if mode not in ("loose", "bundle"):
        raise DriveSafetyError("unknown mode")
    budget = _Budget(max_requests, max_seconds, max_writes, max_upload_bytes)
    if isinstance(max_staging_bytes, bool) or not isinstance(max_staging_bytes, int) or max_staging_bytes < 0:
        raise DriveSafetyError("invalid max_staging_bytes")
    media_factory = media_factory or _default_media
    current = _audit(service, audit_report["root_id"], audit_report["category"], audit_report["environment"],
                     audit_report["stage_dir"], audit_report["max_files"], budget)
    if current["pending"] or current["conflicts"] or current["remote_snapshot"] != audit_report["remote_snapshot"] or current["expected"] != audit_report["expected"]:
        raise DriveSafetyError("conflict or destination/source changed since audit; audit again after uploads stop")
    folders = dict(current["folders"])
    operations = []
    conditional_renames = []

    def rename(item, name):
        response_headers = {}
        request = service.files().get(fileId=item["id"], fields=FIELDS, supportsAllDrives=True)
        if hasattr(request, "add_response_callback"):
            request.add_response_callback(lambda response: response_headers.update(dict(response)))
        fresh = budget.call(request)
        if _signature(fresh) != _signature(item):
            raise DriveSafetyError("file changed immediately before rename")
        update_request = service.files().update(fileId=item["id"], body={"name": name}, fields=FIELDS, supportsAllDrives=True)
        etag = response_headers.get("etag") or response_headers.get("ETag")
        if etag and hasattr(update_request, "headers"):
            update_request.headers["If-Match"] = etag
            conditional_renames.append(True)
        else:
            conditional_renames.append(False)
        return budget.call(update_request, write=True)

    def folder(path):
        if path in folders:
            return folders[path]
        parent_path, _, name = path.rpartition("/")
        parent = folder(parent_path)
        children = _children(service, parent, budget, names=[name])
        if any(i["name"] == name for i in children):
            raise DriveSafetyError("concurrent folder creation")
        item = budget.call(service.files().create(body={"name": name, "mimeType": FOLDER, "parents": [parent]},
                    fields=FIELDS, supportsAllDrives=True), write=True)
        if len(_children(service, parent, budget, names=[name])) != 1:
            raise DriveSafetyError("concurrent duplicate folder creation")
        folders[path] = item["id"]
        return item["id"]

    def upload(path, expected, destination, original=None):
        parent_path, _, name = destination.rpartition("/")
        parent = folder(parent_path)
        key = hashlib.sha256((destination + ":" + expected["md5"]).encode()).hexdigest()
        pending_name = ".upload-" + key
        watched = [name, pending_name]
        if original:
            watched.append(name + ".WRONG_UPLOAD-" + original["id"])
        before = _children(service, parent, budget, names=watched)
        canonical = [i for i in before if i["name"] == name]
        if original:
            if len(canonical) != 1 or _signature(canonical[0]) != _signature(original):
                raise DriveSafetyError("original changed before repair")
        elif canonical:
            if len(canonical) == 1 and _matches(canonical[0], expected):
                return "already_verified"
            raise DriveSafetyError("canonical destination appeared")
        pending = [i for i in before if i["name"] == pending_name]
        if len(pending) > 1:
            raise DriveSafetyError("ambiguous pending uploads")
        if pending:
            staged = pending[0]
            if staged.get("appProperties", {}).get("rewindRepair") != key or not _matches(staged, expected):
                raise DriveSafetyError("unverified pending upload; preserve and inspect")
        else:
            if _digest(Path(path)) != (expected["size"], expected["md5"]):
                raise DriveSafetyError("local source changed before upload")
            staged = budget.upload(service.files().create(body={"name": pending_name, "parents": [parent],
                "appProperties": {"rewindRepair": key}}, media_body=media_factory(path), fields=FIELDS,
                supportsAllDrives=True), expected["size"])
        staged = _get(service, staged["id"], budget)
        if not _matches(staged, expected):
            raise DriveSafetyError("Drive uploaded size/MD5 verification failed; pending retained")
        after = _children(service, parent, budget, names=watched)
        expected_before = {i["id"]: _signature(i) for i in before if i["name"] != pending_name}
        actual_after = {i["id"]: _signature(i) for i in after if i["id"] != staged["id"]}
        if actual_after != expected_before:
            raise DriveSafetyError("destination changed during upload; pending retained")
        if original:
            preserved = name + ".WRONG_UPLOAD-" + original["id"]
            if any(i["name"] == preserved for i in after):
                raise DriveSafetyError("preserved name conflict")
            rename(original, preserved)
        # Recheck after preservation: interrupted runs are recoverable via deterministic pending key.
        final_children = _children(service, parent, budget, names=watched)
        if any(i["name"] == name for i in final_children) or sum(i["name"] == pending_name for i in final_children) != 1:
            raise DriveSafetyError("concurrent canonical/pending change; no promotion")
        promoted = rename(staged, name)
        if not _matches(_get(service, promoted["id"], budget), expected):
            raise DriveSafetyError("promoted verification failed")
        if sum(i["name"] == name for i in _children(service, parent, budget, names=watched)) != 1:
            raise DriveSafetyError("concurrent duplicate after promotion")
        return "repaired"

    prefix = f"{current['category']}/{current['environment']}"
    verified_paths = set(current["verified"])

    def sequence_path(item):
        parts = item["path"].split("/")
        if len(parts) >= 7 and parts[0] in ("Day", "Night"):
            return "/".join(parts[:5])
        return None

    def upload_sequence(sequence, items):
        """없는 motion 하나를 숨김 트리에서 완성하고 폴더 이름 한 번으로 승격한다."""
        destination = prefix + "/" + sequence
        ancestor_path, _, motion_name = destination.rpartition("/")
        parent = folder(ancestor_path)
        manifest = [{"path": item["path"][len(sequence) + 1:], "size": item["size"], "md5": item["md5"]} for item in items]
        key = hashlib.sha256((destination + ":" + json.dumps(manifest, sort_keys=True)).encode()).hexdigest()
        temp_name = ".upload-sequence-" + key
        names = [motion_name, temp_name]
        siblings = _children(service, parent, budget, names=names)
        if any(i["name"] == motion_name for i in siblings):
            raise DriveSafetyError("canonical sequence appeared before staging")
        candidates = [i for i in siblings if i["name"] == temp_name]
        if len(candidates) > 1:
            raise DriveSafetyError("ambiguous staged sequence")
        if candidates:
            staged_root = candidates[0]
            if staged_root.get("mimeType") != FOLDER or staged_root.get("appProperties", {}).get("rewindSequence") != key:
                raise DriveSafetyError("unowned staged sequence")
        else:
            staged_root = budget.call(service.files().create(body={"name": temp_name, "mimeType": FOLDER,
                "parents": [parent], "appProperties": {"rewindSequence": key}}, fields=FIELDS, supportsAllDrives=True), write=True)
        expected = {item["path"]: item for item in manifest}
        required = set()
        for path in expected:
            parts = path.split("/")
            required.update("/".join(parts[:i]) for i in range(1, len(parts)))

        def inspect():
            """정확한 트리 집합. Extras/중복/shortcut/손상은 복구 추정 없이 중단한다."""
            found_files, found_dirs = {}, {"": staged_root["id"]}
            seen_ids = {staged_root["id"]}

            def visit(parent_id, prefix_path):
                children = _children(service, parent_id, budget)
                sibling_names = set()
                for entry in children:
                    name = _name(entry["name"])
                    path = f"{prefix_path}/{name}" if prefix_path else name
                    if name in sibling_names or entry["id"] in seen_ids:
                        raise DriveSafetyError("duplicate/cyclic staged sequence entry")
                    sibling_names.add(name)
                    seen_ids.add(entry["id"])
                    if entry.get("mimeType") == FOLDER:
                        if path not in required:
                            raise DriveSafetyError("extra folder in staged sequence")
                        found_dirs[path] = entry["id"]
                        visit(entry["id"], path)
                    elif path not in expected or not _matches(entry, expected[path]):
                        raise DriveSafetyError("extra/invalid file or shortcut in staged sequence")
                    else:
                        found_files[path] = entry

            visit(staged_root["id"], "")
            return found_files, found_dirs

        existing_files, staged_dirs = inspect()
        reused = len(existing_files)
        for path in sorted(required, key=lambda p: (p.count("/"), p)):
            if path not in staged_dirs:
                parent_path, _, name = path.rpartition("/")
                entry = budget.call(service.files().create(body={"name": name, "mimeType": FOLDER,
                    "parents": [staged_dirs[parent_path]]}, fields=FIELDS, supportsAllDrives=True), write=True)
                staged_dirs[path] = entry["id"]
        for item, member in zip(items, manifest):
            if member["path"] in existing_files:
                continue
            if _digest(Path(item["local_path"])) != (member["size"], member["md5"]):
                raise DriveSafetyError("sequence source changed before upload")
            parent_path, _, name = member["path"].rpartition("/")
            entry = budget.upload(service.files().create(body={"name": name, "parents": [staged_dirs[parent_path]]},
                media_body=media_factory(item["local_path"]), fields=FIELDS, supportsAllDrives=True), member["size"])
            if not _matches(entry, member) or entry.get("name") != name or entry.get("parents") != [staged_dirs[parent_path]]:
                raise DriveSafetyError("sequence upload response size/MD5/path mismatch; staged tree retained")
        complete_files, complete_dirs = inspect()
        if set(complete_files) != set(expected) or set(complete_dirs) != required | {""}:
            raise DriveSafetyError("incomplete staged sequence; no promotion")
        siblings = _children(service, parent, budget, names=names)
        if (len(siblings) != 1 or siblings[0]["id"] != staged_root["id"] or siblings[0]["name"] != temp_name
                or siblings[0].get("appProperties", {}).get("rewindSequence") != key):
            raise DriveSafetyError("canonical/concurrent staged sequence appeared; no promotion")
        rename(siblings[0], motion_name)
        final_siblings = _children(service, parent, budget, names=names)
        if len(final_siblings) != 1 or final_siblings[0]["id"] != staged_root["id"] or final_siblings[0]["name"] != motion_name:
            raise DriveSafetyError("concurrent sequence promotion; final audit required")
        folders[destination] = staged_root["id"]
        operations.append({"sequence": sequence, "status": "sequence_repaired", "files": len(items),
                           "reused_staged_files": reused, "layout": "loose_original_files"})

    if mode == "loose":
        corrupt = {i["path"]: i["remote"] for i in current["corrupt"]}
        needed = set(current["missing"]) | set(corrupt)
        groups = {}
        for item in current["expected"]:
            sequence = sequence_path(item)
            if sequence is not None:
                groups.setdefault(sequence, []).append(item)
        completed = set()
        for sequence, items in sorted(groups.items()):
            if prefix + "/" + sequence not in folders and all(i["path"] in needed for i in items):
                upload_sequence(sequence, items)
                completed.update(i["path"] for i in items)
        # Parent grouping keeps adjacent sequence files together without per-file ancestor lookups.
        for item in sorted(current["expected"], key=lambda i: (str(Path(i["path"]).parent), i["path"])):
            if item["path"] in needed and item["path"] not in completed:
                status = upload(item["local_path"], item, prefix + "/" + item["path"], corrupt.get(item["path"]))
                operations.append({"path": item["path"], "status": status})
    else:
        groups = {}
        for item in current["expected"]:
            parts = item["path"].split("/")
            if len(parts) < 7 or parts[0] not in ("Day", "Night"):
                raise DriveSafetyError("bundle requires Day|Night/human/glasses/take/motion/... layout")
            sequence = "/".join(parts[:5])
            groups.setdefault(sequence, []).append(item)
        for sequence, items in sorted(groups.items()):
            budget.check()
            if all(i["path"] in verified_paths for i in items):
                operations.append({"sequence": sequence, "status": "loose_already_verified"})
                continue
            # ZIP_STORED avoids unexpected CPU expansion, and bounds temporary disk before writing.
            estimate = sum(i["size"] + 2 * len(i["path"].encode("utf-8")) + 256 for i in items) + 65536
            if estimate > max_staging_bytes:
                raise DriveSafetyError("bundle temporary disk reservation limit")
            with tempfile.TemporaryDirectory(prefix="rewind-bundle-", dir=staging_dir) as temp:
                archive = Path(temp) / "sequence.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                    for item in items:
                        budget.check()
                        if _digest(Path(item["local_path"])) != (item["size"], item["md5"]):
                            raise DriveSafetyError("bundle source changed")
                        info = zipfile.ZipInfo(item["path"], date_time=(1980, 1, 1, 0, 0, 0))
                        info.external_attr = 0o100644 << 16
                        with Path(item["local_path"]).open("rb") as src, zf.open(info, "w", force_zip64=True) as dest:
                            for block in iter(lambda: src.read(1024**2), b""):
                                budget.check()
                                dest.write(block)
                with zipfile.ZipFile(archive) as zf:
                    for item in items:
                        md5 = hashlib.md5()
                        with zf.open(item["path"]) as stream:
                            for block in iter(lambda: stream.read(1024**2), b""):
                                budget.check()
                                md5.update(block)
                        if md5.hexdigest() != item["md5"]:
                            raise DriveSafetyError("bundle contents changed")
                size, md5 = _digest(archive)
                # Content-addressed names are idempotent across process interruption (ZIP dates stable).
                label = hashlib.sha256(sequence.encode()).hexdigest()[:16] + "-" + md5 + ".zip"
                destination = prefix + "/__verified_sequence_bundles__/" + label
                status = upload(archive, {"size": size, "md5": md5}, destination)
                operations.append({"sequence": sequence, "bundle_path": destination, "members": [i["path"] for i in items], "status": status})
    return {"mode": mode, "loose_originals_repaired": mode == "loose", "operations": operations,
            "stats": budget.stats(), "requires_final_audit": True,
            "if_match_applied_to_all_renames": all(conditional_renames) if conditional_renames else None,
            "concurrency_contract": "external uploads must be stopped; folder name uniqueness is not transactional"}

"""Standard-library bootstrap embedded in the fixed ArchVis Colab notebook."""

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


def extract_pinned_zip(archive_path, expected_sha256, destination, *,
                       max_zip_bytes, max_output_bytes, max_members, revision=None):
    archive_path, destination = Path(archive_path), Path(destination)
    if not re.fullmatch("[0-9a-f]{64}", expected_sha256) or destination.exists() or destination.is_symlink():
        raise ValueError("zip_pin_or_destination")
    if not destination.parent.is_dir() or archive_path.is_symlink():
        raise ValueError("zip_parent_or_source")
    with archive_path.open("rb") as handle:
        start = os.fstat(handle.fileno())
        if not stat.S_ISREG(start.st_mode) or not 1 <= start.st_size <= max_zip_bytes:
            raise ValueError("zip_input_size")
        digest = hashlib.sha256()
        total = 0
        while chunk := handle.read(min(1024**2, max_zip_bytes - total + 1)):
            total += len(chunk)
            if total > max_zip_bytes:
                raise ValueError("zip_input_size")
            digest.update(chunk)
        if total != start.st_size or digest.hexdigest() != expected_sha256:
            raise ValueError("zip_sha256_mismatch")
        handle.seek(0)
        with zipfile.ZipFile(handle) as archive:
            infos = archive.infolist()
            if not 1 <= len(infos) <= max_members or sum(info.file_size for info in infos) > max_output_bytes:
                raise ValueError("zip_output_limit")
            seen, files, parents = set(), set(), {}
            for info in infos:
                raw_name = info.orig_filename
                if (not raw_name or len(raw_name.encode()) > 1024 or any(c in raw_name for c in '\\:*?<>"|')
                        or any(ord(c) < 32 or ord(c) == 127 for c in raw_name)):
                    raise ValueError("zip_unsafe_name")
                name = raw_name.rstrip("/")
                path = PurePosixPath(name)
                mode = stat.S_IFMT(info.external_attr >> 16)
                if (not path.parts or str(path) != name or path.is_absolute() or ".." in path.parts
                        or mode not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG)
                        or any(part.endswith((".", " ")) or re.match(
                            r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", part, re.I) for part in path.parts)
                        or (info.is_dir() and info.file_size)):
                    raise ValueError("zip_unsafe_entry")
                key = name.casefold()
                if key in seen:
                    raise ValueError("zip_duplicate_name")
                seen.add(key)
                for parent in path.parents:
                    if parent == PurePosixPath("."):
                        continue
                    parent_name = str(parent)
                    parent_key = parent_name.casefold()
                    if parent_key in files or parents.setdefault(parent_key, parent_name) != parent_name:
                        raise ValueError("zip_path_conflict")
                if info.is_dir():
                    if parents.setdefault(key, name) != name:
                        raise ValueError("zip_path_conflict")
                else:
                    if key in parents:
                        raise ValueError("zip_path_conflict")
                    files.add(key)
            if revision is not None:
                stamp = archive.getinfo("SOURCE_REVISION.txt")
                if (not re.fullmatch("[0-9a-f]{40}", revision) or not 40 <= stamp.file_size <= 42
                        or archive.read(stamp).decode("ascii").strip() != revision):
                    raise ValueError("zip_revision_mismatch")
            destination.mkdir()
            written = 0
            for info in infos:
                target = destination / info.filename.rstrip("/")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                remaining = info.file_size
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(min(1024**2, remaining + 1)):
                        remaining -= len(chunk)
                        written += len(chunk)
                        if remaining < 0 or written > max_output_bytes:
                            raise ValueError("zip_output_limit")
                        output.write(chunk)
                if remaining != 0:
                    raise ValueError("zip_short_entry")
        finish = os.fstat(handle.fileno())
        if (start.st_size, start.st_mtime_ns, start.st_ino) != (finish.st_size, finish.st_mtime_ns, finish.st_ino):
            raise ValueError("zip_source_changed")
        # Confirm the same open source after extraction as well.
        handle.seek(0)
        digest = hashlib.sha256()
        remaining = start.st_size
        while chunk := handle.read(min(1024**2, remaining + 1)):
            remaining -= len(chunk)
            if remaining < 0:
                raise ValueError("zip_source_changed")
            digest.update(chunk)
        if remaining or digest.hexdigest() != expected_sha256:
            raise ValueError("zip_source_changed")
    return {"sha256": expected_sha256, "bytes": total, "extracted_bytes": written, "members": len(infos)}

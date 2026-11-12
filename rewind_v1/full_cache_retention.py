"""Local per-chunk cache packages and bounded read-only restore of user-retained copies."""

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import MIB, _local
from rewind_prepare.temporal_cache import _atomic, _hash_file, _locked
from rewind_prepare.temporal_selection import check_pin, seal
from .full_cache import CHUNK, GIB, PreparationBudget, _chunk_rows, _read_receipt, cache_binding


ZIP_LIMIT = 64*MIB
CATALOG_SCHEMA = "archvis_v12_user_retained_frame_cache_v1"


def pack_chunk(cache, manifest, spec_raw, number, output):
    """Create one bounded local ZIP only. The agent never uploads it to Drive."""
    cache, output = _local(cache, directory=True), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(cache):
        raise ValueError("v12_cache_pack_new_output")
    rows = _chunk_rows(manifest["records"], number)
    if type(number) is not int or number < 0 or not rows:
        raise ValueError("v12_cache_pack_number")
    binding = cache_binding(manifest, spec_raw)
    folder = cache / f"chunk-{number:05d}"
    budget = PreparationBudget()
    receipt = _read_receipt(folder, binding, rows, number, budget)
    size = sum(p["bytes"] for p in receipt["files"].values())+MIB
    if size > ZIP_LIMIT or shutil.disk_usage(output.parent).free < size:
        raise ValueError("v12_cache_pack_space")
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".cache-pack-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name in ("images.npy", "targets.pt", "receipt.json"):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100600 << 16
                with (folder / name).open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    while block := source.read(MIB):
                        budget.check()
                        target.write(block)
        if temporary.stat().st_size > ZIP_LIMIT or output.exists() or output.is_symlink():
            raise ValueError("v12_cache_pack_limit")
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"number": number, "frames": len(rows), "receipt_sha256": receipt["content_sha256"],
            "bytes": output.stat().st_size, "sha256": _hash_file(output), "output": str(output)}


def catalog(binding, packages):
    if [p["number"] for p in packages] != list(range(len(packages))):
        raise ValueError("v12_cache_catalog_prefix")
    entries = []
    for package in packages:
        if not 0 < package["bytes"] <= ZIP_LIMIT:
            raise ValueError("v12_cache_catalog_size")
        name = f"chunk-{package['number']:05d}-{package['sha256']}.zip"
        entries.append({key: package[key] for key in ("number", "frames", "receipt_sha256", "bytes", "sha256")} | {"name":name})
    return seal({"schema": CATALOG_SCHEMA, "binding": binding, "chunks": entries,
                 "completed_frames": sum(p["frames"] for p in entries), "status": "user_copy_verification_required",
                 "total_frames": binding["total_frames"]})


def restore_cache(retained, catalog_path, catalog_sha, manifest, spec_raw, output, *, budget=None):
    """Read retained chunks; never modify or upload to the retained source."""
    from scripts.archvis_pilot_bootstrap import extract_pinned_zip
    budget = budget or PreparationBudget()
    retained, catalog_path = _local(retained, directory=True), _local(catalog_path)
    raw = _regular_read(catalog_path, MIB)
    if sha(raw) != catalog_sha:
        raise ValueError("v12_cache_catalog_file_pin")
    doc = json.loads(raw)
    check_pin(doc, CATALOG_SCHEMA, doc.get("content_sha256"))
    binding = cache_binding(manifest, spec_raw)
    if doc["binding"] != binding or len(doc["chunks"]) > (len(manifest["records"])+CHUNK-1)//CHUNK:
        raise ValueError("v12_cache_restore_binding")
    if [p["number"] for p in doc["chunks"]] != list(range(len(doc["chunks"]))):
        raise ValueError("v12_cache_restore_prefix")
    if doc["completed_frames"] != sum(p["frames"] for p in doc["chunks"]):
        raise ValueError("v12_cache_restore_count")
    output = Path(output).absolute()
    if output.is_relative_to(retained) or retained.is_relative_to(output):
        raise ValueError("v12_cache_restore_overlap")
    output.mkdir(exist_ok=True)
    _local(output, directory=True)
    completed, copied = 0, 0
    with _locked(output):
        allowed_chunks = {f"chunk-{p['number']:05d}" for p in doc["chunks"]}
        if any(p.name.startswith("chunk-") and p.name not in allowed_chunks for p in output.iterdir()):
            raise ValueError("v12_cache_restore_older_catalog")
        binding_file = output / "binding.json"
        if binding_file.exists() and json.loads(_regular_read(binding_file, MIB)) != binding:
            raise ValueError("v12_cache_restore_existing_binding")
        if not binding_file.exists():
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("v12_cache_restore_nonempty")
            _atomic(binding_file, canonical(binding)+b"\n")
        for entry in doc["chunks"]:
            budget.check()
            number = entry["number"]
            rows = _chunk_rows(manifest["records"], number)
            expected_name = f"chunk-{number:05d}-{entry['sha256']}.zip"
            if entry["name"] != expected_name or entry["frames"] != len(rows) or not 0 < entry["bytes"] <= ZIP_LIMIT:
                raise ValueError("v12_cache_restore_entry")
            folder = output / f"chunk-{number:05d}"
            if not folder.exists():
                if copied+entry["bytes"] > budget.limit:
                    break
                if shutil.disk_usage(output).free < 2*ZIP_LIMIT+2*GIB:
                    raise ValueError("v12_cache_restore_space")
                path = _local(retained / entry["name"])
                if path.stat().st_size != entry["bytes"]:
                    raise ValueError("v12_cache_restore_zip_size")
                with tempfile.TemporaryDirectory(dir=output, prefix=".pending-") as temporary:
                    staged = Path(temporary) / "restored"
                    local_zip = Path(temporary) / "chunk.zip"
                    measured, size = hashlib.sha256(), 0
                    with path.open("rb") as source, local_zip.open("xb") as destination:
                        while block := source.read(min(MIB, entry["bytes"]-size+1)):
                            budget.check()
                            size += len(block)
                            if size > entry["bytes"]:
                                raise ValueError("v12_cache_restore_zip_growth")
                            measured.update(block)
                            destination.write(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if size != entry["bytes"] or measured.hexdigest() != entry["sha256"]:
                        raise ValueError("v12_cache_restore_zip_hash")
                    extract_pinned_zip(local_zip, entry["sha256"], staged, max_zip_bytes=ZIP_LIMIT,
                                       max_output_bytes=ZIP_LIMIT, max_members=3)
                    copied += entry["bytes"]
                    receipt = _read_receipt(staged, binding, rows, number, budget)
                    if receipt["content_sha256"] != entry["receipt_sha256"]:
                        raise ValueError("v12_cache_restore_receipt_pin")
                    staged.rename(folder)
            receipt = _read_receipt(folder, binding, rows, number, budget)
            if receipt["content_sha256"] != entry["receipt_sha256"]:
                raise ValueError("v12_cache_restore_receipt_pin")
            completed += len(rows)
    return {"status": "completed" if completed == binding["total_frames"] else "partial",
            "completed_frames": completed, "retained_prefix_frames": doc["completed_frames"],
            "source_rgb_pose_bytes": 0, "retained_zip_bytes_this_invocation": copied,
            "cache_verified_bytes_this_invocation": budget.cache_verified_bytes}

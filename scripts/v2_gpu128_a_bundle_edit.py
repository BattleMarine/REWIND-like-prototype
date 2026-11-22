"""Copy a narrow editable model overlay onto a verified frozen source tree."""

import hashlib
import json
from pathlib import Path


SCHEMA = "rewind_v2_gpu128_editable_source_v1"
EDITABLE = frozenset({
    "configs/rewind_v12_full_training.json",
    "rewind_v1/attention.py", "rewind_v1/frontend.py", "rewind_v1/run_stages.py",
    "rewind_v1/teacher.py", "rewind_v1/training.py",
    "rewind_m1/frontend.py", "rewind_m2/diffusion.py", "rewind_m2/models.py", "rewind_m2/training.py",
    "rewind_m3/cascade.py", "rewind_m3/models.py",
})


def canonical(doc):
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def files(root):
    result = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(p in (".git", "__pycache__") for p in relative.parts) or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"editable_source_link: {relative}")
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def pin_source(base, changes):
    body = {"schema": SCHEMA, "base_source_sha256": base, "files": changes}
    return {**body, "source_sha256": hash_bytes(canonical(body))}


def apply_snapshot(frozen, editable, base):
    """Only the declared files are copied. The caller owns the frozen scratch tree."""
    if not editable.is_dir() or editable.is_symlink() or editable.resolve() != editable:
        raise ValueError("editable_source_directory")
    original, fork = files(frozen), files(editable)
    if original.keys() != fork.keys():
        raise ValueError("editable_source_file_set: preserve the files created by fork-source")
    payloads = {}
    for name, path in sorted(original.items()):
        candidate = fork[name]
        if name in EDITABLE:
            if candidate.stat().st_size > 2*1024**2:
                raise ValueError(f"editable_source_file_too_large: {name}")
            new = candidate.read_bytes()
            if new != path.read_bytes():
                payloads[name] = new
        else:
            if path.stat().st_size != candidate.stat().st_size:
                raise ValueError(f"editable_source_out_of_scope: {name}")
            with path.open("rb") as a, candidate.open("rb") as b:
                if hashlib.file_digest(a, "sha256").digest() != hashlib.file_digest(b, "sha256").digest():
                    raise ValueError(f"editable_source_out_of_scope: {name}")
    for name, raw in payloads.items():
        original[name].write_bytes(raw)
    return pin_source(base, {name: hash_bytes(raw) for name, raw in payloads.items()})


def validate_pin(manifest, source, base):
    changes = manifest.get("files", {})
    if not set(changes) <= EDITABLE or manifest != pin_source(base, changes):
        raise ValueError("editable_source_identity_manifest")
    for name, expected in changes.items():
        if hash_bytes((source/name).read_bytes()) != expected:
            raise ValueError(f"editable_source_snapshot_sha: {name}")
    return manifest["source_sha256"]


def require_resume_source(signature, source_sha):
    if signature.get("source") != source_sha:
        raise ValueError("portable_full_run_source_changed: restore the original editable source files or create a new output")

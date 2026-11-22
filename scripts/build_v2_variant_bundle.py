"""Package one completed v2 GPU128 B/C run without copying the A data cache."""

import argparse
from contextlib import contextmanager
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

if __package__:
    from . import v2_gpu128_a_bundle_format as fmt
else:
    import v2_gpu128_a_bundle_format as fmt


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rewind_v2_gpu128_variant_v1"
BASE_SHA = "4591393d3aceb4bb4af9b387c4fda038bdcdd5fd5d613fdcc95f7cb50cdc736a"
BASE_INDEX_SHA = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
SEEDS = {"B": 71, "C": 103}
DIAGNOSTIC_SHA = {
    "B": "31b76e298140aed080f135755f3a09f6f4a95d420d470d5224e6707b5314ae43",
    "C": "e786a21d9c00f384df9eb6389d405103e2ae30882dee6002014ec7dc2f40a83b",
}
COMPARISON_SHA = {
    "B": "b153fc42f1e3f0f628ade52fc44e0a45794dd0f13e71b6b089feacb0023e46c0",
    "C": "4b2f404d1b02ceb75e280172b072c9f2a6fefa2b4502c97f7f2a8d67536297ad",
}
FEATURE_IDENTITIES = {
    "B": "be05c981b4e84f59c3986c07ded70dfe31c303ea80b98cd435f5f82c4977be98",
    "C": "3f1294ad63ff1ac32d6076cc9a21473b6234a96bf054a6b36549d873aed6de4c",
}
ROLES = ("frontend", "body", "hand")
CURSOR = {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}
MARKER = "variant.json"
COMMON_TOOLS = ("format", "local", "predict", "tune", "train_full", "edit")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def _regular(path):
    _require(path.is_file() and not path.is_symlink(), f"Expected regular file: {path}")
    with path.open("rb") as stream:
        yield stream


def _read(path, limit=fmt.MIB):
    with _regular(path) as stream:
        raw = stream.read(limit + 1)
    _require(0 < len(raw) <= limit, f"File size outside limit: {path}")
    return raw


def _pin(raw):
    return {"bytes": len(raw), "sha256": fmt.sha(raw)}


def _archive_pin(record, limit):
    _require(type(record["bytes"]) is int and 0 < record["bytes"] <= limit
             and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "Invalid evidence archive pin")
    return {key: record[key] for key in ("bytes", "sha256")}


def _base_metadata(path):
    with _regular(path) as stream:
        _require(file_digest(stream, "sha256").hexdigest() == BASE_SHA, "Fixed A bundle SHA-256 mismatch")
    with fmt.BundleView(path) as base:
        raw = base.read(fmt.INDEX, 2 * fmt.MIB)
        _require(fmt.sha(raw) == BASE_INDEX_SHA, "Fixed A index SHA-256 mismatch")
        index = base.index()
        _, diagnostic, _, comparison = fmt.metadata(base, index["members"])
    identity = index["identity"]
    _require(identity == diagnostic["identity"] and identity["slot"] == "A" and identity["seed"] == 42
             and identity["cache_boundary"] == index["cache_boundary_sha256"] == fmt.FINAL_BOUNDARY
             and index["cache_shards"] == 1063 and index["frame_pairs"] == 136011
             and index["features"]["identity"] == fmt.FEATURE_IDENTITY, "Fixed A data identity mismatch")
    return diagnostic, comparison, index["members"]


def _validate_evidence(slot, diagnostic, comparison, base_diagnostic, base_comparison):
    fmt.check_json_pin(diagnostic, "rewind_v2_gpu_invocation_v1")
    fmt.check_json_pin(comparison, "rewind_v2_gpu_comparison_v1")
    step, identity = diagnostic["checkpoint"], diagnostic["identity"]
    fmt.check_json_pin(step, "rewind_v2_gpu_training_checkpoint_v1")
    cursor = diagnostic["checkpoint_cursor"]
    _require(diagnostic["slot"] == identity["slot"] == slot and identity["seed"] == SEEDS[slot]
             and step["identity"] == identity and diagnostic["status"] == "completed"
             and step["completed"] is True and step["cursor"] == CURSOR
             and diagnostic["trained_updates"] == CURSOR["updates"]
             and all(cursor[key] == value for key, value in CURSOR.items()), "Completed slot or checkpoint identity mismatch")
    base_identity = base_diagnostic["identity"]
    _require(all(identity[key] == base_identity[key] for key in
                 ("run_id", "source", "manifest", "cache_boundary", "eval_seed"))
             and diagnostic["cache_shards"] == base_diagnostic["cache_shards"]
             and diagnostic["counts"] == base_diagnostic["counts"], "A image cache identity mismatch")
    binding = comparison["binding"]
    _require(comparison == diagnostic["comparison"] and binding["experiment"] == identity
             and binding["eval_seed"] == identity["eval_seed"] == 71
             and binding["clips"] == diagnostic["common_clips"] == base_comparison["binding"]["clips"],
             "Comparison does not match the completed slot and shared validation clips")
    best = cursor["best"]
    _require(set(best) == set(binding["model"]) == set(ROLES)
             and {role: best[role]["sha256"] for role in ROLES} == binding["model"]
             and binding["model"]["frontend"] != base_comparison["binding"]["model"]["frontend"]
             and diagnostic["timing"]["feature_identity"] == FEATURE_IDENTITIES[slot],
             "Model selection or frontend-specific feature identity mismatch")


def _check_checkpoint(view, diagnostic):
    receipt = json.loads(view.read("recovery_receipt.json", 8 * fmt.MIB))
    fmt.check_json_pin(receipt, "rewind_v2_result_recovery_v1")
    files = receipt["files"]
    step = diagnostic["checkpoint"]
    _require(view.members.keys() == files.keys() | {"recovery_receipt.json"}
             and files["last.pt"] == _archive_pin(step["last"], fmt.GIB),
             "Checkpoint archive listing or last state does not match the diagnostic")
    for name, item in files.items():
        pin = _archive_pin(item, fmt.GIB)
        _require(fmt.name_path(name).name == name and view.members[name].file_size == pin["bytes"],
                 f"Checkpoint member size or name mismatch: {name}")
    cursor = diagnostic["checkpoint_cursor"]
    selected = [*cursor["best"].values(),
                *(pin for group in cursor["milestones"].values() for pin in group.values())]
    _require(all(files.get(item["name"]) == _archive_pin(item, fmt.GIB) for item in selected),
             "Checkpoint does not contain the recorded model weights")


def _check_comparison(view, diagnostic, comparison):
    clips = comparison["binding"]["clips"]
    expected = {f"{name}.{suffix}" for name in clips for suffix in ("gif", "npz")}
    metrics = json.loads(view.read("comparison_metrics.json", fmt.MIB))
    _require(view.members.keys() == expected | {"comparison_metrics.json"}
             and metrics["binding"] == comparison["binding"]
             and metrics["models"] == diagnostic["checkpoint_cursor"]["best"]
             and metrics["scenario_scores"] == comparison["scores"]
             and metrics["macro_score_mm"] == comparison["macro_score_mm"],
             "Comparison archive does not match the selected models and evaluation")


def _tools(base_pins):
    scripts = ROOT / "scripts"
    runtimes = sorted(scripts.glob("v2_variant_runtime*.py"))
    _require(scripts / "v2_variant_runtime.py" in runtimes, "Missing v2_variant_runtime.py")
    paths = {"v2_variant.py": scripts / "v2_variant.py",
             "GUIDE_KO.md": ROOT / "docs" / "V2_GPU128_BC_PORTABLE.md",
             **{f"tools/{path.name}": path for path in runtimes}}
    sources = {name: _read(path, 2 * fmt.MIB) for name, path in paths.items()}
    for suffix in COMMON_TOOLS:
        name = f"v2_gpu128_a_bundle_{suffix}.py"
        key = f"tools/{name}"
        raw = _read(scripts / name, 2 * fmt.MIB)
        # Git may check out these files with CRLF. The original A exporter used UTF-8 LF.
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        _require(_pin(raw) == base_pins.get(key), f"Fixed A code pin mismatch: {key}")
        sources[key] = raw
    for name, raw in sources.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    return sources


def build(slot, base, diagnostic, checkpoint, comparison, comparison_zip, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    small = {"diagnostic.json": _read(diagnostic), "comparison.json": _read(comparison)}
    _require(fmt.sha(small["diagnostic.json"]) == DIAGNOSTIC_SHA[slot], "Fixed slot diagnostic SHA-256 mismatch")
    _require(fmt.sha(small["comparison.json"]) == COMPARISON_SHA[slot], "Fixed slot comparison SHA-256 mismatch")
    report = json.loads(small["diagnostic.json"])
    marker = json.loads(small["comparison.json"])
    evidence = {"checkpoint.zip": (checkpoint, _archive_pin(report["checkpoint"]["archive"], fmt.GIB)),
                "comparison.zip": (comparison_zip, _archive_pin(marker["archive"], 32 * fmt.MIB))}
    for path, pin in evidence.values():
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"],
                 f"Evidence archive missing or size mismatch: {path}")
    base_diagnostic, base_comparison, base_pins = _base_metadata(base)
    _validate_evidence(slot, report, marker, base_diagnostic, base_comparison)
    small.update(_tools(base_pins))
    with fmt.BundleView(checkpoint) as view:
        _check_checkpoint(view, report)
    with fmt.BundleView(comparison_zip) as view:
        _check_comparison(view, report, marker)
    members = {**{name: _pin(raw) for name, raw in small.items()},
               **{name: pin for name, (_, pin) in evidence.items()}}
    manifest = {"schema": SCHEMA, "slot": slot, "seed": SEEDS[slot], "members": members}
    sealed = fmt.canonical({**manifest, "content_sha256": fmt.sha(fmt.canonical(manifest))})
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v2-variant-") as scratch:
        staged = Path(scratch) / output.name
        with zipfile.ZipFile(staged, "x", zipfile.ZIP_STORED) as archive:
            for name, raw in sorted({**small, MARKER: sealed}.items()):
                archive.writestr(zipfile.ZipInfo(name), raw)
            for name, (path, pin) in evidence.items():
                with _regular(path) as source, archive.open(zipfile.ZipInfo(name), "w", force_zip64=True) as entry:
                    fmt.hash_stream(source, pin, destination=entry)
        with fmt.BundleView(staged) as view:
            _require(view.members.keys() == members.keys() | {MARKER}
                     and view.read(MARKER, fmt.MIB) == sealed, "Output ZIP listing or manifest mismatch")
            for name, pin in members.items():
                with view.open(name, pin["bytes"]) as entry:
                    fmt.hash_stream(entry, pin)
        with staged.open("rb") as stream:
            result = {"bytes": staged.stat().st_size, "sha256": file_digest(stream, "sha256").hexdigest()}
        # Windows rename fails if the destination exists. POSIX rename would replace it.
        if os.name == "nt":
            staged.rename(output)
        else:
            os.link(staged, output)
    return {"path": str(output), "slot": slot, "seed": SEEDS[slot], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", choices=SEEDS, required=True)
    for name in ("base", "diagnostic", "checkpoint", "comparison", "comparison-zip", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == "__main__":
    main()

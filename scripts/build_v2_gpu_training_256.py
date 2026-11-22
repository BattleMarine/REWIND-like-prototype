"""Release a 256px GPU notebook over the verified fixed b7a2d00 128 training ZIP."""

import argparse
import json
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _hash_file
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_v2_gpu_training import ROOT, notebook, source_map, write_source_zip
from scripts.run_v2_gpu_training_256 import BOUNDARY, PRODUCER_BINDING, RESOURCE_LIMITS, SIDE
from scripts.v2_gpu_training_cache256 import sha
from scripts.v2_gpu_training_storage import FRAMES, MIB, SHARDS


BASELINE_REVISION = "b7a2d00f3bc755808a451ed4ead873e6513d094b"
BASELINE_ZIP_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
BASELINE_RECEIPT_SHA = "7a699521ec937b4481ec544f9d1e01fc7a7f7abb58d20442afab86c85e73d43d"
EVIDENCE_SHA = "765c112f0dde33414da3bd1f0d730ac20ba903d4b33dddeaceee1aa95caee9fd"
SOURCE = "v2_gpu_training_256_v1.zip"
NOTEBOOK = "v2_gpu_training_256_v1.ipynb"
TRAINING = ("scripts/run_v2_gpu_training.py", "scripts/v2_gpu_training_pipeline.py",
            "scripts/v2_gpu_training_features.py", "scripts/v2_gpu_training_comparison.py",
            "scripts/v2_gpu_training_cache256.py", "scripts/run_v2_gpu_training_256.py")
PRODUCER = ("scripts/run_v2_cache256.py", "scripts/run_v2_resolution_ab.py", "scripts/v2_ordered_prefetch.py",
            "scripts/v2_bulk_quota.py")


def load_evidence(completed_report, baseline):
    raw = _regular_read(completed_report, 192*1024)
    if sha(raw) != EVIDENCE_SHA:
        raise ValueError("v2_gpu256_completed_evidence_pin")
    evidence = json.loads(raw)
    worker, pins = evidence["worker"], evidence["pins"]
    scope = (worker["status"], worker["stop_reason"], worker["mode"], worker["boundary_sha256"], worker["binding"],
             worker["shards"], worker["frames"], worker["drive_bytes"])
    if scope != ("passed", "target_reached", "long", BOUNDARY, PRODUCER_BINDING, SHARDS, FRAMES, 214229436279):
        raise ValueError("v2_gpu256_completed_evidence_scope")
    for key in ("embedded_manifest", "archive_sha256", "manifest_bytes", "manifest_file_sha256",
                "manifest_content_sha256", "spec_sha256", "frames"):
        if pins[key] != baseline[key]:
            raise ValueError(f"v2_gpu256_128_input_comparability: {key}")
    return worker, pins


def build(baseline_zip, baseline_receipt, completed_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_gpu256_release_directory")
    raw = _regular_read(baseline_receipt, 64*1024)
    if sha(raw) != BASELINE_RECEIPT_SHA:
        raise ValueError("v2_gpu256_128_receipt_pin")
    baseline = json.loads(raw)
    base_pins = baseline["pins"]
    if base_pins["source_sha256"] != BASELINE_ZIP_SHA:
        raise ValueError("v2_gpu256_128_source_pin")
    worker, producer = load_evidence(completed_report, base_pins)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gpu256-release-") as directory:
        root, release = Path(directory)/"source", Path(directory)/"release"
        release.mkdir()
        extract_pinned_zip(baseline_zip, BASELINE_ZIP_SHA, root, max_zip_bytes=64*MIB,
                           max_output_bytes=192*MIB, max_members=3000, revision=base_pins["revision"])
        for name in (*TRAINING, *PRODUCER):
            source = (ROOT/name).read_text(encoding="utf-8").encode("utf-8")
            if name in PRODUCER and sha(source) != producer["consumer_overlay"][name]:
                raise ValueError(f"v2_gpu256_producer_code_pin: {name}")
            (root/name).write_bytes(source)
        core = digest(source_map(root))
        if core != base_pins["training_core_sha256"]:
            raise ValueError("v2_gpu256_training_core_drift")
        source_path = release/SOURCE
        write_source_zip(root, source_path)
        names = dict.fromkeys((*base_pins["training_modules"], *TRAINING, *PRODUCER))
        pins = {key: base_pins[key] for key in ("revision", "embedded_manifest", "archive_sha256", "manifest_bytes",
                                                "manifest_file_sha256", "manifest_content_sha256", "spec_sha256", "frames",
                                                "seeds", "eval_seed")}
        pins.update(source_name=SOURCE, source_sha256=_hash_file(source_path), size=SIDE,
                    baseline_revision=BASELINE_REVISION, baseline_source_sha256=BASELINE_ZIP_SHA,
                    baseline_notebook_sha256=baseline["notebook_sha256"], baseline_receipt_sha256=BASELINE_RECEIPT_SHA,
                    final_evidence_sha256=EVIDENCE_SHA, final_boundary_sha256=BOUNDARY, reported_cache_bytes=worker["drive_bytes"],
                    producer_256_binding=PRODUCER_BINDING, producer_256_overlay=producer["consumer_overlay"],
                    producer_256_runtime=producer["cpu_runtime"], training_core_sha256=core,
                    training_modules={name: _hash_file(root/name) for name in names}, resources=RESOURCE_LIMITS)
        code = canonical(notebook(pins, side=SIDE))
        (release/NOTEBOOK).write_bytes(code)
        (release/"receipt.json").write_bytes(canonical({"pins": pins, "source_bytes": source_path.stat().st_size,
            "notebook": NOTEBOOK, "notebook_bytes": len(code), "notebook_sha256": sha(code), "default_off": True}))
        release.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-zip", "baseline-receipt", "completed-report", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    build(**vars(parser.parse_args()))

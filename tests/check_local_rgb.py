"""작은 합성 PNG만 사용하는 독립 CPU 검사. 전체 회귀/학습/네트워크 없음.

사용자 notebook이 이 명령을 subprocess timeout=120으로 한 번 실행한다.
이번 준비에서는 정적 검토만 했으며 실행 결과는 아직 없다.
"""

import argparse
import builtins
import copy
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
from unittest.mock import patch


LOCAL_RGB_CHECK_API = 1
ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_REVISION = "7ccf11c1823e6bba7e9f567362eaea3bb3edc597"
LOADER_SHA256_LF = "69c82379447bdde8141a8f20d2fe78b3bd5d4c018919ede17fc25c86811a7896"
FIXTURE_LIMIT = 128 * 1024
REPORT_LIMIT = 256 * 1024
CASE_NAMES = (
    "constructor_lazy_and_requested_pair", "rgb_rgba_values_and_batch",
    "missing_view_duplicate_identity_and_member", "path_escape",
    "split_boundaries", "declared_and_png_resolution", "size_and_hash_mismatch",
    "manifest_record_file_pixel_caps", "batch_caps_before_read",
    "transparent_png", "truncated_png", "handles_and_sample_reread",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exception_info(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:1200],
            "traceback": traceback.format_exc()[-5000:]}


def save_report(output, report):
    raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(len(raw) <= REPORT_LIMIT, "report_byte_limit")
    temporary = output / "report.json.tmp"
    temporary.write_bytes(raw)
    temporary.replace(output / "report.json")


def source_info():
    paths = [ROOT / "tests/check_local_rgb.py"]
    for package in ("rewind_m1", "rewind_archive", "rewind_data"):
        paths.extend(sorted((ROOT / package).glob("*.py")))
    stamp = ROOT / "SOURCE_REVISION.txt"
    revision = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else "unversioned"
    return {"revision": revision, "production_revision": PRODUCTION_REVISION,
            "expected_loader_sha256_lf": LOADER_SHA256_LF,
            "files_sha256_lf": {p.relative_to(ROOT).as_posix(): sha(p.read_bytes().replace(b"\r\n", b"\n")) for p in paths}}


def run_checks(output, report):
    sys.path.insert(0, str(ROOT))
    import torch
    import PIL
    from PIL import Image
    from rewind_archive.contracts import digest
    import rewind_m1.local_rgb as module

    require(Path(module.__file__).resolve() == ROOT / "rewind_m1/local_rgb.py", "wrong_loader_module")
    torch.set_num_threads(1)
    report["environment"].update(torch=str(torch.__version__), Pillow=PIL.__version__, device="cpu", threads=1)
    Loader, Limits, Error = module.LocalStereoRGB, module.RGBLimits, module.RGBContractError

    class Fixture:
        def __init__(self, base):
            self.root = base / "rgb"
            self.root.mkdir()
            self.manifest = base / "manifest.json"
            self.expected, self.original, rows = {}, {}, []
            for index in range(3):
                seq = "Fixture/Day/person/avatar/001/Motion" if index < 2 else "Fixture/Day/person/avatar/002/Other"
                frame = index if index < 2 else 0
                row = {"sequence_id": seq, "frame_index": frame,
                       "split": "train" if index < 2 else "validation",
                       "split_evidence": "Synthetic CPU contract fixture only.", "views": {}}
                for view, side in enumerate(("left", "right")):
                    colors = [((17*x + 29*index + 31*view) % 256,
                               (47*y + 13*index + 61*view) % 256,
                               (19*x + 23*y + 71*index + 7*view) % 256) for y in range(6) for x in range(8)]
                    colors[0] = (0, 128, 255) if view == 0 else (255, 0, 64)
                    rgb = bytes(v for color in colors for v in color)
                    self.expected[index, side] = rgb
                    mode = "RGBA" if index == 1 else "RGB"
                    pixels = bytes(v for color in colors for v in (*color, 255)) if mode == "RGBA" else rgb
                    member = f"{seq}/fisheye_final_image/camera_{side}/final_{frame}.png"
                    path = self.root / "files" / member
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.frombytes(mode, (8, 6), pixels) as image:
                        image.save(path, format="PNG", compress_level=1)
                    raw = path.read_bytes()
                    self.original[index, side] = raw
                    row["views"][side] = {"path": path.relative_to(self.root).as_posix(), "member": member,
                        "size_bytes": len(raw), "sha256": sha(raw), "image_size": [8, 6],
                        "source_id": "fixture", "tar_complete": True}
                rows.append(row)
            archive = {"schema_version": "archive_manifest_v1", "archive_id": "local_rgb_cpu_fixture",
                "dataset_version": "synthetic_rgb_cpu_v1", "kind": "rgb", "source_scope": "synthetic_fixture",
                "total_parts": 1, "parts": [{"index": 0, "name": "fixture.tar.gz000", "size_bytes": 1,
                "version": "synthetic", "source": {"type": "local", "path": str(base / "declared_fixture_source")},
                "identity": {"mtime_ns": 0}}]}
            self.doc = {"schema_version": "local_stereo_rgb_v1", "id": "synthetic_rgb_cpu_v1",
                "dataset_version": "synthetic_rgb_cpu_v1", "data_kind": "synthetic_fixture", "image_size": [8, 6],
                "sources": {"fixture": {"archive_manifest": archive, "extraction_manifest_sha256": "0" * 64}},
                "records": rows}
            self.write()

        def write(self, doc=None):
            doc = copy.deepcopy(self.doc if doc is None else doc)
            doc["content_sha256"] = digest({k: v for k, v in doc.items() if k != "content_sha256"})
            self.manifest.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
            return self.manifest

        def loader(self, doc=None, **limits):
            return Loader(self.write(doc), self.root, limits=Limits(**limits))

        def path(self, index, side):
            return self.root / self.doc["records"][index]["views"][side]["path"]

        def replace_png(self, index, side, raw):
            self.path(index, side).write_bytes(raw)
            doc = copy.deepcopy(self.doc)
            doc["records"][index]["views"][side].update(size_bytes=len(raw), sha256=sha(raw))
            return doc

        def tensor(self, index):
            return torch.stack([torch.tensor(list(self.expected[index, side]), dtype=torch.float32)
                                .reshape(6, 8, 3).permute(2, 0, 1) / 255.0 for side in ("left", "right")])

    @contextmanager
    def trace_io():
        paths, handles, png_fds, decodes = [], [], set(), []
        real_os_open, real_fdopen = os.open, os.fdopen
        real_builtin, real_io, real_image = builtins.open, io.open, Image.open

        def record(path):
            if isinstance(path, (str, os.PathLike)) and Path(path).suffix.lower() == ".png":
                paths.append(Path(path).resolve())
                return True
            return False

        def os_open(path, *args, **kwargs):
            is_png = record(path)
            fd = real_os_open(path, *args, **kwargs)
            if is_png:
                png_fds.add(fd)
            return fd

        def fdopen(fd, *args, **kwargs):
            handle = real_fdopen(fd, *args, **kwargs)
            if fd in png_fds:
                handles.append(handle)
            return handle

        def file_open(original):
            def wrapped(path, *args, **kwargs):
                is_png = record(path)
                handle = original(path, *args, **kwargs)
                if is_png:
                    handles.append(handle)
                return handle
            return wrapped

        def image_open(*args, **kwargs):
            decodes.append(True)
            return real_image(*args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(os, "open", os_open))
            stack.enter_context(patch.object(os, "fdopen", fdopen))
            stack.enter_context(patch.object(builtins, "open", file_open(real_builtin)))
            stack.enter_context(patch.object(io, "open", file_open(real_io)))
            stack.enter_context(patch.object(Image, "open", image_open))
            yield paths, handles, decodes

    expected_errors = []

    def reject(function, reason):
        try:
            function()
        except Error as exc:
            require(str(exc) == reason, f"wrong_rejection: {str(exc)!r} != {reason!r}")
            expected_errors.append({"type": type(exc).__name__, "reason": str(exc)})
        else:
            raise AssertionError("required_rejection_missing: " + reason)

    def constructor_lazy_and_requested_pair(f):
        with trace_io() as (paths, handles, decodes):
            loader = Loader(f.manifest, f.root)
            loader.metadata(2)
            require(not paths and not decodes, "constructor_or_metadata_reads_png")
            sample = loader[1]
            require(paths == [f.path(1, s).resolve() for s in ("left", "right")], "wrong_pair_or_eager_read")
            require(bool(decodes) and torch.equal(sample["images"], f.tensor(1)), "pair_decode_or_pixels")
            require(all(h.closed for h in handles), "png_handle_left_open")

    def rgb_rgba_values_and_batch(f):
        loader = f.loader()
        for i in (0, 1):
            sample = loader[i]
            value = sample["images"]
            require(value.shape == (2, 3, 6, 8) and value.dtype == torch.float32 and value.device.type == "cpu", "sample_shape_dtype_device")
            require(torch.equal(value, f.tensor(i)) and float(value.min()) == 0.0 and float(value.max()) == 1.0, "rgb_channels_or_scaling")
            require(sample["metadata"]["data_kind"] == "synthetic_fixture", "data_kind_changed")
        batch = loader.load_batch([0, 1])
        require(batch["images"].shape == (2, 2, 3, 6, 8) and torch.equal(batch["images"], torch.stack([f.tensor(0), f.tensor(1)])), "batch_pixels_or_order")
        require([m["record"]["frame_index"] for m in batch["metadata"]] == [0, 1], "batch_identity_order")

    def missing_view_duplicate_identity_and_member(f):
        doc = copy.deepcopy(f.doc); del doc["records"][0]["views"]["right"]
        reject(lambda: f.loader(doc), "rgb_missing_or_extra_view")
        doc = copy.deepcopy(f.doc); doc["records"].append(copy.deepcopy(doc["records"][0]))
        reject(lambda: f.loader(doc), "rgb_duplicate_sequence_frame")
        doc = copy.deepcopy(f.doc); doc["records"][0]["views"]["left"]["member"] = doc["records"][0]["views"]["right"]["member"]
        reject(lambda: f.loader(doc), "rgb_member_identity_mismatch")
        f.path(0, "left").unlink()
        reject(lambda: f.loader(), "rgb_manifest_or_local_file_invalid")

    def path_escape(f):
        for relative in ("../outside.png", str(f.path(0, "left").resolve())):
            doc = copy.deepcopy(f.doc); doc["records"][0]["views"]["left"]["path"] = relative
            reject(lambda: f.loader(doc), "rgb_unsafe_path")

    def split_boundaries(f):
        doc = copy.deepcopy(f.doc); doc["records"][1]["split"] = "validation"
        reject(lambda: f.loader(doc), "rgb_sequence_split_overlap")
        loader = f.loader()
        reject(lambda: loader.load_batch([0, 2]), "rgb_batch_mixed_split")

    def declared_and_png_resolution(f):
        doc = copy.deepcopy(f.doc); doc["records"][0]["views"]["right"]["image_size"] = [9, 6]
        reject(lambda: f.loader(doc), "rgb_incomplete_or_mixed_resolution")
        doc = copy.deepcopy(f.doc); doc["image_size"] = [9, 6]
        for row in doc["records"]:
            for view in row["views"].values(): view["image_size"] = [9, 6]
        loader = f.loader(doc)
        reject(lambda: loader[0], "rgb_decoded_size_or_pixel_limit")

    def size_and_hash_mismatch(f):
        doc = copy.deepcopy(f.doc); doc["records"][0]["views"]["left"]["size_bytes"] += 1
        reject(lambda: f.loader(doc), "rgb_file_size_mismatch")
        loader = f.loader(); raw = f.original[0, "left"]
        f.path(0, "left").write_bytes(raw + b"x")
        reject(lambda: loader[0], "rgb_file_size_mismatch")
        f.path(0, "left").write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        reject(lambda: loader[0], "rgb_file_hash_mismatch")

    def manifest_record_file_pixel_caps(f):
        reject(lambda: f.loader(max_manifest_bytes=1), "rgb_file_type_or_byte_limit")
        reject(lambda: f.loader(max_records=1), "rgb_record_limit")
        reject(lambda: f.loader(max_file_bytes=1), "rgb_view_size_or_hash")
        reject(lambda: f.loader(max_pixels_per_view=47), "rgb_image_size_or_pixel_limit")
        largest = max(len(v) for v in f.original.values())
        loader = f.loader(max_file_bytes=largest)
        f.path(0, "left").write_bytes(f.original[0, "left"] + b"x" * (largest + 1))
        reject(lambda: loader[0], "rgb_file_type_or_byte_limit")

    def batch_caps_before_read(f):
        loader = f.loader(max_batch_size=1)
        bounded = f.loader(max_batch_tensor_bytes=1800)
        with trace_io() as (paths, _, decodes):
            reject(lambda: loader.load_batch([0, 1]), "rgb_batch_size_limit")
            reject(lambda: loader.load_batch([]), "rgb_batch_size_limit")
            reject(lambda: bounded.load_batch([0, 1]), "rgb_batch_tensor_byte_limit")
            require(not paths and not decodes, "batch_rejected_after_png_io")

    def transparent_png(f):
        rgba = bytes(v for color in zip(*[iter(f.expected[1, "left"])] * 3) for v in (*color, 0))
        buffer = io.BytesIO()
        with Image.frombytes("RGBA", (8, 6), rgba) as image: image.save(buffer, format="PNG")
        loader = f.loader(f.replace_png(1, "left", buffer.getvalue()))
        reject(lambda: loader[1], "rgb_transparent_pixels_not_supported")

    def truncated_png(f):
        loader = f.loader(f.replace_png(0, "left", f.original[0, "left"][:-8]))
        reject(lambda: loader[0], "rgb_png_decode_or_file_failure")

    def handles_and_sample_reread(f):
        loader = f.loader()
        with trace_io() as (paths, handles, _):
            first, second = loader[0]["images"], loader[0]["images"]
            require(len(paths) == 4 and torch.equal(first, second), "sample_cached_or_reread_changed")
            raw = f.original[0, "left"]
            f.path(0, "left").write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            reject(lambda: loader[0], "rgb_file_hash_mismatch")
            require(handles and all(h.closed for h in handles), "success_or_failure_handle_open")
            f.path(0, "left").write_bytes(raw)
            require(torch.equal(loader[0]["images"], first), "reread_after_failure")

    functions = locals()
    for record in report["cases"]:
        started = time.monotonic(); expected_errors.clear()
        try:
            with tempfile.TemporaryDirectory(prefix="rgb_case_", dir=output) as directory:
                require(Path(directory).resolve().is_relative_to(output), "fixture_outside_output")
                fixture = Fixture(Path(directory))
                functions[record["name"]](fixture)
                total = sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
                require(total <= FIXTURE_LIMIT, "fixture_disk_limit")
                report["peak_fixture_bytes_observed"] = max(report["peak_fixture_bytes_observed"], total)
            record["status"] = "passed"
        except Exception as exc:
            record.update(status="failed", exception=exception_info(exc))
        record.update(elapsed_seconds=time.monotonic()-started, expected_rejections=list(expected_errors))
        print(record["name"], record["status"], flush=True)
        report["completed_cases"] += 1
        save_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="새 보고서 디렉터리; 재사용 금지")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / "outputs")):
        parser.error("새 임시 경로 또는 프로젝트 outputs 아래 새 경로를 지정하세요. 기존 경로를 덮지 않습니다.")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema_version": "local_rgb_cpu_check_v1", "check_api": LOCAL_RGB_CHECK_API,
              "status": "running", "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "required_cases": len(CASE_NAMES), "completed_cases": 0,
              "cases": [{"name": name, "status": "not_run"} for name in CASE_NAMES],
              "data_kind": "synthetic_fixture", "peak_fixture_bytes_observed": 0,
              "budgets": {"fixture_bytes": FIXTURE_LIMIT, "report_bytes": REPORT_LIMIT,
                          "external_subprocess_timeout_seconds": 120},
              "limitations": ["small CPU contract check only", "no RSS/performance benchmark", "no real data/model/training/Drive"]}
    try:
        report["source"] = source_info()
        require(report["source"]["files_sha256_lf"]["rewind_m1/local_rgb.py"] == LOADER_SHA256_LF, "production_loader_hash_mismatch")
        for package in ("torch", "Pillow"):
            try: report["environment"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: report["environment"][package] = None
        save_report(output, report)
        run_checks(output, report)
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    except Exception as exc:
        report.update(status="failed", setup_exception=exception_info(exc))
    report["elapsed_seconds"] = time.monotonic()-started
    report["fixture_directories_remaining"] = [p.name for p in output.glob("rgb_case_*")]
    if report["fixture_directories_remaining"]:
        report["status"] = "failed"
    save_report(output, report)
    print(json.dumps({"status": report["status"], "cases": report["completed_cases"],
                      "required": report["required_cases"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

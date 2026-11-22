"""Mount the destination and retain results from the pinned CPU worker."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


def publish_tail_file(path, raw, cap):
    if not 0 < len(raw) <= cap:
        raise ValueError(f"Invalid result size: {path.name}")
    with path.open("xb") as handle:
        if handle.write(raw) != len(raw):
            raise OSError(f"Short Drive write: {path.name}")
    with path.open("rb") as handle:
        actual = handle.read(cap + 1)
    expected = hashlib.sha256(raw).hexdigest()
    if len(actual) != len(raw) or hashlib.sha256(actual).hexdigest() != expected:
        raise OSError(f"Drive readback failed: {path.name}")
    return {"bytes": len(raw), "sha256": expected}


def publish_tail_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return publish_tail_file(path, raw, 64 * 1024)


def small_tail_file(path, cap):
    if not 0 < path.stat().st_size <= cap:
        raise ValueError(f"Invalid worker file size: {path.name}")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    if not 0 < len(raw) <= cap:
        raise ValueError(f"Invalid worker file bytes: {path.name}")
    return raw


def run_tail_notebook(input_zip, project, worker_source, pins, *, mock_mount=None):
    started = time.monotonic()
    limits = pins["limits"]

    def remaining():
        left = limits["total_seconds"] - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Clip report total time exceeded")
        return left

    input_zip = Path(input_zip)
    if mock_mount is None and (input_zip.parent != Path("/content") or input_zip.name != "v210_groups.zip"):
        raise ValueError("Upload v210_groups.zip directly to /content")
    if input_zip.is_symlink() or not input_zip.is_file() or input_zip.stat().st_size != pins["input_bytes"]:
        raise ValueError("Expected the fixed v210_groups.zip")
    if hashlib.sha256(worker_source.encode()).hexdigest() != pins["worker_sha256"]:
        raise ValueError("Embedded worker SHA mismatch")
    mount = Path("/content/drive") if mock_mount is None else mock_mount
    if mock_mount is None:
        from google.colab import drive

        drive.mount(str(mount), timeout_ms=120_000)
        if not os.path.ismount(mount):
            raise OSError("Drive mount is not visible")
    remaining()
    my_drive = (mount / "MyDrive").resolve(strict=True)
    project = Path(project).resolve(strict=True)
    if project == my_drive or not project.is_relative_to(my_drive):
        raise ValueError("Project must be below the mounted MyDrive")
    root = project / "reports"
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("Expected project reports directory")
    parent = root / "v210_clip_errors_20261122"
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Drive output parent changed or points elsewhere")
    destination = parent / f"run-{uuid.uuid4().hex}"
    destination.mkdir()
    result = {"schema": "rewind_v210_clip_errors_receipt_v1", "status": "failed",
              "run_directory": str(destination), "input_sha256": pins["input_sha256"],
              "worker_sha256": pins["worker_sha256"], "limits": limits, "files": {}}
    result["start_receipt"] = publish_tail_json(destination / "started.json",
                                                {**result, "status": "running"})
    print("Drive output:", destination, flush=True)
    stderr = b""
    try:
        with tempfile.TemporaryDirectory(prefix="v210-tail-", dir="/content" if mock_mount is None else None) as tmp:
            scratch = Path(tmp)
            if shutil.disk_usage(scratch).free < limits["scratch_free_bytes"]:
                raise OSError("Insufficient local scratch space")
            script = scratch / "worker.py"
            script.write_bytes(worker_source.encode())
            output = scratch / "output"
            command = [sys.executable, str(script), "--input", str(input_zip), "--output", str(output)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            try:
                run = subprocess.run(command, cwd=scratch, env=env, capture_output=True,
                                     timeout=min(limits["worker_seconds"], remaining()))
            except subprocess.TimeoutExpired as exc:
                stderr = (exc.stderr or b"")[-16 * 1024:]
                raise
            result["returncode"] = run.returncode
            stderr = run.stderr[-16 * 1024:]
            if run.returncode != 0:
                raise RuntimeError(f"Clip worker exited with status {run.returncode}")
            if len(run.stdout) > 16 * 1024:
                raise ValueError("Clip worker stdout exceeded report cap")
            report = json.loads(run.stdout)
            if report["status"] != "passed" or report["counts"] != pins["counts"] or report["files"].keys() != set(pins["output_names"]):
                raise ValueError("Clip worker receipt shape mismatch")
            for name in pins["output_names"]:
                remaining()
                cap = limits["result_file_bytes"]
                data = small_tail_file(output / name, cap)
                expected = report["files"][name]
                if expected != {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
                    raise ValueError(f"Clip worker output SHA mismatch: {name}")
                result["files"][name] = publish_tail_file(destination / name, data, cap)
            remaining()
            result["status"] = "passed"
            result["counts"] = report["counts"]
    except BaseException as exc:
        if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)):
            result["status"] = "timed_out"
        if isinstance(exc, KeyboardInterrupt):
            result["status"] = "stopped"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        if stderr:
            result["files"]["worker-stderr.txt"] = publish_tail_file(destination / "worker-stderr.txt", stderr, 16 * 1024)
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        receipt = publish_tail_json(destination / "result.json", result)
        print(json.dumps({"path": str(destination / "result.json"), "status": result["status"], **receipt}), flush=True)
    if result["status"] != "passed":
        raise RuntimeError(f"Review {destination / 'result.json'}")
    return result

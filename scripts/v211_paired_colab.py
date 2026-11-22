"""Run and retain the pinned v2.1.1 paired CPU report in Colab."""

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

if "publish_tail_file" not in globals():
    from scripts.v210_tail_colab import publish_tail_file, publish_tail_json, small_tail_file


def run_paired_notebook(project, mode, modules, pins, *, mock_mount=None, mock_upload=None):
    started = time.monotonic()
    limits = pins["limits"]

    def remaining():
        seconds = limits["total_seconds"] - (time.monotonic() - started)
        if seconds <= 0:
            raise TimeoutError("Paired Colab time limit")
        return seconds

    if mode not in ("drive", "zip"):
        raise ValueError("INPUT_MODE must be drive or zip")
    if modules.keys() != pins["modules"].keys() or any(
            hashlib.sha256(source.encode()).hexdigest() != pins["modules"][name] for name, source in modules.items()):
        raise ValueError("Paired worker module SHA mismatch")
    upload = Path("/content") if mock_upload is None else mock_upload
    if mode == "zip" and not all((upload / name).is_file() for name in ("v210_groups.zip", "v211_groups.zip")):
        raise FileNotFoundError("Upload both recovered ZIP files to /content")
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
        raise ValueError("Project must be below mounted MyDrive")
    reports = project / "reports"
    if reports.resolve(strict=True) != reports or not reports.is_dir():
        raise ValueError("Expected a direct project reports directory")
    parent = reports / "v211_paired_clips_20261122"
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Unsafe result parent")
    output = parent / f"run-{uuid.uuid4().hex}"
    output.mkdir()
    base, variants = ((reports / "v210_eval128_a_v1", reports / "v211_body_head128_a_v1") if mode == "drive"
                      else (upload / "v210_groups.zip", upload / "v211_groups.zip"))
    receipt = {"schema": "rewind_v211_paired_colab_result_v1", "status": "running",
               "mode": mode, "base": str(base), "variants": str(variants), "limits": limits, "modules": pins["modules"],
               "variant_summary_sha256": pins["seed_summary_sha256"], "files": {}}
    receipt["start"] = publish_tail_json(output / "started.json", receipt)
    print("Paired Drive output:", output, flush=True)
    log = b""
    try:
        with tempfile.TemporaryDirectory(prefix="v211-paired-", dir="/content" if mock_mount is None else None) as tmp:
            scratch = Path(tmp)
            if shutil.disk_usage(scratch).free < limits["scratch_free_bytes"]:
                raise OSError("Insufficient CPU scratch")
            for name, source in modules.items():
                (scratch / name).write_bytes(source.encode())
            work = scratch / "results"
            cmd = [sys.executable, str(scratch / "v211_paired.py"), "--base", str(base),
                   "--variants", str(variants), "--output", str(work)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            try:
                result = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True,
                                        timeout=min(limits["worker_seconds"], remaining()))
            except subprocess.TimeoutExpired as exc:
                log = (exc.stderr or b"")[-16 * 1024:]
                raise
            receipt["returncode"] = result.returncode
            log = result.stderr[-16 * 1024:]
            if result.returncode or len(result.stdout) > 16 * 1024:
                raise RuntimeError(f"Paired CPU worker exit status {result.returncode}")
            worker = json.loads(result.stdout)
            if (worker["status"] != "passed" or worker["rows"] != 1263
                    or worker["files"].keys() != set(pins["outputs"])):
                raise ValueError("Invalid worker scope or output list")
            receipt["logical_read_observed_bytes"] = worker["read_bytes"]
            for name in pins["outputs"]:
                remaining()
                raw = small_tail_file(work / name, limits["result_file_bytes"])
                if worker["files"][name] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                    raise ValueError(f"Paired output pin mismatch: {name}")
                receipt["files"][name] = publish_tail_file(output / name, raw, limits["result_file_bytes"])
            remaining()
            receipt["status"] = "passed"
            receipt["rows"] = worker["rows"]
    except BaseException as exc:
        receipt["status"] = ("timed_out" if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError))
                             else "stopped" if isinstance(exc, KeyboardInterrupt) else "failed")
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        if log:
            receipt["files"]["worker-stderr.txt"] = publish_tail_file(output / "worker-stderr.txt", log, 16 * 1024)
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        recovered = publish_tail_json(output / "result.json", receipt)
        print(json.dumps({"path": str(output / "result.json"), "status": receipt["status"], **recovered}), flush=True)
    if receipt["status"] != "passed":
        raise RuntimeError(f"Review {output / 'result.json'}")
    return receipt

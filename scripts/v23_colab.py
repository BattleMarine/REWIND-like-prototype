"""Standard-library Colab supervisor embedded in the v2.3 notebook."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def run_v23_notebook(project_setting, pins, *, mock_mount=None, mock_upload=None):
    started = time.monotonic()

    def time_left():
        return pins["limits"]["seconds"] - (time.monotonic() - started)

    mount = Path("/content/drive") if mock_mount is None else mock_mount
    upload = Path("/content") if mock_upload is None else mock_upload
    source_zip, toolkit = upload / pins["base"]["source_name"], upload / pins["toolkit"]["name"]
    if not source_zip.is_file() or not toolkit.is_file():
        raise FileNotFoundError(f"Upload {source_zip.name} and {toolkit.name}")
    if mock_mount is None:
        from google.colab import drive

        drive.mount(str(mount), timeout_ms=120_000)
        if not os.path.ismount(mount):
            raise OSError("Drive mount not visible")
    mydrive = (mount / "MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    if project == mydrive or not project.is_relative_to(mydrive):
        raise ValueError("Expected project below MyDrive")
    reports = project / "reports"
    if not reports.is_dir() or reports.resolve(strict=True) != reports:
        raise ValueError("Expected project's direct reports directory")
    parent = reports / pins["output_folder"]
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Output parent escapes project")
    target = parent / f"run-{uuid.uuid4().hex}"
    target.mkdir()
    result = {"schema": "rewind_v23_invocation_v1", "status": "running", "files": {}, "limits": pins["limits"],
              "source_sha256": pins["base"]["source_sha256"], "toolkit_sha256": pins["toolkit"]["sha256"],
              "path": str(target), "process_returncode": None}
    result["start"] = publish_tail_json(target / "started.json", result)
    print("Drive v2.3 result:", target, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rewind-v23-", dir=str(upload)) as directory:
            work = Path(directory)
            source, tools = work / "source", work / "tools"
            extract_pinned_zip(source_zip, pins["base"]["source_sha256"], source, max_zip_bytes=32 * 1024**2,
                               max_output_bytes=192 * 1024**2, max_members=3000, revision=pins["base"]["revision"])
            extract_pinned_zip(toolkit, pins["toolkit"]["sha256"], tools, max_zip_bytes=2 * 1024**2,
                               max_output_bytes=4 * 1024**2, max_members=16)
            import hashlib

            for name, expected in pins["modules"].items():
                raw = (tools / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(f"Overlay module SHA mismatch: {name}")
                with (source / "scripts" / name).open("xb") as stream:
                    stream.write(raw)
            pin_file = work / "pins.json"
            pin_file.write_text(json.dumps(pins), encoding="utf-8")
            command = [sys.executable, "-u", "-m", "scripts.v23_run", "--project", str(project),
                       "--work", str(work), "--pins", str(pin_file), "--started", str(started)]
            remaining = time_left() - 35
            if remaining <= 0:
                raise TimeoutError("v2.3 preparation exceeded budget")
            error = None
            process_log = work / "worker.log"
            with process_log.open("xb") as log:
                with subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT) as child:
                    try:
                        result["process_returncode"] = child.wait(timeout=remaining)
                    except BaseException as exc:
                        error = exc
                        if child.poll() is None:
                            child.send_signal(signal.SIGINT)
                        try:
                            result["process_returncode"] = child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            result["process_returncode"] = child.wait(timeout=10)
            worker_path = work / "worker-result.json"
            if worker_path.is_file():
                worker_bytes = small_tail_file(worker_path, pins["limits"]["result_file_bytes"])
                worker = json.loads(worker_bytes)
                if worker["schema"] != "rewind_v23_worker_v1":
                    raise ValueError("Unexpected worker receipt")
                result["files"][worker_path.name] = publish_tail_file(target / worker_path.name, worker_bytes,
                                                                     pins["limits"]["result_file_bytes"])
                for name in ("latency.json", "quality.json"):
                    if name in worker["files"]:
                        raw = small_tail_file(work / name, pins["limits"]["result_file_bytes"])
                        if worker["files"][name] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                            raise ValueError(f"Worker evidence SHA mismatch: {name}")
                        result["files"][name] = publish_tail_file(target / name, raw, pins["limits"]["result_file_bytes"])
                if (error is None and result["process_returncode"] == 0 and worker["status"] == "completed"
                        and set(result["files"]) == {"worker-result.json", "latency.json", "quality.json"}):
                    if time_left() <= 0:
                        raise TimeoutError("v2.3 output verification time")
                    result["status"] = "completed"
            if result["status"] != "completed":
                with process_log.open("rb") as log:
                    log.seek(max(0, process_log.stat().st_size - 32 * 1024))
                    tail = log.read(32 * 1024)
                if tail:
                    result["files"]["worker.log"] = publish_tail_file(target / "worker.log", tail, 32 * 1024)
                if error:
                    raise error
                raise RuntimeError(f"v2.3 worker not completed ({result['process_returncode']})")
    except BaseException as exc:
        result["status"] = ("stopped" if isinstance(exc, KeyboardInterrupt)
                            else "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed")
        result["error"] = {"type": type(exc).__name__, "reason": str(exc)[:800]}
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        receipt = publish_tail_json(target / "result.json", result)
        print(json.dumps({"path": str(target / "result.json"), "status": result["status"], **receipt}), flush=True)
    if result["status"] != "completed":
        raise RuntimeError(f"Review {target / 'result.json'}")
    return result

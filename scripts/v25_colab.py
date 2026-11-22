"""Standard-library controller embedded in the fixed v2.5 notebook."""

import base64
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
import zlib


def run_v25_notebook(project_setting, pins, embedded):
    from google.colab import drive

    started = time.monotonic()
    limits = pins["limits"]
    upload, mount = Path("/content"), Path("/content/drive")
    source_zip = upload/pins["base"]["source_name"]
    if source_zip.is_symlink() or not source_zip.is_file():
        raise FileNotFoundError(f"Upload {source_zip.name} directly to /content")
    drive.mount(str(mount), timeout_ms=120_000)
    if not os.path.ismount(mount):
        raise OSError("Drive mount not visible")
    mydrive = (mount/"MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    reports = project/"reports"
    if (project == mydrive or not project.is_relative_to(mydrive) or not reports.is_dir()
            or reports.resolve(strict=True) != reports):
        raise ValueError("Expected project's direct reports directory under MyDrive")
    parent = reports/pins["output_folder"]
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Drive output parent changed or points elsewhere")
    target = parent/f"run-{uuid.uuid4().hex}"
    target.mkdir()
    result = {"schema": "rewind_v25_invocation_v1", "status": "failed", "files": {},
              "limits": limits, "path": str(target), "source_sha256": pins["base"]["source_sha256"]}
    result["start"] = publish_tail_json(target/"started.json", {**result, "status": "running"})
    print("Drive v2.5 output:", target, flush=True)
    work = None
    try:
        work = Path(tempfile.mkdtemp(prefix="rewind-v25-", dir=upload))
        source = work/"source"
        extract_pinned_zip(source_zip, pins["base"]["source_sha256"], source, max_zip_bytes=32*1024**2,
                           max_output_bytes=192*1024**2, max_members=3000, revision=pins["base"]["revision"])
        for name, expected in pins["modules"].items():
            raw = zlib.decompress(base64.b64decode(embedded[name], validate=True))
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Overlay module SHA mismatch: {name}")
            with (source/"scripts"/name).open("xb") as stream:
                stream.write(raw)
        for filename, pin in (("selected-head.pt", pins["head"]), ("v23-quality.json", pins["baseline"])):
            raw = zlib.decompress(base64.b64decode(embedded[filename], validate=True))
            if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
                raise ValueError(f"Embedded evidence differs: {filename}")
            with (work/filename).open("xb") as stream:
                stream.write(raw)
        pin_file = work/"pins.json"
        pin_file.write_text(json.dumps(pins), encoding="utf-8")
        command = [sys.executable, "-u", "-m", "scripts.v25_run", "--project", str(project), "--work", str(work),
                   "--pins", str(pin_file), "--started", str(started)]
        timeout = limits["seconds"] - (time.monotonic()-started) - limits["supervisor_reserve_seconds"]
        if timeout <= 0:
            raise TimeoutError("v2.5 preparation exceeded budget")
        log_path = work/"worker.log"
        worker_error = None
        with log_path.open("xb") as log:
            try:
                result["process_returncode"] = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                                               timeout=timeout).returncode
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                worker_error = exc
        cap = limits["result_file_bytes"]
        worker_path = work/"worker-result.json"
        if worker_path.is_file():
            raw = small_tail_file(worker_path, cap)
            worker = json.loads(raw)
            if worker["schema"] != "rewind_v25_worker_v1":
                raise ValueError("Unexpected worker receipt")
            result["files"][worker_path.name] = publish_tail_file(target/worker_path.name, raw, cap)
            for name in pins["result_names"]:
                if name in worker["files"]:
                    raw = small_tail_file(work/name, cap)
                    if worker["files"][name] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                        raise ValueError(f"Worker evidence SHA mismatch: {name}")
                    result["files"][name] = publish_tail_file(target/name, raw, cap)
            if (worker_error is None and result["process_returncode"] == 0 and worker["status"] == "completed"
                    and set(result["files"]) == {worker_path.name, *pins["result_names"]}):
                if time.monotonic()-started >= limits["seconds"]:
                    raise TimeoutError("v2.5 output verification exceeded budget")
                result["status"] = "completed"
        if result["status"] != "completed":
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size-32*1024))
                tail = log.read(32*1024)
            if tail:
                result["files"]["worker.log"] = publish_tail_file(target/"worker.log", tail, 32*1024)
            if worker_error is not None:
                raise worker_error
            raise RuntimeError("v2.5 worker did not complete")
    except BaseException as exc:
        result["status"] = ("stopped" if isinstance(exc, KeyboardInterrupt) else
                            "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed")
        result["error"] = {"type": type(exc).__name__, "reason": str(exc)[:800]}
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        if result["status"] != "completed" and work is not None:
            result["local_work_path"] = str(work)
            print("Local v2.5 work retained:", work, flush=True)
        try:
            receipt = publish_tail_json(target/"result.json", result)
        except BaseException as exc:
            if work is not None:
                result.update(status="failed", drive_receipt_error={"type": type(exc).__name__, "reason": str(exc)[:800]},
                              local_work_path=str(work))
                print("Local v2.5 work retained:", work, flush=True)
                try:
                    (work/"retention-error.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                except OSError as record_error:
                    print("Local receipt could not be written:", record_error, flush=True)
            raise
        print(json.dumps({"path": str(target/"result.json"), "status": result["status"], **receipt}), flush=True)
    if result["status"] != "completed":
        raise RuntimeError(f"Review {target/'result.json'}")
    shutil.rmtree(work, ignore_errors=True)
    return result

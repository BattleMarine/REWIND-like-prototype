"""Standard-library supervisor for v2.4 numerical-only and paired pilot modes."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def stream_v24_state(source, destination, expected, cap, check):
    if not 0 < expected["bytes"] == source.stat().st_size <= cap:
        raise ValueError("Pilot checkpoint size mismatch")
    incoming = destination.with_suffix(".partial")
    count, current = 0, hashlib.sha256()
    try:
        with source.open("rb") as original, incoming.open("xb") as copy:
            while block := original.read(1024**2):
                check()
                count += len(block)
                if count > expected["bytes"] or copy.write(block) != len(block):
                    raise OSError("Pilot checkpoint grew or Drive write was short")
                current.update(block)
        if {"bytes": count, "sha256": current.hexdigest()} != expected:
            raise ValueError("Pilot checkpoint source SHA mismatch")
        verified_count, verified = 0, hashlib.sha256()
        with incoming.open("rb") as saved:
            while block := saved.read(1024**2):
                check()
                verified_count += len(block)
                if verified_count > expected["bytes"]:
                    raise OSError("Drive checkpoint copy grew")
                verified.update(block)
        if {"bytes": verified_count, "sha256": verified.hexdigest()} != expected:
            raise IOError("Drive checkpoint readback SHA mismatch")
        incoming.rename(destination)
        return expected
    finally:
        incoming.unlink(missing_ok=True)


def run_v24_notebook(project_setting, mode, pins, *, mock_mount=None, mock_upload=None):
    start = time.monotonic()
    if mode not in ("numerical", "pilot"):
        raise ValueError("MODE must be numerical or pilot")
    limits = pins["limits"]
    seconds = limits["pilot_seconds"] if mode == "pilot" else limits["numerical_seconds"]

    def remaining():
        left = seconds - (time.monotonic() - start)
        if left <= 0:
            raise TimeoutError("v2.4 invocation deadline")
        return left

    upload = Path("/content") if mock_upload is None else mock_upload
    mount = Path("/content/drive") if mock_mount is None else mock_mount
    source_zip, kit = upload / pins["base"]["source_name"], upload / pins["toolkit"]["name"]
    if not source_zip.is_file() or not kit.is_file():
        raise FileNotFoundError(f"Upload {source_zip.name} and {kit.name}")
    if mock_mount is None:
        from google.colab import drive

        drive.mount(str(mount), timeout_ms=120_000)
        if not os.path.ismount(mount):
            raise OSError("Drive mount not visible")
    root = (mount / "MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    if project == root or not project.is_relative_to(root):
        raise ValueError("Expected project below MyDrive")
    reports = project / "reports"
    if not reports.is_dir() or reports.resolve(strict=True) != reports:
        raise ValueError("Expected direct project reports parent")
    parent = reports / pins["output_folder"]
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Output parent escapes project")
    output = parent / f"run-{mode}-{uuid.uuid4().hex}"
    output.mkdir()
    result = {"schema": "rewind_v24_invocation_v1", "status": "running", "mode": mode, "files": {},
              "seconds_reserved": seconds, "limits": limits, "path": str(output),
              "source_sha256": pins["base"]["source_sha256"], "kit_sha256": pins["toolkit"]["sha256"]}
    result["start"] = publish_tail_json(output / "started.json", result)
    print("Drive v2.4 result:", output, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="v24-pilot-", dir=upload) as tmp:
            work = Path(tmp)
            source, overlays = work / "source", work / "overlay"
            extract_pinned_zip(source_zip, pins["base"]["source_sha256"], source, max_zip_bytes=32 * 1024**2,
                               max_output_bytes=192 * 1024**2, max_members=3000, revision=pins["base"]["revision"])
            extract_pinned_zip(kit, pins["toolkit"]["sha256"], overlays, max_zip_bytes=2 * 1024**2,
                               max_output_bytes=4 * 1024**2, max_members=16)
            for name, expected in pins["modules"].items():
                raw = (overlays / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(f"Overlay SHA mismatch: {name}")
                with (source / "scripts" / name).open("xb") as stream:
                    stream.write(raw)
            pin_file = work / "pins.json"
            pin_file.write_text(json.dumps(pins), encoding="utf-8")
            cmd = [sys.executable, "-u", "-m", "scripts.v24_run", "--project", str(project), "--work", str(work),
                   "--pins", str(pin_file), "--started", str(start)]
            if mode == "pilot":
                cmd.append("--pilot")
            window = remaining() - (225 if mode == "pilot" else 30)
            if window <= 0:
                raise TimeoutError("No worker time remains before publication reservation")
            process_error = None
            log_path = work / "worker.log"
            with log_path.open("xb") as log:
                with subprocess.Popen(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT) as process:
                    try:
                        result["returncode"] = process.wait(timeout=window)
                    except BaseException as exc:
                        process_error = exc
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                        try:
                            result["returncode"] = process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            result["returncode"] = process.wait(timeout=10)
            small = {"numerical.json", "pilot-plan.json", "x0-pilot.json", "v-pilot.json", "pilot-summary.json"}
            states = {"x0-pilot-final.pt", "v-pilot-final.pt"}
            expected_files = small | states if mode == "pilot" else {"numerical.json"}
            cap = limits["report_bytes"]
            marker = work / "worker-result.json"
            if marker.is_file():
                raw = small_tail_file(marker, cap)
                receipt = json.loads(raw)
                if receipt["schema"] != "rewind_v24_worker_v1" or not receipt["files"].keys() <= expected_files:
                    raise ValueError("Unexpected v2.4 worker files")
                result["files"][marker.name] = publish_tail_file(output / marker.name, raw, cap)
                for name, pin in receipt["files"].items():
                    remaining()
                    if name in states:
                        result["files"][name] = stream_v24_state(work / name, output / name, pin, limits["state_bytes"], remaining)
                    else:
                        raw = small_tail_file(work / name, cap)
                        if pin != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                            raise ValueError(f"v2.4 worker output SHA mismatch: {name}")
                        result["files"][name] = publish_tail_file(output / name, raw, cap)
                expected_status = "completed_pilot" if mode == "pilot" else "completed_numerical"
                if (receipt["status"] == expected_status and not result["returncode"] and process_error is None
                        and expected_files == receipt["files"].keys()):
                    remaining()
                    result["status"] = expected_status
            if result["status"] == "running":
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 32 * 1024))
                    tail = log.read(32 * 1024)
                if tail:
                    result["files"]["worker.log"] = publish_tail_file(output / "worker.log", tail, 32 * 1024)
                if process_error is not None:
                    raise process_error
                raise RuntimeError(f"v2.4 worker exit status {result['returncode']}")
    except BaseException as exc:
        result["status"] = ("stopped" if isinstance(exc, KeyboardInterrupt) else
                            "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed")
        result["error"] = {"type": type(exc).__name__, "reason": str(exc)[:800]}
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        receipt = publish_tail_json(output / "result.json", result)
        print(json.dumps({"path": str(output / "result.json"), "status": result["status"], **receipt}), flush=True)
    if not result["status"].startswith("completed_"):
        raise RuntimeError(f"Review {output / 'result.json'}")
    return result

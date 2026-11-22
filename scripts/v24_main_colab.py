"""Colab controller for one approved paired-comparison segment."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


def run_v24_main(project_setting, arm, seed, before, after, resume, resume_sha, protocol_sha, extension, pins):
    started = time.monotonic()
    protocol = pins["protocol"]
    budgets = protocol["budgets"]
    if protocol_sha != pins["protocol_sha256"] or not protocol_sha:
        raise ValueError("Use the protocol SHA only after execution approval")
    if (arm not in ("x0", "v") or seed not in protocol["initial_seeds"] + protocol["conditional_seeds"]
            or (seed in protocol["conditional_seeds"] and not extension)
            or (before, after) not in list(zip(protocol["boundaries"], protocol["boundaries"][1:]))
            or bool(before) != bool(resume and resume_sha) or bool(resume) != bool(resume_sha)):
        raise ValueError("Invalid arm, seed, resume, or evaluation boundary")

    def remaining():
        seconds = budgets["seconds"] - (time.monotonic()-started)
        if seconds <= 0:
            raise TimeoutError("v2.4 main invocation reached its wall limit")
        return seconds

    upload = Path("/content")
    source_zip = upload / pins["base"]["source_name"]
    kit = upload / pins["toolkit"]["name"]
    for path, expected in ((source_zip, pins["source_upload_bytes"]), (kit, pins["toolkit"]["bytes"])):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"Missing pinned upload: {path.name}")
    from google.colab import drive
    mount = upload / "drive"
    if not os.path.ismount(mount):
        drive.mount(str(mount), timeout_ms=120_000)
    if not os.path.ismount(mount):
        raise OSError("Drive mount not visible")
    root = (mount / "MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    reports = project / "reports"
    if project == root or not project.is_relative_to(root) or reports.resolve(strict=True) != reports or not reports.is_dir():
        raise ValueError("Expected the project reports folder under MyDrive")
    parent = reports / "v24_main128_a_candidate"
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("Output folder redirect")
    output = parent / f"run-{uuid.uuid4().hex}"
    output.mkdir()
    result = {"schema": "rewind_v24_main_invocation_v1", "status": "running", "files": {},
              "protocol_sha256": protocol_sha, "seed": seed, "arm": arm, "before": before, "after": after,
              "resume_result_sha256": resume_sha or None, "budgets": budgets, "path": str(output)}
    publish_tail_json(output / "started.json", result)
    print("v2.4 main output:", output, flush=True)
    work = None
    can_retain_local = False
    published = False
    try:
        with tempfile.TemporaryDirectory(dir=upload, prefix="v24-main-setup-") as directory:
            work = Path(tempfile.mkdtemp(dir=upload, prefix="v24-main-recovery-"))
            install = Path(directory)
            source = install / "source"
            overlay = install / "overlay"
            extract_pinned_zip(source_zip, pins["base"]["source_sha256"], source, max_zip_bytes=32*1024**2,
                               max_output_bytes=192*1024**2, max_members=3000, revision=pins["base"]["revision"])
            extract_pinned_zip(kit, pins["toolkit"]["sha256"], overlay, max_zip_bytes=2*1024**2,
                               max_output_bytes=4*1024**2, max_members=32)
            for name, expected in pins["modules"].items():
                raw = (overlay / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(f"Module SHA mismatch: {name}")
                with (source / "scripts" / name).open("xb") as stream:
                    stream.write(raw)
            pin_file = work / "pins.json"
            pin_file.write_text(json.dumps(pins), encoding="utf-8")
            cmd = [sys.executable, "-u", "-m", "scripts.v24_main_run", "--project", str(project), "--work", str(work),
                   "--pins", str(pin_file), "--started", str(started), "--arm", arm, "--seed", str(seed),
                   "--before", str(before), "--after", str(after), "--protocol-sha", protocol_sha]
            if extension:
                cmd.append("--extension")
            if resume:
                cmd += ["--resume", resume, "--resume-sha", resume_sha]
            timeout = remaining() - budgets["publish_reserve_seconds"]
            if timeout <= 0:
                raise TimeoutError("Insufficient worker time before publication reserve")
            failure = None
            log_path = work / "worker.log"
            with log_path.open("xb") as log, subprocess.Popen(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT) as worker:
                try:
                    result["returncode"] = worker.wait(timeout=timeout)
                except BaseException as exc:
                    failure = exc
                    if worker.poll() is None:
                        worker.send_signal(signal.SIGINT)
                    try:
                        result["returncode"] = worker.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        result["returncode"] = worker.wait(timeout=10)
            marker = work / "worker-result.json"
            if marker.is_file():
                cap = budgets["report_bytes"]
                raw = small_tail_file(marker, cap)
                receipt = json.loads(raw)
                can_retain_local = (receipt.get("status") == "segment_complete" and result["returncode"] == 0
                                    and failure is None and set(receipt.get("files", ())) == {"body-state.pt", "segment.json"})
                result["files"][marker.name] = publish_tail_file(output / marker.name, raw, cap)
                if receipt["schema"] != "rewind_v24_main_worker_v1" or any(receipt.get(key) != result[key] for key in
                        ("arm", "seed", "before", "after", "protocol_sha256")):
                    raise ValueError("Worker receipt invocation mismatch")
                if (receipt["status"] == "segment_complete" and result["returncode"] == 0 and failure is None
                        and set(receipt["files"]) == {"body-state.pt", "segment.json"}):
                    state, segment = "body-state.pt", "segment.json"
                    raw = small_tail_file(work / segment, cap)
                    if receipt["files"][segment] != {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
                        raise ValueError("Segment report SHA mismatch")
                    result["files"][segment] = publish_tail_file(output / segment, raw, cap)
                    result["files"][state] = stream_v24_state(work / state, output / state, receipt["files"][state],
                                                            budgets["state_bytes"], remaining)
                    result.update(status="segment_complete", binding_sha256=receipt["binding_sha256"],
                                  common_sha256=receipt["common_sha256"])
            if result["status"] != "segment_complete":
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size-32768))
                    tail = log.read(32768)
                if tail:
                    result["files"]["worker.log"] = publish_tail_file(output / "worker.log", tail, 32768)
                if failure is not None:
                    raise failure
                raise RuntimeError(f"Worker did not complete segment, exit status {result['returncode']}")
    except BaseException as exc:
        result["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else (
            "timed_out" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed")
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:800]}
    finally:
        if work is not None and can_retain_local and result["status"] != "segment_complete":
            result["local_recovery"] = str(work)
        result["elapsed_seconds"] = time.monotonic()-started
        try:
            pin = publish_tail_json(output / "result.json", result)
            published = True
            print(json.dumps({"path": str(output / "result.json"), "status": result["status"], **pin}), flush=True)
        finally:
            if work is not None and can_retain_local and (not published or result["status"] != "segment_complete"):
                print("v2.4 local recovery directory:", work, flush=True)
            elif work is not None:
                shutil.rmtree(work)

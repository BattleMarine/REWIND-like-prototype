"""Standard-library Colab controller embedded in the fixed CPU notebook."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid


def publish_cpu_file(path, raw, limit=2 * 1024**2):
    if len(raw) > limit:
        raise ValueError(f"CPU result publish size: {path.name}")
    with path.open("xb") as handle:
        if handle.write(raw) != len(raw):
            raise OSError(f"CPU result short write: {path.name}")
    with path.open("rb") as handle:
        returned = handle.read(limit + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(returned) != len(raw) or hashlib.sha256(returned).hexdigest() != digest:
        raise OSError(f"CPU result readback failed: {path}")
    return {"name": path.name, "bytes": len(raw), "sha256": digest}


def publish_cpu_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return publish_cpu_file(path, raw)


def read_cpu_small(path, limit):
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"CPU process output size: {path.name}")
    return raw


def run_cpu_child(job, limits, kit, project, scratch, destination, *, command=None):
    local = scratch / job
    target = destination / job
    target.mkdir()
    process_log = scratch / f"{job}.log"
    began = time.monotonic()
    rc = None
    process_error = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join((str(kit), str(kit / "frozen"))))
    args = command or [sys.executable, "-u", "-m", "scripts.v2_tuning_cpu_worker", job,
                       "--kit", str(kit), "--project", str(project), "--workspace", str(local)]
    try:
        with process_log.open("xb") as log:
            with subprocess.Popen(args, cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                try:
                    rc = process.wait(timeout=limits["seconds"])
                except BaseException:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=15)
                    raise
    except BaseException as exc:
        process_error = {"type": type(exc).__name__, "message": str(exc)[:1600]}
    files, report = {}, None
    for name in ("usage.json", "report.json", "trial-plan.json"):
        path = local / name
        if path.is_file():
            raw = read_cpu_small(path, 2 * 1024**2)
            files[name] = publish_cpu_file(target / name, raw)
            if name == "report.json":
                report = json.loads(raw)
    if process_log.is_file():
        with process_log.open("rb") as log:
            log.seek(max(0, process_log.stat().st_size - 64 * 1024))
            files["process.log"] = publish_cpu_file(target / "process.log", log.read(64 * 1024))
    settled = (report is not None and report.get("job") == job and "usage" in report)
    passed = settled and report.get("status") == "passed" and rc == 0 and process_error is None
    status = "passed" if passed else "failed"
    if process_error and process_error["type"] in ("TimeoutExpired", "KeyboardInterrupt"):
        status = "timed_out" if process_error["type"] == "TimeoutExpired" else "stopped"
    elif settled and report["status"] in ("timed_out", "stopped"):
        status = report["status"]
    result = {"status": status, "returncode": rc, "process_error": process_error, "files": files,
              "elapsed_seconds": time.monotonic() - began, "budget": limits,
              "settlement": "observed" if settled else "reserved_upper_bound",
              "project_read_charged_bytes": (report["usage"]["project_read_observed_bytes"] if settled
                                             else limits["read_bytes"])}
    result_pin = publish_cpu_json(target / "process-result.json", result)
    return {"path": str(target), "result": result, "result_pin": result_pin}


def run_cpu_notebook(jobs, project_setting, kit_upload, pins, budgets, expected_mount=None):
    if not jobs or len(set(jobs)) != len(jobs) or any(job not in budgets for job in jobs):
        raise ValueError("Choose one or more unique CPU jobs")
    mount = Path("/content/drive")
    if expected_mount is None:
        from google.colab import drive
        drive.mount(str(mount), timeout_ms=120_000)
    else:
        mount = expected_mount
    if expected_mount is None and not os.path.ismount(mount):
        raise OSError("Drive mount not visible as a filesystem")
    my_drive = (mount / "MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    if project == my_drive or not project.is_relative_to(my_drive):
        raise ValueError("Project must be under mounted MyDrive")
    reports = project / "reports"
    if reports.resolve(strict=True) != reports or not reports.is_dir():
        raise ValueError("Expected the existing project reports directory")
    parent = reports / "v2_tuning_cpu_20261122"
    parent.mkdir(exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("CPU result parent may not be a link")
    output = parent / ("run-" + uuid.uuid4().hex)
    output.mkdir()
    start = publish_cpu_json(output / "invocation-start.json",
                             {"jobs": list(jobs), "kit": pins["kit_sha256"], "budgets": {job: budgets[job] for job in jobs},
                              "status": "running"})
    print("CPU Drive report:", output, flush=True)
    scratch = Path(tempfile.mkdtemp(prefix="v2-tuning-cpu-", dir="/content" if expected_mount is None else None))
    result = {"schema": "rewind_v2_tuning_cpu_invocation_v1", "start": start, "jobs": {},
              "bootstrap_error": None, "unstarted_jobs": list(jobs), "status": "failed"}
    try:
        kit = scratch / "kit"
        extract_pinned_zip(Path(kit_upload), pins["kit_sha256"], kit, max_zip_bytes=32 * 1024**2,
                           max_output_bytes=32 * 1024**2, max_members=32)
        extract_pinned_zip(kit / pins["source_name"], pins["source_sha256"], kit / "frozen",
                           max_zip_bytes=32 * 1024**2, max_output_bytes=192 * 1024**2, max_members=3000,
                           revision=pins["source_revision"])
        for job in jobs:
            settled = run_cpu_child(job, budgets[job], kit, project, scratch, output)
            result["jobs"][job] = settled
            result["unstarted_jobs"].remove(job)
            if settled["result"]["status"] == "stopped":
                break
        if len(result["jobs"]) == len(jobs) and all(row["result"]["status"] == "passed" for row in result["jobs"].values()):
            result["status"] = "passed"
        elif result["jobs"]:
            result["status"] = "one_or_more_checks_unpassed"
    except BaseException as exc:
        result["bootstrap_error"] = {"type": type(exc).__name__, "message": str(exc)[:1600]}
    finally:
        receipt = publish_cpu_json(output / "invocation-result.json", result)
        print(json.dumps({"drive_directory": str(output), "local_scratch": str(scratch),
                          "status": result["status"], "receipt": receipt}, ensure_ascii=False), flush=True)
    if result["status"] != "passed":
        raise RuntimeError(f"CPU invocation did not fully pass. Read {output / 'invocation-result.json'}")
    return result

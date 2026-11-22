"""Run one explicitly approved contiguous v2.4 metrics R2 sequence."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


V24_CHAIN_OVERHEAD_SECONDS = 180
V24_RESULT_SCHEMA = "rewind_v24_main_invocation_v1"
V24_OUTPUT_FOLDER = "v24_main128_a_candidate"


def v24_chain_digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def v24_chain_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def v24_chain_read(path, cap, sha, size=None):
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f"Unexpected Drive file: {path}")
    with path.open("rb") as stream:
        data = stream.read(cap + 1)
    if (not 0 < len(data) <= cap or (size is not None and len(data) != size)
            or not v24_chain_sha(sha) or hashlib.sha256(data).hexdigest() != sha):
        raise ValueError(f"Drive receipt size or SHA mismatch: {path}")
    return json.loads(data)


def v24_chain_mount(project_setting):
    from google.colab import drive

    mount = Path("/content/drive")
    if not os.path.ismount(mount):
        drive.mount(str(mount), timeout_ms=120_000)
    if not os.path.ismount(mount):
        raise OSError("Drive mount not visible")
    root = (mount / "MyDrive").resolve(strict=True)
    project = Path(project_setting).resolve(strict=True)
    reports = project / "reports"
    if project == root or not project.is_relative_to(root) or not reports.is_dir() or reports.resolve(strict=True) != reports:
        raise ValueError("Expected project reports folder under MyDrive")
    return project


def v24_chain_receipt(handoff, parent, *, pins, arm, seed, before, after, prior=None, anchor=False):
    path = Path(handoff["path"])
    if (not path.is_absolute() or path.name != "result.json" or path.parent.parent != parent
            or re.fullmatch(r"run-[0-9a-f]{32}", path.parent.name) is None):
        raise ValueError(f"Unexpected returned result path: {path}")
    receipt = v24_chain_read(path, 64 * 1024, handoff["sha256"], handoff.get("bytes"))
    expected = {"schema": V24_RESULT_SCHEMA, "status": "segment_complete", "returncode": 0,
                "arm": arm, "seed": seed, "before": before, "after": after,
                "protocol_sha256": pins["protocol_sha256"], "budgets": pins["protocol"]["budgets"],
                "path": str(path.parent)}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Metrics R2 receipt does not match the approved stage: {path}")
    previous_sha = receipt.get("resume_result_sha256")
    limit = expected["budgets"]["seconds"]
    elapsed = receipt.get("elapsed_seconds")
    if (not (v24_chain_sha(previous_sha) if before else previous_sha is None)
            or (not anchor and previous_sha != (prior["sha256"] if prior else None))
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= limit):
        raise ValueError(f"Metrics R2 receipt predecessor or elapsed time mismatch: {path}")
    pins_by_file = receipt["files"]
    if set(pins_by_file) != {"worker-result.json", "segment.json", "body-state.pt"}:
        raise ValueError("Incomplete stage files")
    budget = expected["budgets"]
    marker_name, report_name, state_name = "worker-result.json", "segment.json", "body-state.pt"
    def read_report(name):
        return v24_chain_read(path.parent / name, budget["report_bytes"], pins_by_file[name]["sha256"], pins_by_file[name]["bytes"])
    worker, report = read_report(marker_name), read_report(report_name)
    common_fields = ("arm", "seed", "before", "after", "status", "protocol_sha256", "binding_sha256", "common_sha256")
    binding = report["binding"]
    common = binding["common"]
    protocol = pins["protocol"]
    if (worker.get("schema") != "rewind_v24_main_worker_v1"
            or any(worker.get(k) != receipt.get(k) for k in common_fields)
            or worker.get("files") != {k: pins_by_file[k] for k in (report_name, state_name)}
            or binding.get("arm") != arm or binding.get("module_sha256") != pins["modules"]
            or binding.get("source_sha256") != pins["base"]["source_sha256"]
            or common.get("seed") != seed or common.get("protocol_sha256") != pins["protocol_sha256"]
            or any(common["environment"].get(k) != v for k, v in protocol["runtime"].items())
            or v24_chain_digest(binding) != receipt.get("binding_sha256")
            or v24_chain_digest(common) != receipt.get("common_sha256")
            or (prior and prior["binding_sha256"] != receipt["binding_sha256"])
            or (report.get("before"), report.get("after"), report.get("resume_result_sha256")) != (before, after, previous_sha)):
        raise ValueError(f"Metrics R2 worker or checkpoint lineage mismatch: {path}")
    # Existing publisher verifies every checkpoint byte on Drive. The next worker also streams and verifies
    # its complete input state before loading. Check presence and matching length without a third full read.
    state = path.parent / state_name
    state_pin = pins_by_file[state_name]
    if (state.is_symlink() or state.resolve(strict=True) != state or not state.is_file() or not v24_chain_sha(state_pin["sha256"])
            or not 0 < state.stat().st_size == state_pin["bytes"] <= budget["state_bytes"]):
        raise ValueError(f"Missing or changed checkpoint: {state}")
    updates = report["updates"]
    if ([r["step"] for r in updates] != list(range(before + 1, after + 1))
            or not all(v24_chain_sha(r["noise_sha256"]) for r in updates)
            or v24_chain_digest([[r["step"], r["key"], r["noise_sha256"]] for r in updates]) != report["paired_update_sha256"]):
        raise ValueError(f"Update cursor or paired noise mismatch: {path}")
    scores = report["scores"]
    if [row["update"] for row in scores] != ([0, after] if before == 0 else [after]):
        raise ValueError(f"Evaluation boundary mismatch: {path}")
    for row in scores:
        final = row["update"] == protocol["official_validation_update"]
        scope, clips, frames = ("official_validation", 421, 16425) if final else ("pilot_validation_subset", 6, 300)
        pooled = row["metrics"]["pooled"]
        if ((row["scope"], row["clips"], row["frames"], pooled["clips"], pooled["frames"]) != (scope, clips, frames, clips, frames)
                or not v24_chain_sha(row["noise_and_roster_sha256"]) or (final and len(row["metrics"]["per_scenario"]) != 3)):
            raise ValueError(f"Evaluation scope or sample count mismatch: {path}")
    return {"path": str(path), "sha256": handoff["sha256"], "binding_sha256": receipt["binding_sha256"],
            "after": after, "elapsed_seconds": elapsed, "scope": scores[-1]["scope"]}


def run_v24_chain(project_setting, arm, seed, boundaries, resume, resume_sha, protocol_sha, extension, pins, *,
                  run_segment, prepare_project=v24_chain_mount, clock=time.monotonic):
    protocol = pins["protocol"]
    all_boundaries = protocol["boundaries"]
    seeds = protocol["initial_seeds"] + protocol["conditional_seeds"]
    if (not v24_chain_sha(protocol_sha) or protocol_sha != pins["protocol_sha256"] or v24_chain_digest(protocol) != protocol_sha
            or arm not in ("x0", "v") or type(seed) is not int or seed not in seeds or type(extension) is not bool
            or (seed in protocol["conditional_seeds"] and not extension)
            or not isinstance(boundaries, (tuple, list)) or len(boundaries) < 2
            or any(type(step) is not int or step not in all_boundaries for step in boundaries)):
        raise ValueError("Explicitly approved arm, seed, protocol and consecutive boundaries are required")
    first = all_boundaries.index(boundaries[0])
    if (list(boundaries) != all_boundaries[first:first + len(boundaries)] or bool(first) != bool(resume)
            or bool(resume) != bool(resume_sha) or (resume and not v24_chain_sha(resume_sha))):
        raise ValueError("Approved boundaries must be consecutive; a nonzero start needs its exact predecessor path and SHA")
    started = clock()
    plan = list(zip(boundaries, boundaries[1:]))
    budget = protocol["budgets"]
    per_stage = budget["seconds"]
    ceiling = len(plan) * per_stage + V24_CHAIN_OVERHEAD_SECONDS
    deadline = started + ceiling
    print(f"v2.4 {protocol['runtime']['gpu']} {arm}/seed {seed}, approved stages: {plan}", flush=True)
    print(f"Plan: {len(plan)} × {per_stage}s + {V24_CHAIN_OVERHEAD_SECONDS}s chain preflight/readback = {ceiling}s; "
          f"worker read {len(plan)*budget['worker_logical_read_bytes']} bytes, total logical writes "
          f"{len(plan)*budget['all_logical_write_bytes']} bytes; scratch {budget['scratch_free_bytes']} bytes per stage", flush=True)
    project = prepare_project(project_setting)
    parent = project / "reports" / V24_OUTPUT_FOLDER
    prior = None
    if first:
        prior = v24_chain_receipt({"path": resume, "sha256": resume_sha}, parent, pins=pins, arm=arm, seed=seed,
                                  before=all_boundaries[first - 1], after=boundaries[0], anchor=True)
        print(f"Verified resume anchor {prior['path']} SHA-256 {prior['sha256']}", flush=True)
    seen = {prior["path"]} if prior else set()
    completed = []
    for index, (before, after) in enumerate(plan, 1):
        try:
            if deadline - clock() < (len(plan) - index + 1) * per_stage:
                raise TimeoutError("Insufficient overall time for the remaining approved stages")
            print(f"Stage {index}/{len(plan)} starting {before} → {after}; {per_stage}s, "
                  f"worker/publish reserve {budget['worker_reserve_seconds']}/{budget['publish_reserve_seconds']}s", flush=True)
            handoff = run_segment(str(project), arm, seed, before, after, prior["path"] if prior else "",
                                  prior["sha256"] if prior else "", protocol_sha, extension, pins)
            if handoff["status"] != "segment_complete" or handoff["path"] in seen:
                raise RuntimeError(f"Stage returned {handoff['status']} or a reused UUID: {handoff['path']} SHA-256 {handoff['sha256']}")
            accepted = v24_chain_receipt(handoff, parent, pins=pins, arm=arm, seed=seed, before=before, after=after, prior=prior)
            if clock() >= deadline:
                raise TimeoutError("Chain time budget exhausted during Drive verification")
            prior = accepted
            seen.add(prior["path"])
            completed.append(prior)
            print(f"Stage {index}/{len(plan)} verified at {after} ({prior['scope']}), {prior['elapsed_seconds']:.1f}s; "
                  f"{prior['path']} SHA-256 {prior['sha256']}", flush=True)
        except BaseException:
            if prior:
                print(f"Last verified resume anchor: {prior['path']} SHA-256 {prior['sha256']}", flush=True)
            print(f"Chain stopped at {before} → {after}; no automatic retry", flush=True)
            raise
    print(f"All approved stages verified, elapsed {clock()-started:.1f}/{ceiling}s", flush=True)
    return completed

"""Colab stages for the pinned ArchVis 1,000-pair comparison."""

from collections import defaultdict
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _hash, _no_link, _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import ExtractedLimits, _local, prepare_extracted_t1


ENVIRONMENT = "ArchVisInterior_ArchVis_RT"
DATASET_VERSION = "unrealego2_v1_archvis_train_val"
INVENTORY_SHA256 = "957cf19666419d18ee7aa9601a1a4ba8359c1a2e32dfc37a3fe1da9a374620a0"
MIB = 1024**2
TRAIN_COUNT = 1000
SEED = 71
CATEGORY = {"rgb": "UnrealEgoData2_train_val_rgb", "pose": "UnrealEgoData2_train_val_pose"}
ASSETS = ("split/train.txt", "split/validation.txt", "split/test.txt", "scaramuzza.json",
          "inventory.json", "motion_map.json", "motion_review.json", "mse.json", "ce.json")
LIMIT_FIELDS = ("max_seconds", "max_attempts", "max_read_bytes", "max_rss_bytes",
                "max_disk_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes")


def _json(path, limit=8 * MIB):
    return read_json(_regular_read(_local(path), limit))


def _write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def _sealed(body):
    return {**body, "content_sha256": digest(body)}


def _verify(doc, expected_sha256, schema):
    if (not _hash(expected_sha256) or doc.get("schema") != schema
            or doc.get("content_sha256") != expected_sha256
            or digest({k: v for k, v in doc.items() if k != "content_sha256"}) != expected_sha256):
        raise ValueError("archvis_stage_pin")


def assets(work):
    root = _local(Path(work) / "inputs", directory=True)
    receipt = _json(root / "input_receipt.json")
    if receipt.get("schema") != "archvis_pilot_inputs_v1" or set(receipt["files"]) != set(ASSETS):
        raise ValueError("archvis_inputs_receipt")
    configs = {}
    for name in ASSETS:
        row = receipt["files"][name]
        raw = _regular_read(_local(root / name), 8 * MIB, row["bytes"])
        if sha(raw) != row["sha256"]:
            raise ValueError("archvis_inputs_changed")
        if name in ("mse.json", "ce.json"):
            configs[name] = read_json(raw)
    if receipt["files"]["inventory.json"]["sha256"] != INVENTORY_SHA256:
        raise ValueError("archvis_inventory_pin")
    mse, ce = configs["mse.json"], configs["ce.json"]
    if (mse.get("loss_kind") != "mse" or ce != {**mse, "loss_kind": "gaussian_cross_entropy"}
            or mse.get("batch_size") != 2 or mse.get("amp") is not False):
        raise ValueError("archvis_comparison_loss_only")
    return root, receipt


def discover_roots(mount_root, *, max_entries=5000, max_depth=4, max_seconds=60):
    """Search only for category roots. Never visit frame directories inside a hit."""
    mount = _local(mount_root, directory=True)
    if not (1 <= max_entries <= 20_000 and 0 <= max_depth <= 6 and 0 < max_seconds <= 300):
        raise ValueError("archvis_discovery_limits")
    started = time.monotonic()
    candidates, visited, stopped = defaultdict(list), 0, None
    pending = [(mount, 0)]
    while pending and stopped is None:
        path, depth = pending.pop()
        role = next((k for k, name in CATEGORY.items() if path.name == name), None)
        if role:
            if (path / ENVIRONMENT).is_dir():
                candidates[role].append(str(path))
            continue
        if depth == max_depth:
            continue
        with os.scandir(path) as entries:
            for entry in entries:
                visited += 1
                if visited > max_entries or time.monotonic() - started > max_seconds:
                    stopped = "discovery_budget"
                    break
                if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False):
                    child = Path(entry.path)
                    try:
                        _no_link(child)
                    except ValueError:
                        continue
                    pending.append((child, depth + 1))
    return {"mount_root": str(mount), "candidates": {k: sorted(candidates[k]) for k in CATEGORY},
            "visited_entries": visited, "stopped": stopped,
            "scope": "directory metadata; exact roots require explicit confirmation"}


def _roots(mount_root, rgb_root, pose_root):
    mount = _local(mount_root, directory=True)
    roots = {k: _local(v, directory=True) for k, v in (("rgb", rgb_root), ("pose", pose_root))}
    if (roots["rgb"] == roots["pose"] or any(not v.is_relative_to(mount)
            or not _local(v / ENVIRONMENT, directory=True).is_relative_to(v) for v in roots.values())):
        raise ValueError("archvis_confirmed_roots")
    return roots


def _options(work, roots, limits):
    root, receipt = assets(work)
    spec = _local(Path(work) / "source/configs/real_archvis62_v1.json")
    if sha(_regular_read(spec, 256 * 1024)) != receipt["spec_sha256"]:
        raise ValueError("archvis_spec_pin")
    return dict(rgb_root=roots["rgb"], pose_root=roots["pose"], environment=ENVIRONMENT,
                dataset_version=DATASET_VERSION, split_dir=root / "split", motion_map=root / "motion_map.json",
                spec_path=spec, calibration_path=root / "scaramuzza.json",
                output_dir=Path(work) / "manifest", limits=limits, train_limit=TRAIN_COUNT, seed=SEED,
                inventory=root / "inventory.json", inventory_sha256=INVENTORY_SHA256)


def stat_cost(plan):
    train, validation = (plan["payload_bytes"][split] + 3 * plan["counts"][split]
                         for split in ("train", "validation"))
    return {"metadata_file_bytes": plan["metadata_bytes"], "directory_entries": plan["directory_entries"],
            "selected_file_stats": plan["selected_file_stats"],
            "selected_payload_bytes": sum(plan["payload_bytes"].values()),
            "prepare_min_read_bytes": plan["metadata_bytes"] + train + validation + 1,
            "prepare_png_decodes": 2 * sum(plan["counts"].values()),
            "prepare_pose_checks": sum(plan["counts"].values()),
            "fresh_one_epoch_data_bytes_each": 3 * train + 2 * validation,
            "fresh_two_run_data_bytes": 2 * (3 * train + 2 * validation),
            "resume_same_epoch_data_bytes_upper_each": 2 * train + validation,
            "formula": "fresh train: initial+epoch+final; validation: initial+final; "
                       "resume: remaining train updates plus final train and validation; one byte per file reserved",
            "scope": "local stat-based input read estimate; network transfer and wall time unmeasured"}


def preflight(work, mount_root, rgb_root, pose_root, *, max_seconds, max_entries):
    work = _local(work, directory=True)
    if (work / "preflight.json").exists():
        raise ValueError("archvis_preflight_already_frozen")
    roots = _roots(mount_root, rgb_root, pose_root)
    plan = prepare_extracted_t1(**_options(work, roots, ExtractedLimits(8 * MIB, max_seconds, max_entries)),
                               plan_only=True)
    report = _sealed({"schema": "archvis_pilot_preflight_v1", "plan": plan, "cost": stat_cost(plan),
                      "metadata_limits": {"max_seconds": max_seconds, "max_entries": max_entries},
                      "mount_root": str(_local(mount_root, directory=True)),
                      "status": "review_inventory" if plan["inventory_missing_eligible"]
                      or plan["inventory_extra_eligible"] else "ready_for_manifest_approval"})
    _write(work / "preflight.json", report)
    return report


def prepare(work, approved_preflight_sha256, *, max_read_bytes, max_seconds, max_entries):
    work = _local(work, directory=True)
    report = _json(work / "preflight.json", 64 * MIB)
    _verify(report, approved_preflight_sha256, "archvis_pilot_preflight_v1")
    if (report["status"] != "ready_for_manifest_approval"
            or max_read_bytes < report["cost"]["prepare_min_read_bytes"]
            or (work / "prepare_report.json").exists() or (work / "manifest").exists()):
        raise ValueError("archvis_manifest_preflight_or_budget")
    plan = report["plan"]
    roots = _roots(report["mount_root"], plan["roots"]["rgb"], plan["roots"]["pose"])
    started = time.monotonic()
    try:
        manifest = prepare_extracted_t1(
            **_options(work, roots, ExtractedLimits(max_read_bytes, max_seconds, max_entries)),
            expected_plan_sha256=plan["content_sha256"])
        path = work / "manifest/manifest.json"
        payload = _regular_read(path, 64 * MIB)
        result = {"status": "completed", "manifest_sha256": manifest["content_sha256"],
                  "manifest_file_sha256": sha(payload), "manifest_file_bytes": len(payload),
                  "validation_keys_sha256": manifest["selection"]["validation_keys_sha256"],
                  "audit": manifest["audit"], "roots": plan["roots"],
                  "preflight_sha256": approved_preflight_sha256,
                  "read_headroom_pct_of_stat_minimum":
                      (max_read_bytes / report["cost"]["prepare_min_read_bytes"] - 1) * 100}
    except (Exception, KeyboardInterrupt) as exc:
        _write(work / "prepare_report.json", {"status": "failed", "type": type(exc).__name__,
               "reason": str(exc)[:512], "elapsed_seconds": time.monotonic() - started})
        raise
    result["elapsed_seconds"] = time.monotonic() - started
    _write(work / "prepare_report.json", result)
    return result


def _minimum_reads(work, receipt, prepared, preflight_doc):
    spec_size = (work / "source/configs/real_archvis62_v1.json").stat().st_size + 1
    metadata = spec_size + prepared["manifest_file_bytes"] + 1
    return {loss: preflight_doc["cost"]["fresh_one_epoch_data_bytes_each"] + metadata
            + receipt["files"][loss + ".json"]["bytes"] + 1 for loss in ("mse", "ce")}


def runtime_resources(work):
    import psutil
    import torch

    memory = psutil.virtual_memory()
    cuda = torch.cuda.is_available()
    return {"host_total_bytes": memory.total, "host_available_bytes": memory.available,
            "disk_free_bytes": shutil.disk_usage(work).free, "torch": str(torch.__version__),
            "cuda_available": cuda, "gpu_name": torch.cuda.get_device_name() if cuda else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None}


def comparison_plan(work, *, device, limits):
    from rewind_m1.extracted_rgb_cli import RunLimits

    work = _local(work, directory=True)
    _, receipt = assets(work)
    prepared = _json(work / "prepare_report.json")
    preflight_doc = _json(work / "preflight.json", 64 * MIB)
    _verify(preflight_doc, prepared["preflight_sha256"], "archvis_pilot_preflight_v1")
    if (prepared["status"] != "completed" or device not in ("cuda", "cpu")
            or set(limits) != set(LIMIT_FIELDS)):
        raise ValueError("archvis_comparison_inputs")
    resources = runtime_resources(work)
    if device == "cuda" and not resources["cuda_available"]:
        raise ValueError("archvis_cuda_unavailable")
    RunLimits(**limits)
    minimum = _minimum_reads(work, receipt, prepared, preflight_doc)
    if (limits["max_attempts"] < (TRAIN_COUNT + 1) // 2 or limits["max_read_bytes"] < max(minimum.values())
            or shutil.disk_usage(work).free < 2 * limits["max_disk_bytes"] + 4 * MIB):
        raise ValueError("archvis_comparison_budget")
    payload = _sealed({"schema": "archvis_pilot_comparison_v1", "device": device, "epochs": 1,
                      "limits_each": limits, "roots": prepared["roots"],
                      "runtime_observation": resources,
                      "manifest_sha256": prepared["manifest_sha256"], "manifest_file_sha256": prepared["manifest_file_sha256"],
                      "validation_keys_sha256": prepared["validation_keys_sha256"],
                      "config_file_sha256": {loss: receipt["files"][loss + ".json"]["sha256"] for loss in minimum},
                      "fresh_minimum_read_bytes_each": minimum, "fresh_minimum_read_bytes_both": sum(minimum.values()),
                      "read_headroom_pct_each": {k: (limits["max_read_bytes"] / v - 1) * 100 for k, v in minimum.items()},
                      "resume_same_epoch_data_bytes_upper_each": preflight_doc["cost"]["resume_same_epoch_data_bytes_upper_each"],
                      "resume_note": "add saved checkpoint and metadata bytes; remaining train can be smaller; "
                                     "separate approved recovery, no automatic retries"})
    _write(work / "comparison_plan.json", payload)
    return payload


def compare(work, approved_plan_sha256, *, launch=subprocess.run):
    work = _local(work, directory=True)
    plan = _json(work / "comparison_plan.json")
    _verify(plan, approved_plan_sha256, "archvis_pilot_comparison_v1")
    inputs, receipt = assets(work)
    manifest = work / "manifest/manifest.json"
    manifest_raw = _regular_read(_local(manifest), 64 * MIB)
    doc = read_json(manifest_raw)
    if (sha(manifest_raw) != plan["manifest_file_sha256"] or doc["content_sha256"] != plan["manifest_sha256"]
            or any((work / name).exists() or (work / name).is_symlink()
                   for name in ("mse", "ce", "comparison_result.json"))):
        raise ValueError("archvis_comparison_input_or_output_changed")
    for loss in ("mse", "ce"):
        if plan["config_file_sha256"][loss] != receipt["files"][loss + ".json"]["sha256"]:
            raise ValueError("archvis_comparison_config_changed")
    result = {"comparison_plan_sha256": approved_plan_sha256, "manifest_sha256": plan["manifest_sha256"],
              "status": "started", "runs": {}}
    try:
        for loss in ("mse", "ce"):
            if sha(_regular_read(manifest, 64 * MIB)) != plan["manifest_file_sha256"]:
                raise ValueError("archvis_comparison_manifest_changed_between_runs")
            config_path = inputs / (loss + ".json")
            config_raw = _regular_read(_local(config_path), 64 * 1024)
            if sha(config_raw) != plan["config_file_sha256"][loss]:
                raise ValueError("archvis_comparison_config_changed_between_runs")
            command = [sys.executable, "-m", "rewind_m1.extracted_rgb_cli",
                       "--manifest", str(manifest), "--manifest-sha256", plan["manifest_sha256"],
                       "--rgb-root", plan["roots"]["rgb"], "--pose-root", plan["roots"]["pose"],
                       "--spec", str(work / "source/configs/real_archvis62_v1.json"), "--config", str(config_path),
                       "--output", str(work / loss), "--device", plan["device"], "--epochs", str(plan["epochs"])]
            for key in LIMIT_FIELDS:
                command.extend(["--" + key.replace("_", "-"), str(plan["limits_each"][key])])
            child = launch(command, cwd=work / "source", timeout=plan["limits_each"]["max_seconds"] + 15)
            result["runs"][loss] = {"returncode": child.returncode}
            report_path = work / loss / "report.json"
            if report_path.is_file():
                raw = _regular_read(_local(report_path), 4 * MIB)
                report = read_json(raw)
                result["runs"][loss].update(report_sha256=sha(raw), status=report["status"])
                if child.returncode == 0 and (report.get("binding", {}).get("manifest_sha256") != plan["manifest_sha256"]
                        or report.get("config") != read_json(config_raw)):
                    raise ValueError("archvis_run_manifest_or_config_mismatch")
            if child.returncode != 0 or result["runs"][loss].get("status") != "completed":
                result["status"] = "stopped_with_results"
                break
        else:
            result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
        raise
    finally:
        _write(work / "comparison_result.json", result)
    return result


def recover(work):
    work = _local(work, directory=True)
    output = work / "archvis_pilot_recovery.zip"
    names = ["bootstrap_report.json", "roots_candidates.json", "preflight.json", "prepare_report.json",
             "manifest/manifest.json", "comparison_plan.json", "comparison_result.json",
             "inputs/input_receipt.json", "inputs/motion_review.json", "inputs/mse.json", "inputs/ce.json"]
    names.extend(f"{stage}_failure.json" for stage in ("discover", "preflight", "prepare", "budget", "train"))
    names.extend(f"{loss}/{name}" for loss in ("mse", "ce") for name in ("last.pt", "weights.pt", "report.json"))
    paths = [(name, _local(work / name)) for name in names if (work / name).is_file()]
    size = sum(path.stat().st_size for _, path in paths)
    if not paths or size > 128 * MIB or output.exists() or output.is_symlink():
        raise ValueError("archvis_recovery_bounds_or_exists")
    if shutil.disk_usage(work).free < size + 2 * MIB:
        raise ValueError("archvis_recovery_disk_space")
    receipt = {}
    remaining = 128 * MIB
    temporary = work / "archvis_pilot_recovery.zip.partial"
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths:
            raw = _regular_read(path, min(64 * MIB, remaining))
            remaining -= len(raw)
            archive.writestr(name, raw)
            receipt[name] = {"bytes": len(raw), "sha256": sha(raw)}
        archive.writestr("recovery_receipt.json", canonical({"files": receipt}) + b"\n")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("archvis_recovery_crc")
    temporary.rename(output)
    return output

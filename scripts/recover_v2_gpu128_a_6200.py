"""One candidate promotion for 128/A; import only the verified frozen training ZIP."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_v1 import v2_contract as contract, v2_retention as retention
from rewind_v1.run_stages import FrameStage
from rewind_v1.runner import LAST, Pipeline, _read_checkpoint, _restore_state
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_bulk_cache import writer
from scripts.run_v2_gpu_training import identity_for, training_config
from scripts.v2_gpu_training_storage import Checkpoints, MIB, verify_first


SOURCE_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
CORE_SHA = "59eede3f696bc9d9f054a0e4a5d4188e372ced2ceee3e3ef951246584121fa4f"
MODULES = {
    "scripts/run_v2_gpu_training.py": "b4c20541bf5d93c4000ff693d26e1b2a591fea6912397b8d41bc962eef8a0fee",
    "scripts/v2_gpu_training_storage.py": "5f15e81799440684d31e7a780d24e772ff2278e4445bddfaec7af3efb393205a",
    "scripts/run_v2_bulk_cache.py": "36fffef104e1000fd480c748f07b815aabaf34986a688729dd36f42146d514fd",
    "scripts/archvis_pilot_bootstrap.py": "b9ca011bf3f63b04edfece2cb3aa3ddcd412b5aa389e05118f0bcf9759f9915e",
}
SPEC_SHA = "32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187"
MANIFEST_SHA = "51657741aedb25ddafaf6fbffef706f4465729f787a3fa0fa004d8dcaa65dd9c"
RUN = "reports/v2_gpu_training_128_v1/A-seed-42"
CANDIDATE = "recovery-candidates/invocation-5ednwc4x-42c7bfe8f55d4fc09ab56a1e600da8e4/run"
CURSOR = {"stage": "frontend", "epoch": 0, "batch": 6200, "updates": 6200}
LAST_PIN = {"name": LAST, "bytes": 4694368, "sha256": "53b6287264d15eb5c5f9a33cf84f4d067ec1438b5c7bf8926d8cf817acd51746"}
REPORT_PIN = {"name": "run.json", "bytes": 5143, "sha256": "45a00e519cc2e8670bcc6c59c87670e6d6df6b7864d51cacd68eb39806585c2d"}
GATE_PIN = {"name": "first-update.json", "bytes": 1912, "sha256": "24f5d7fd8211ff3f602314a9574ee66c72fd5c314704079af97f0378c53fa5f4"}
FIRST_STEP = "c15fe06f707ce6e14a22ea8b19b9759ff4ce42963b3f2279dd90395a22bd7aa5"
FIRST_GATE = "18923ab13c444f2c8d44d96023778997eb110a17a9f0994b28a363e1dd2dd7d3"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned_source():
    if digest(contract.source()) != CORE_SHA:
        raise ValueError("recovery_requires_frozen_128_core")
    for name, expected in MODULES.items():
        if sha(_regular_read(contract.ROOT/name, MIB)) != expected:
            raise ValueError(f"recovery_requires_frozen_module: {name}")
    raw = _regular_read(contract.ROOT/"reference/spec.json", MIB)
    if sha(raw) != SPEC_SHA:
        raise ValueError("recovery_spec_pin")
    spec = json.loads(raw)
    return spec["joint_names"]


def read_pin(root, pin, ledger):
    ledger.reserve("read", pin["bytes"])
    raw = _regular_read(root/pin["name"], pin["bytes"], pin["bytes"])
    ledger.observed("read", len(raw), persist=False)
    if sha(raw) != pin["sha256"]:
        raise ValueError(f"recovery_pin: {pin['name']}")
    return raw


def within(path, root):
    if path.is_symlink() or not path.is_dir() or path.resolve() != path or not path.is_relative_to(root):
        raise ValueError(f"recovery_directory: {path}")
    return path


def recovery_config(training):
    return {**training, "seconds_per_invocation": 780, "external_timeout_seconds": 780, "terminal_reserve_seconds": 45,
            "package_seconds": 90, "copy_seconds": 90, "resources": {**training["resources"],
            "read_bytes": 128*MIB, "write_bytes": 256*MIB, "archive_bytes": 32*MIB}}


def pipeline_at(path, binding, environment):
    # _resume is the actual pinned runner check. FullData and CUDA are unnecessary for this cursor.
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.root, pipeline.binding, pipeline.environment = path, binding, environment
    pipeline.cfg = binding["config"]
    pipeline.updates = {"frontend": 59793, "body": 3097, "hand": 3097}
    return pipeline


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def dry_resume(pipeline, pin, cursor, joints):
    fit = pipeline._resume(pin["sha256"])
    if pipeline.last != pin or any(pipeline.progress[k] != value for k, value in cursor.items()):
        raise ValueError("recovery_normal_cursor_or_pin")
    if not finite_tree(fit) or set(fit) != {"model", "optimizer", "scheduler", "scaler", "rng"}:
        raise ValueError("recovery_nonfinite_or_missing_fit")
    rng = fit["rng"]
    random.Random().setstate(rng["python"])
    torch.Generator().set_state(rng["torch_cpu"])
    if (len(rng["torch_cuda"]) != 1 or not rng["torch_cuda"][0].numel() or rng["torch_cuda"][0].ndim != 1
            or rng["torch_cuda"][0].dtype != torch.uint8):
        raise ValueError("recovery_cuda_rng_shape")
    original_rng = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            stage = FrameStage(pipeline.cfg["frontend"], joints, torch.device("cpu"))
            # CUDA RNG is verified structurally above and preserved byte-for-byte in the archive.
            _restore_state(stage, {**fit, "rng": {**rng, "torch_cuda": []}}, pipeline._step())
    finally:
        random.setstate(original_rng)
    optimizer_states = stage.optimizer.state
    if not optimizer_states:
        raise ValueError("recovery_empty_optimizer")
    steps = []
    for parameter, state in optimizer_states.items():
        scalar = float(state["step"].item())
        if (not scalar.is_integer() or not 0 < scalar <= cursor["updates"]
                or state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape):
            raise ValueError("recovery_optimizer_state")
        steps.append(int(scalar))
    return {"optimizer_step_max": max(steps), "scheduler_epoch": fit["scheduler"]["last_epoch"],
            "cuda_rng_bytes": rng["torch_cuda"][0].numel()}


def candidate(scratch, directory, identity, training, environment, ledger):
    local = scratch/"candidate"
    local.mkdir()
    for pin in (LAST_PIN, REPORT_PIN):
        _atomic(local/pin["name"], read_pin(directory, pin, ledger))
    state = _read_checkpoint(local/LAST, LAST_PIN["sha256"], LAST_PIN["bytes"])
    binding, progress = state["binding"], state["progress"]
    report = json.loads((local/"run.json").read_bytes())
    if (binding["experiment"] != identity or binding["v2_config"] != training or binding["source_code_sha256"] != SOURCE_SHA
            or binding["format"] != "rewind_v2_gpu_training_v1" or binding["config"]["frontend"]["size"] != [128, 128]
            or report["actual_environment"] != environment
            or binding["environment"] != {k: v for k, v in environment.items() if k not in {"device_name", "device_memory_bytes"}}
            or any(report[k] != v for k, v in binding.items())
            or report["last_checkpoint"] != LAST_PIN or report["checkpoint_cursor"] != progress
            or any(progress[k] != v for k, v in CURSOR.items())
            or progress["history"] or progress["best"] or progress["milestones"]):
        raise ValueError("recovery_candidate_binding_or_retained_cursor")
    return pipeline_at(local, binding, environment)


def recover(project, scratch, publish, joints, result):
    seed = within(project/RUN, project)
    directory = within(seed/CANDIDATE, seed)
    root = within(seed/"checkpoints", seed)
    identity = identity_for("A", {"source_sha256": SOURCE_SHA, "manifest_content_sha256": MANIFEST_SHA})
    training = training_config(42)
    cfg = recovery_config(training)
    ledger = Ledger(scratch/"usage", identity, cfg)
    with writer(seed) as owner, ledger.invocation("recover_128_a_6200"):
        saver = Checkpoints(root, scratch, identity, None, cfg, ledger, owner)
        history = saver.history()
        gate = json.loads(read_pin(root, GATE_PIN, ledger))
        if (len(history) not in (1, 2) or history[0] != gate["receipt"] or history[0]["content_sha256"] != FIRST_STEP
                or gate["content_sha256"] != FIRST_GATE or history[0]["cursor"] != {"stage": "frontend", "epoch": 0, "batch": 1, "updates": 1}):
            raise ValueError("recovery_unsupported_history")
        first, tip = history[0], history[-1]
        if len(history) == 2 and (tip["last"] != LAST_PIN or tip["cursor"] != CURSOR or tip["completed"] is not False
                                  or tip["environment"] != first["environment"] or "first_proof" in tip):
            raise ValueError("recovery_unexpected_current_tip")
        original_gate = verify_first(root, scratch, identity, ledger, cfg)
        if original_gate != FIRST_GATE:
            raise ValueError("recovery_original_gate")
        proposed = candidate(scratch, directory, identity, training, first["environment"], ledger)
        saver.pipeline = pipeline_at(scratch/"original", proposed.binding, first["environment"])
        saver.restore(first)  # Verifies exact candidate binding against the pinned original checkpoint.
        result["candidate_state"] = dry_resume(proposed, LAST_PIN, CURSOR, joints)
        result.update(previous_receipt_sha256=FIRST_STEP, first_gate_sha256=FIRST_GATE, cursor=CURSOR, last=LAST_PIN)
        if len(history) == 1:
            dry = scratch/"dry.zip"
            dry_pin = retention.package_run(proposed.root, dry, cfg, ledger)
            destination = scratch/"dry-restored"
            retention.restore_run(dry, dry_pin, destination, cfg, ledger, proposed.binding)
            dry_resume(pipeline_at(destination, proposed.binding, proposed.environment), LAST_PIN, CURSOR, joints)
            result["status"] = "verified_only"
            if not publish:
                return
            if any(root.glob("result-000002-*.zip*")):
                raise ValueError("recovery_unreceipted_step2_archive_requires_review")
            saver.pipeline = proposed
            tip = saver.save()
            result.update(status="published_restore_pending", receipt=tip)
        saver.pipeline = pipeline_at(scratch/"published-restored", proposed.binding, proposed.environment)
        saver.restore(tip)
        dry_resume(saver.pipeline, LAST_PIN, CURSOR, joints)
        if saver.history() != [first, tip] or verify_gate_bytes(root, ledger) != FIRST_GATE:
            raise ValueError("recovery_final_history_or_gate")
        result.update(status="published_verified" if len(history) == 1 else "already_published_verified", receipt=tip,
                      normal_resume_tip=tip["content_sha256"], archive=tip["archive"])


def verify_gate_bytes(root, ledger):
    return json.loads(read_pin(root, GATE_PIN, ledger))["content_sha256"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if work.is_relative_to(project) or project.is_relative_to(work):
        raise ValueError("recovery_separate_local_workspace_required")
    torch.set_num_threads(1)
    result = {"status": "failed", "publish_requested": args.publish, "source_sha256": SOURCE_SHA}
    try:
        with tempfile.TemporaryDirectory(dir=work, prefix="operation-") as directory:
            recover(project, Path(directory), args.publish, len(pinned_source()), result)
    except BaseException as exc:
        result["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:512]}
        raise
    finally:
        _atomic(work/"recovery-result.json", canonical(result))
        print(canonical(result).decode(), flush=True)


if __name__ == "__main__":
    main()

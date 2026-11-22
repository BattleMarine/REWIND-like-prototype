"""Audit the durable A checkpoint on CPU and verify the proposed trial controls."""

import io
import json
import random
from types import SimpleNamespace

import torch

from rewind_archive.contracts import digest
from rewind_v1.frontend import ARCHITECTURE, frontend_fingerprint
from rewind_v1.run_stages import CascadeStages, FrameStage
from rewind_v1.runner import _fit_state, _restore_state
from scripts.v2_gpu128_a_bundle_format import A, FEATURE_IDENTITY, hash_stream
from scripts.v2_tuning_cpu_common import (BundleView, MIB, check_json_pin, native_evidence, require,
                                         staged_archive, write_json)
from scripts.v2_tuning_cpu_selection import BODY, INITIAL, ADDITIONAL, METRICS, SCENARIOS, plan, summarize


ROLES = ("frontend", "body", "hand")
CURSOR = {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


def read_weight(view, name, receipt):
    raw = view.read(name, 256 * MIB, receipt[name])
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def body_targets(model):
    params = dict(model.named_parameters())
    hidden = model.hidden
    head = {"final_head.weight": (3, hidden), "final_head.bias": (3,)}
    qkv = {name: (3 * hidden, hidden) for name in params if name.endswith(".qkv.weight") and name.startswith("temporal.")}
    require(set(qkv) == {f"temporal.{i}.qkv.weight" for i in range(3)}
            and all(tuple(params[name].shape) == shape for name, shape in {**head, **qkv}.items()),
            "body head or temporal QKV target shape")
    return {"final_head": {name: list(shape) for name, shape in head.items()},
            "temporal_qkv": {name: list(shape) for name, shape in qkv.items()}}


def audit_checkpoint(copy, step, diagnostic, comparison, spec, meter):
    with BundleView(copy) as view:
        receipt = json.loads(view.read("recovery_receipt.json", 8 * MIB))
        check_json_pin(receipt, "rewind_v2_result_recovery_v1")
        files = receipt["files"]
        require(view.members.keys() == files.keys() | {"recovery_receipt.json"}
                and files["last.pt"] == {key: step["last"][key] for key in ("bytes", "sha256")},
                "checkpoint roster / last pin")
        meter.mark("trials/checkpoint-inner-sha")
        for name, pin in files.items():
            require("/" not in name, "unexpected checkpoint inner path")
            with view.open(name, pin["bytes"]) as stream:
                hash_stream(stream, pin, meter.check)
        meter.mark("trials/checkpoint-native-state")
        original = read_weight(view, "last.pt", files)
        progress = original["progress"]
        binding = original["binding"]
        require(step["cursor"] == CURSOR and {key: progress[key] for key in CURSOR} == CURSOR
                and binding["experiment"] == diagnostic["identity"] == comparison["binding"]["experiment"]
                and binding["source_code_sha256"] == diagnostic["identity"]["source"]
                and progress == diagnostic["checkpoint_cursor"],
                "native checkpoint parent/cursor/report mismatch")
        fit = original["fit"]
        require(set(fit) == {"model", "optimizer", "scheduler", "scaler", "rng"}
                and fit["scheduler"]["last_epoch"] == 9291
                and bool(fit["optimizer"]["state"]) and finite_tensors(fit)
                and set(fit["rng"]) == {"python", "torch_cpu", "torch_cuda"}
                and len(fit["rng"]["torch_cuda"]) == 1
                and fit["rng"]["torch_cpu"].dtype == torch.uint8
                and fit["rng"]["torch_cuda"][0].dtype == torch.uint8,
                "native GPU optimizer/scheduler/scaler/RNG storage")
        fit_report = {"scheduler_last_epoch": fit["scheduler"]["last_epoch"],
                      "optimizer_state_entries": len(fit["optimizer"]["state"]),
                      "saved_cuda_rng_devices": len(fit["rng"]["torch_cuda"])}
        cfg = binding["config"]
        del original, fit
        meter.mark("trials/selected-model-strict-load")
        data = SimpleNamespace(spec=spec)
        cascade = CascadeStages(data, {"train": {None: SimpleNamespace(input_sha256=FEATURE_IDENTITY)}},
                                cfg, torch.device("cpu"), 0)
        chosen = {}
        checkpoints = comparison["binding"]["model"]
        for i, role in enumerate(ROLES):
            pin = progress["best"][role]
            require(checkpoints[role] == pin["sha256"] and pin["name"] == f"best-{role}-e{pin['epoch']:03d}.pt"
                    and files[pin["name"]] == {key: pin[key] for key in ("bytes", "sha256")},
                    "selected checkpoint role pin")
            candidate = read_weight(view, pin["name"], files)
            require(candidate["binding"] == binding and candidate["stage"] == role and candidate["epoch"] == pin["epoch"]
                    and candidate["score"] == pin["score"] and finite_tensors(candidate["model"])
                    and candidate["dependencies"] == {key: checkpoints[key] for key in ROLES[:i]},
                    "selected checkpoint schema/dependencies")
            stage = (FrameStage(cfg["frontend"], len(spec["joint_names"]), torch.device("cpu"))
                     if role == "frontend" else cascade.make(role, checkpoints["body"]))
            stage.model.load_state_dict(candidate["model"], strict=True)
            chosen[role] = {key: pin[key] for key in ("name", "sha256", "epoch", "bytes")}
            if role == "frontend":
                feature_binding = {"data": binding["input"], "frontend_weight_sha256": frontend_fingerprint(stage.model),
                                   "frontend": ARCHITECTURE, "size": [128, 128], "padding": "post_zero_time_invalid_v1"}
                require(digest(feature_binding) == FEATURE_IDENTITY, "A frontend fingerprint/feature identity")
            if role == "body":
                chosen[role]["candidate_parameter_targets"] = body_targets(stage.model)
    return {"cursor": CURSOR, "files_sha_checked": len(files), "state": fit_report, "selected": chosen,
            "feature_identity": FEATURE_IDENTITY, "resume_scope": "stored_gpu_state_cpu_load_only"}


def resume_math(workspace):
    """Use the existing checkpoint fit helpers on an isolated, deterministic CPU update."""
    def make(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        return SimpleNamespace(model=model, optimizer=opt, scheduler=sched,
                               scaler=torch.amp.GradScaler("cuda", enabled=False), step=0)

    def update(stage):
        require(not torch.cuda.is_available(), "CPU-only synthetic resume expects CUDA hidden")
        stage.optimizer.zero_grad(set_to_none=True)
        loss = stage.model(torch.rand((4, 3))).square().mean() * (0.5 + random.random())
        loss.backward()
        stage.optimizer.step()
        stage.scheduler.step()
        stage.step += 1
        return loss.detach()

    first = make(71)
    update(first)
    path = workspace / "synthetic-fit.pt"
    torch.save({"fit": _fit_state(first), "step": first.step}, path)
    expected = update(first)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    second = make(11)
    _restore_state(second, loaded["fit"], loaded["step"])
    got = update(second)
    require(torch.equal(got, expected) and second.scheduler.state_dict() == first.scheduler.state_dict()
            and all(torch.equal(value, second.model.state_dict()[name])
                    for name, value in first.model.state_dict().items()),
            "synthetic CPU uninterrupted/resumed equality")
    return {"resume_from": loaded["step"], "verified_update": second.step,
            "optimizer": "AdamW", "same_loss_and_weights": True}


def selection_fixture(trial_plan):
    parent = trial_plan["parent_macro_mm"]

    def row(method, seed, offset):
        return {"method": method, "seed": seed, "trial_plan_sha256": trial_plan["content_sha256"],
                "checkpoint_sha256": f"{seed:064x}", "status": "completed",
                "comparison_common": trial_plan["common"], "updates": 3097,
                "training": {"learning_rate": 5e-5, "batch_size": 1, "epoch": 1},
                "scenario_scores": {scene: {metric: parent + offset for metric in METRICS} for scene in SCENARIOS}}

    initial = [row(method, seed, j - 2) for j, method in enumerate(BODY) for seed in INITIAL]
    extra = [row(method, seed, j - 2) for j, method in enumerate(BODY[:2]) for seed in ADDITIONAL]
    require(summarize(trial_plan, initial[:-1], phase="screening")["status"] == "awaiting_screening_rows"
            and summarize(trial_plan, initial, phase="screening")["shortlist"] == list(BODY[:2]),
            "screening accepts only paired seed sets")
    need_target = summarize(trial_plan, initial + extra, phase="final")
    chosen = summarize(trial_plan, initial + extra, phase="final", target_verdict=dict.fromkeys(BODY[:2], True))
    require(need_target["status"] == "awaiting_target_metric_verdict"
            and chosen["selected_method"] == BODY[0] and chosen["candidate"]["seed"] == min(INITIAL),
            "final selection and checkpoint tie break")
    return {"screening_rows": len(initial), "extended_rows": len(initial) + len(extra),
            "fixture_only": True, "gate_before_target_verdict": need_target["status"]}


def run(kit, project, index, workspace):
    meter = project.meter
    meter.mark("trials/original-evidence")
    _, diagnostic, step, comparison = native_evidence(kit, project, index)
    path = f"{A}/checkpoints/{step['archive']['name']}"
    copy = staged_archive(project, path, step["archive"], workspace)
    spec = json.loads((kit / "frozen/reference/spec.json").read_bytes())
    actual = audit_checkpoint(copy, step, diagnostic, comparison, spec, meter)
    meter.mark("trials/synthetic-save-resume-and-selection")
    trial_plan = plan(step["identity"], comparison["binding"]["model"], comparison["macro_score_mm"])
    state_math = resume_math(workspace)
    selection = selection_fixture(trial_plan)
    write_json(workspace / "trial-plan.json", trial_plan)
    return {"checkpoint_zip": {"path": path, **step["archive"]}, "checkpoint": actual,
            "cpu_resume": state_math, "trial_plan_sha256": trial_plan["content_sha256"],
            "selection_contract": selection}

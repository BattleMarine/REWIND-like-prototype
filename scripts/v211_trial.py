"""Deterministic body output-head training and compact resumable state."""

import hashlib
import io

import torch
from rewind_archive.contracts import canonical, digest
from rewind_m1.training import restore_rng, rng_state


METHOD = "v2.1.1"
SEEDS = (42, 71, 103)
TRAINING = {"learning_rate": 5e-5, "batch_size": 1, "epoch": 1}
UPDATES = 3097
HEAD = ("final_head.weight", "final_head.bias")
TRAJECTORY = "rewind_v2_tuning_order_noise_v1"
STATE = "rewind_v211_body_output_head_fit_v1"


def trajectory_seed(seed, purpose, step=0):
    raw = hashlib.sha256(canonical([TRAJECTORY, purpose, seed, step])).digest()
    return int.from_bytes(raw[:8], "big") % (2**63)


def order(keys, seed):
    generator = torch.Generator(device="cpu").manual_seed(trajectory_seed(seed, "clip-order"))
    return [keys[index] for index in torch.randperm(len(keys), generator=generator).tolist()]


def head(stage):
    params = dict(stage.model.named_parameters())
    result = {name: params[name] for name in HEAD}
    if tuple(p.numel() for p in result.values()) != (1536, 3):
        raise ValueError("v211_parent_head_shape")
    return result


def begin_fit(stage):
    stage.model.requires_grad_(False)
    weights = head(stage)
    for weight in weights.values():
        weight.requires_grad_(True)
    stage.optimizer = torch.optim.Adam(weights.values(), lr=TRAINING["learning_rate"])
    stage.scheduler = torch.optim.lr_scheduler.StepLR(stage.optimizer, stage.config["scheduler_step"],
                                                      stage.config["scheduler_gamma"])
    stage.scaler = torch.amp.GradScaler("cuda", enabled=False)
    stage.step = 0


def update(stage, clip, seed):
    # The frozen TeacherRun samples k and then epsilon. Model construction and
    # future adapter initialization cannot advance this per-update stream.
    devices = [stage.device.index or 0] if stage.device.type == "cuda" else []
    value = trajectory_seed(seed, "teacher-update", stage.step)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(value)
        if devices:
            torch.cuda.manual_seed(value)
        return stage.update(clip)


def backbone_sha(stage):
    hasher = hashlib.sha256()
    for name, tensor in stage.model.state_dict().items():
        if name in HEAD:
            continue
        hasher.update(canonical([name, list(tensor.shape), str(tensor.dtype)]))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def save_state(stage, binding, path, last_loss):
    torch.save({"schema": STATE, "binding_sha256": digest(binding), "step": stage.step,
                "head": {name: p.detach().cpu() for name, p in head(stage).items()},
                "optimizer": stage.optimizer.state_dict(), "scheduler": stage.scheduler.state_dict(),
                "scaler": stage.scaler.state_dict(), "rng": rng_state(), "last_loss": last_loss}, path)


def load_state(stage, binding, raw, expected_step):
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (state["schema"] != STATE or state["binding_sha256"] != digest(binding)
            or state["step"] != expected_step or set(state["head"]) != set(HEAD)):
        raise ValueError("v211_checkpoint_identity")
    with torch.no_grad():
        for name, weight in head(stage).items():
            saved = state["head"][name]
            if saved.shape != weight.shape or saved.dtype != torch.float32 or not torch.isfinite(saved).all():
                raise ValueError("v211_checkpoint_head")
            weight.copy_(saved)
    stage.optimizer.load_state_dict(state["optimizer"])
    stage.scheduler.load_state_dict(state["scheduler"])
    stage.scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    stage.step = expected_step
    return state["last_loss"]


def scenario_scores(metrics):
    return {name: {f"{role}_mpjpe_mm": entry["official"][role]["mpjpe_mm"]
                   for role in ("body", "hand_global")}
            for name, entry in metrics["per_scenario"].items()}


def measurement(plan, seed, checkpoint, metrics):
    return {"method": METHOD, "seed": seed, "trial_plan_sha256": plan["content_sha256"],
            "status": "completed", "comparison_common": plan["common"],
            "updates": UPDATES, "training": TRAINING, "checkpoint_sha256": checkpoint,
            "scenario_scores": scenario_scores(metrics)}

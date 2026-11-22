"""Body x0/v adapters with matched x0-space objective and explicit initialization."""

import hashlib
import math

import torch
from torch import nn

from rewind_archive.contracts import canonical
from rewind_m2.diffusion import DiffusionSchedule
from rewind_v1.teacher import motion_losses


ARMS = ("x0", "v")
FINAL = ("final_head.weight", "final_head.bias")
FRAME = ("frame_head.weight", "frame_head.bias")
SEED = 42
UPDATES = 24
EVAL_UPDATES = (0, 12, 24)
LEARNING_RATE = 5e-5
TRAJECTORY = "rewind_v24_matched_pilot_v1"


def tensor_sha(items):
    hasher = hashlib.sha256()
    for name, tensor in sorted(items):
        hasher.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def model_sha(model):
    return tensor_sha(model.state_dict().items())


def stream_seed(purpose, step=0, *, seed=SEED, trajectory=TRAJECTORY):
    raw = hashlib.sha256(canonical([trajectory, seed, purpose, step])).digest()
    return int.from_bytes(raw[:8], "big") % 2**63


def common_head(weight, bias, *, seed=SEED, trajectory=TRAJECTORY):
    if weight.shape != (3, 512) or bias.shape != (3,) or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("v24_A_final_head_shape")
    generator = torch.Generator(device="cpu").manual_seed(stream_seed("final-head", seed=seed, trajectory=trajectory))
    bound = 1 / math.sqrt(weight.shape[1])
    return {name: torch.empty(tensor.shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
            for name, tensor in zip(FINAL, (weight, bias), strict=True)}


def setup(stage, *, seed=SEED, trajectory=TRAJECTORY):
    parameters = dict(stage.model.named_parameters())
    if len(parameters) != 99 or sum(p.numel() for p in parameters.values()) != 15_010_822:
        raise ValueError("v24_A_body_parameter_layout")
    original = stage.model.state_dict()
    backbone = tensor_sha((name, value) for name, value in original.items() if name not in FINAL)
    frame = tensor_sha((name, original[name]) for name in FRAME)
    old_head = tensor_sha((name, original[name]) for name in FINAL)
    initial_head = common_head(*(parameters[name] for name in FINAL), seed=seed, trajectory=trajectory)
    with torch.no_grad():
        for name, value in initial_head.items():
            parameters[name].copy_(value)
    current = stage.model.state_dict()
    if (tensor_sha((name, value) for name, value in current.items() if name not in FINAL) != backbone
            or tensor_sha((name, current[name]) for name in FINAL) == old_head
            or tensor_sha((name, current[name]) for name in FINAL) != tensor_sha(initial_head.items())):
        raise ValueError("v24_init_replaced_only_final_head")
    stage.model.requires_grad_(True)
    stage.optimizer = torch.optim.Adam(stage.model.parameters(), lr=LEARNING_RATE)
    stage.scheduler = torch.optim.lr_scheduler.StepLR(stage.optimizer, stage.config["scheduler_step"], stage.config["scheduler_gamma"])
    stage.scaler = torch.amp.GradScaler("cuda", enabled=False)
    stage.step = 0
    return {"body_initial_sha256": model_sha(stage.model), "compatible_A_backbone_sha256": backbone,
            "preserved_A_frame_head_sha256": frame, "reset_final_head_sha256": tensor_sha(initial_head.items()),
            "replaced_A_final_head_sha256": old_head,
            "head_initialization_seed": stream_seed("final-head", seed=seed, trajectory=trajectory)}


def coefficients(schedule, k, like):
    alpha = schedule.alpha(k, like)
    return alpha.sqrt(), (1 - alpha).sqrt()


def velocity_target(clean, epsilon, sqrt_alpha, sqrt_beta):
    return sqrt_alpha * epsilon - sqrt_beta * clean


def decode_velocity(noisy, prediction, sqrt_alpha, sqrt_beta):
    return sqrt_alpha * noisy - sqrt_beta * prediction


class X0View(nn.Module):
    """Adapt only the final output for the existing x0 DDIM sampler."""

    def __init__(self, model, schedule, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("v24_unknown_arm")
        self.model, self.schedule, self.arm = model, schedule, arm

    def forward(self, noisy, features, camera, k, time_valid, positions=None):
        final, frame = self.model(noisy, features, camera, k, time_valid, positions=positions)
        if self.arm == "v":
            final = decode_velocity(noisy, final, *coefficients(self.schedule, k, noisy))
            final = torch.where(time_valid[..., None, None], final, 0.)
        return final, frame


def draw(schedule, step, shape, *, seed=SEED, trajectory=TRAJECTORY):
    generator = torch.Generator(device="cpu").manual_seed(stream_seed("teacher-k-epsilon", step, seed=seed, trajectory=trajectory))
    k = torch.randint(1, schedule.steps + 1, (shape[0],), generator=generator)
    epsilon = torch.randn(shape, generator=generator, dtype=torch.float32)
    return k, epsilon, tensor_sha((("k", k), ("epsilon", epsilon)))


def loss(stage, arm, clip, k, epsilon):
    if arm not in ARMS or clip.split != "train":
        raise ValueError("v24_train_scope_or_arm")
    features, camera, time_valid = stage._inputs(clip)
    target_mm, target_valid = clip.target_mm.to(stage.device), clip.target_valid.to(stage.device)
    mask = target_valid & time_valid[..., None]
    scale = stage.config["target_scale_mm"]
    clean = torch.where(mask[..., None], target_mm / scale, 0.)
    schedule = stage.diffusion
    k, epsilon = k.to(stage.device), epsilon.to(stage.device)
    noisy = schedule.q_sample(clean, epsilon, k)
    raw, frame_x0 = stage.model(noisy, features, camera, k, time_valid)
    final_x0 = raw if arm == "x0" else decode_velocity(noisy, raw, *coefficients(schedule, k, noisy))
    objective, components = motion_losses(final_x0, frame_x0, target_mm, target_valid, time_valid,
                                          scale_mm=scale, frame_weight=stage.config["frame_loss_weight"],
                                          velocity_weight=stage.config["velocity_loss_weight"],
                                          time_seconds=clip.seconds.to(stage.device))
    if arm == "v":
        a, b = coefficients(schedule, k, noisy)
        # L_x0 equals mean((1-alpha_bar) * (v_hat-v_target)^2) on the same mask.
        # Diagnostics expose the equivalent weighted term and raw v MSE. They carry no extra gradient.
        with torch.no_grad():
            error = raw - velocity_target(clean, epsilon, a, b)
            raw_v = error[mask].square().mean()
            weighted = (b * error)[mask].square().mean()
            if not torch.allclose(weighted, components["position"], atol=2e-5, rtol=2e-5):
                raise ValueError("v24_weighted_v_x0_loss_mismatch")
            components.update(v_raw=raw_v, v_position_equivalent=weighted)
    return objective, components, int(mask.sum())


def update(stage, arm, clip, noise):
    stage.model.train()
    stage.optimizer.zero_grad(set_to_none=True)
    objective, values, count = loss(stage, arm, clip, *noise)
    objective.backward()
    gradients = [p.grad for p in stage.model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("v24_missing_or_nonfinite_body_gradients")
    named = dict(stage.model.named_parameters())
    norms = {head: float(torch.linalg.vector_norm(named[f"{head}.weight"].grad))
             for head in ("final_head", "frame_head")}
    if stage.step == 0 and any(not 0 < x < math.inf for x in norms.values()):
        raise ValueError("v24_first_valid_update_head_gradient")
    stage.optimizer.step()
    stage.scheduler.step()
    stage.step += 1
    return {"total": float(objective.detach()), "valid_joint_observations": count,
            "head_gradient_l2": norms, **{key: float(value.detach()) for key, value in values.items()}}


@torch.no_grad()
def sample(stage, arm, clip, noise):
    features, camera, time_valid = stage._inputs(clip)
    if stage.model.training or stage.config["sampling_steps"] != 10:
        raise ValueError("v24_expected_eval_DDIM10")
    model = X0View(stage.model, stage.diffusion, arm).eval()
    return stage.diffusion.sample(model, features, camera, time_valid, noise.to(stage.device), 10) * stage.config["target_scale_mm"]


def numerical():
    schedule = DiffusionSchedule(1000, .0001, .02)
    generator = torch.Generator().manual_seed(stream_seed("numerical"))
    rand = lambda: torch.randn((3, 5, 2, 3), generator=generator, dtype=torch.float64)
    clean, epsilon = rand(), rand()
    k = torch.tensor([1, 500, 1000])
    a, b = coefficients(schedule, k, clean)
    xt = schedule.q_sample(clean, epsilon, k)
    v = velocity_target(clean, epsilon, a, b)
    recovered = decode_velocity(xt, v, a, b)
    eps_recovered = b * xt + a * v
    error = rand() * .25
    prediction = decode_velocity(xt, v + error, a, b)
    valid = torch.tensor([[True, True, False, False, False], [True, False, True, True, False],
                          [True, True, True, True, True]])[..., None].expand(-1, -1, 2)
    position = (prediction - clean)[valid].square().mean()
    equivalent = (b * error)[valid].square().mean()

    class Oracle(nn.Module):
        def __init__(self, arm):
            super().__init__()
            self.arm = arm

        def forward(self, noisy, features, camera, step, time_valid, positions=None):
            a, b = coefficients(schedule, step, noisy)
            final = clean if self.arm == "x0" else (a * noisy - clean) / b
            return final, clean + .25

    temporal = valid[..., 0]
    direct = X0View(Oracle("x0"), schedule, "x0").eval()
    adapted = X0View(Oracle("v"), schedule, "v").eval()
    expected = schedule.sample(direct, None, None, temporal, epsilon, 10)
    simulated = schedule.sample(adapted, None, None, temporal, epsilon, 10)
    _, frame = adapted(xt, None, None, k, temporal)
    report = {"x0_max_abs": float((clean - recovered).abs().max()), "epsilon_max_abs": float((epsilon - eps_recovered).abs().max()),
              "masked_position_loss_abs_difference": float((position - equivalent).abs()),
              "DDIM10_output_max_abs": float((expected - simulated).abs().max())}
    if (any(value > 1e-12 for value in report.values()) or not torch.equal(frame, clean + .25)
            or bool(simulated[~temporal].any())):
        raise ValueError(f"v24_numerical_equivalence: {report}")
    return {"precision": "float64", "frame_x0_passthrough": True, "time_mask_preserved": True,
            "sampler_timesteps": schedule.timesteps(10), **report}

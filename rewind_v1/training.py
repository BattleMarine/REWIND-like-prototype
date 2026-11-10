"""Small-run v1 teacher update, cascade and checkpoint interfaces."""

import copy
import hashlib
import io
from pathlib import Path
import platform
import time
from contextlib import contextmanager

import torch

from rewind_m1.local_rgb import _hash, _regular_read
from rewind_m1.training import restore_rng, rng_state, save_checkpoint
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.training import source_identity as m2_identity
from rewind_m3.cascade import UpperCondition
from rewind_m3.models import BoundHand
from rewind_prepare.temporal_cache import producer_identity
from .device import require_a100
from .teacher import V1Body, V1Hand, motion_losses


FORMAT = "rewind_v1_teacher_checkpoint_v1"
MAX_CHECKPOINT_BYTES = 256 * 1024**2


def gradient_l2(model):
    norms = [parameter.grad.detach().norm() for parameter in model.parameters() if parameter.grad is not None]
    value = float(torch.stack(norms).norm()) if norms else 0.
    if not 0 <= value < float("inf"):
        raise ValueError("rewind_v1_nonfinite_gradient_norm")
    return value


def _source():
    root = Path(__file__).resolve().parent.parent
    hashes = {**m2_identity()["code_sha256"], **producer_identity()}
    for package in ("rewind_v1", "rewind_m3"):
        for path in sorted((root / package).glob("*.py")):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


class TeacherRun:
    def __init__(self, config, *, joints, edges, source, input_sha256, upper_joints=0,
                 body_checkpoint_sha256=None, device="cpu"):
        self.device = torch.device(device)
        accelerator = None
        if self.device.type != "cpu":
            accelerator = require_a100(self.device)
            self.device = torch.device(accelerator["device"])
        if source not in ("gt_2d_diagnostic", "predicted_2d") or not _hash(input_sha256):
            raise ValueError("rewind_v1_training_input_identity")
        if (upper_joints and not _hash(body_checkpoint_sha256)) or (not upper_joints and body_checkpoint_sha256):
            raise ValueError("rewind_v1_hand_body_identity")
        self.spec = dict(joints=joints, edges=edges, source=source, input_sha256=input_sha256,
                         upper_joints=upper_joints, body_checkpoint_sha256=body_checkpoint_sha256,
                         device=str(self.device))
        self.config = copy.deepcopy(config)
        self.source_code = _source()
        self.environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                            "cuda_runtime": torch.version.cuda, "accelerator": accelerator,
                            "precision": "float32", "threads": torch.get_num_threads()}
        self.runtime = {}
        self.measure_gradient = False
        self.model = (V1Hand(config, joints, edges, upper_joints) if upper_joints else
                      V1Body(config, joints, edges)).to(self.device)
        self.diffusion = DiffusionSchedule(config["diffusion_steps"], config["beta_start"], config["beta_end"])
        self.diffusion.timesteps(config["sampling_steps"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config["scheduler_step"], config["scheduler_gamma"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.step = 0
        self.checkpoint_sha256 = None

    @contextmanager
    def _timed(self, name):
        start = time.perf_counter()
        if self.device.type == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(self.device))
        yield
        if self.device.type == "cuda":
            end.record(torch.cuda.current_stream(self.device))
            end.synchronize()
            self.runtime[name + "_cuda_stream_seconds"] = begin.elapsed_time(end) / 1000
        self.runtime[name + "_wall_seconds"] = time.perf_counter() - start

    def _inputs(self, clip):
        if clip.source != self.spec["source"] or clip.input_sha256 != self.spec["input_sha256"]:
            raise ValueError("rewind_v1_teacher_observation_source")
        return (clip.features.to(self.device), clip.camera.to(self.device), clip.time_valid.to(self.device))

    def _condition(self, upper, *, training):
        if not self.spec["upper_joints"]:
            if upper is not None:
                raise ValueError("rewind_v1_body_rejects_upper")
            return self.model
        if not isinstance(upper, UpperCondition):
            raise ValueError("rewind_v1_upper_condition_required")
        mode = "gt_upper_train" if training else "predicted_frozen_body"
        upper.require(mode, {"checkpoint_sha256": self.spec["body_checkpoint_sha256"]})
        if not isinstance(upper.values_mm, torch.Tensor) or not isinstance(upper.valid, torch.Tensor):
            raise ValueError("rewind_v1_upper_condition_tensor")
        return BoundHand(self.model, upper.values_mm.to(self.device) / self.config["target_scale_mm"],
                         upper.valid.to(self.device))

    def update(self, clip, *, upper=None):
        if clip.split != "train":
            raise ValueError("rewind_v1_update_requires_train_split")
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("rewind_v1_frozen_teacher")
        with self._timed("train_input_transfer"):
            features, camera, time_valid = self._inputs(clip)
            conditioned = self._condition(upper, training=True)
            target = clip.target_mm.to(self.device)
            target_valid = clip.target_valid.to(self.device)
            seconds = clip.seconds.to(self.device)
        with self._timed("train_compute"):
            metrics = self._update(features, camera, time_valid, conditioned, target, target_valid, seconds)
        return metrics

    def _update(self, features, camera, time_valid, conditioned, target, target_valid, seconds):
        mask = target_valid & time_valid[..., None]
        scale = self.config["target_scale_mm"]
        k = torch.randint(1, self.config["diffusion_steps"] + 1, (len(target),), device=self.device)
        clean = torch.where(mask[..., None], target / scale, 0.)
        noisy = self.diffusion.q_sample(clean, torch.randn_like(clean), k)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction, frame = conditioned(noisy, features, camera, k, time_valid)
        loss, metrics = motion_losses(prediction, frame, target, mask, time_valid, scale_mm=scale,
                                      frame_weight=self.config["frame_loss_weight"],
                                      velocity_weight=self.config["velocity_loss_weight"],
                                      time_seconds=seconds)
        if not torch.isfinite(loss):
            raise ValueError("rewind_v1_nonfinite_teacher_loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("rewind_v1_nonfinite_teacher_gradient")
        gradient = gradient_l2(self.model) if self.measure_gradient else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        self.checkpoint_sha256 = None
        result = {name: float(value.detach()) for name, value in metrics.items()}
        if gradient is not None:
            result["first_update_gradient_l2"] = gradient
        return result

    @torch.no_grad()
    def predict(self, clip, noise, *, upper=None):
        if self.model.training:
            raise ValueError("rewind_v1_teacher_requires_eval_mode")
        with self._timed("evaluation_input_transfer"):
            features, camera, time_valid = self._inputs(clip)
            conditioned = self._condition(upper, training=False).eval()
            noise = noise.to(self.device)
        with self._timed("evaluation_compute"):
            prediction = self.diffusion.sample(conditioned, features, camera, time_valid, noise,
                                               self.config["sampling_steps"])
        return prediction * self.config["target_scale_mm"]

    def frozen_upper(self, clip, noise, indices):
        if (self.spec["upper_joints"] or not self.checkpoint_sha256
                or any(p.requires_grad or p.grad is not None for p in self.model.parameters())):
            raise ValueError("rewind_v1_frozen_body_checkpoint_required")
        predicted = self.predict(clip, noise)[..., indices, :]
        valid = clip.time_valid.to(predicted.device)[..., None].expand(predicted.shape[:-1])
        return UpperCondition(predicted, valid, "predicted_frozen_body", self.checkpoint_sha256)

    def save(self, path):
        save_checkpoint({
            "format_version": FORMAT, "config": self.config, "spec": self.spec, "source_code": self.source_code,
            "environment": self.environment, "step": self.step, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(), "rng": rng_state(),
        }, path)

    def resume(self, path, expected_sha256):
        raw = _regular_read(Path(path), MAX_CHECKPOINT_BYTES)
        if not _hash(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("rewind_v1_checkpoint_pin")
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if (state["format_version"], state["config"], state["spec"], state["source_code"], state["environment"]) != (
            FORMAT, self.config, self.spec, self.source_code, self.environment
        ):
            raise ValueError("rewind_v1_checkpoint_identity")
        step = state["step"]
        if type(step) is not int or step < 0 or state["scheduler"]["last_epoch"] != step:
            raise ValueError("rewind_v1_checkpoint_step")
        before = rng_state()
        try:
            resumed = TeacherRun(self.config, **self.spec)
            resumed.model.load_state_dict(state["model"], strict=True)
            resumed.optimizer.load_state_dict(state["optimizer"])
            resumed.scheduler.load_state_dict(state["scheduler"])
            resumed.scaler.load_state_dict(state["scaler"])
            resumed.step = step
            resumed.checkpoint_sha256 = expected_sha256
            restore_rng(state["rng"])
        except BaseException:
            restore_rng(before)
            raise
        self.__dict__.update(resumed.__dict__)

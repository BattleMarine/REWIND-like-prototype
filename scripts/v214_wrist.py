"""Change the final body position penalty at the two valid wrists."""

from types import SimpleNamespace

import torch
from rewind_m1.metrics import masked_mse
from rewind_v1.training import TeacherRun
from scripts import v211_trial, v212_lora
from scripts.v211_trial import TRAINING, UPDATES


METHOD = "v2.1.4"
PARENTS = {strategy.METHOD: strategy for strategy in (v211_trial, v212_lora)}
BRANCH = {"prerequisite": "selected_v2.1.1_to_v2.1.3", "wrist_position_weight": 2,
          "control_wrist_position_weight": 1, "updates": UPDATES, **TRAINING}
FIT = "rewind_v214_wrist_fit_receipt_v1"
SUMMARY = "rewind_v214_wrist_validation_v1"


def strategy(method):
    parent = PARENTS[method]
    return SimpleNamespace(METHOD=METHOD, BRANCH=BRANCH, TRAINABLE=parent.TRAINABLE, FIT_SCHEMA=FIT,
                           ARTIFACT_PREFIX=parent.ARTIFACT_PREFIX, begin_fit=parent.begin_fit,
                           backbone_sha=parent.backbone_sha, save_state=parent.save_state, load_state=parent.load_state)


def position_loss(prediction, target, mask, scale, wrists, weight):
    pred_mm = prediction.float()*scale
    if weight == 1 or not mask[..., list(wrists)].any():
        return masked_mse(pred_mm, target, mask, scale)
    weights = torch.ones_like(mask, dtype=pred_mm.dtype)
    weights[..., list(wrists)] = weight
    residual = (pred_mm[mask]-target[mask])/scale
    valid_weights = weights[mask]
    return (residual.square()*valid_weights[:, None]).sum() / (3*valid_weights.sum())


class WristTeacher(TeacherRun):
    def _update(self, features, camera, time_valid, conditioned, target, target_valid, seconds):
        if self.wrist_weight == 1:
            return super()._update(features, camera, time_valid, conditioned, target, target_valid, seconds)
        if self.wrist_weight != 2 or self.config["velocity_loss_weight"] != 0:
            raise ValueError("v214_frozen_loss_scope")
        mask = target_valid & time_valid[..., None]
        scale = self.config["target_scale_mm"]
        k = torch.randint(1, self.config["diffusion_steps"] + 1, (len(target),), device=self.device)
        clean = torch.where(mask[..., None], target/scale, 0.)
        noisy = self.diffusion.q_sample(clean, torch.randn_like(clean), k)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction, frame = conditioned(noisy, features, camera, k, time_valid)
        position = position_loss(prediction, target, mask, scale, self.wrists, self.wrist_weight)
        frame_loss = masked_mse(frame.float()*scale, target, mask, scale)
        loss = position + self.config["frame_loss_weight"]*frame_loss
        if not torch.isfinite(loss):
            raise ValueError("v214_nonfinite_teacher_loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("v214_nonfinite_teacher_gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        self.checkpoint_sha256 = None
        return {"position": float(position.detach()), "frame": float(frame_loss.detach())}


def make_body(cascade, pipe, weight=2):
    names = cascade.body_names
    wrists = tuple(names.index(cascade.spec["wrists"][side]) for side in ("left", "right"))
    if len(set(wrists)) != 2:
        raise ValueError("v214_wrist_joint_layout")
    body = WristTeacher(cascade.cfg["teacher"], joints=len(names), edges=cascade._edges(names), source="predicted_2d",
                        input_sha256=cascade.input_sha256, device=cascade.device)
    body.wrists = wrists
    body.wrist_weight = weight
    pipe._load_best("body", body)
    return body

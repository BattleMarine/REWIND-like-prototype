"""Frame pretraining and stage operations for the bounded v1 runner."""

from contextlib import contextmanager
import time

import torch

from rewind_m1.frontend import heatmap_loss
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m1.training import restore_rng, rng_state
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.cascade import UpperCondition
from rewind_prepare.extracted import IMAGE_SIZE
from .data import attach_predicted_hands, gt_upper_train, hand_clip
from .frontend import PaperHeatmapFrontend
from .training import TeacherRun


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class FrameStage:
    def __init__(self, cfg, joints, device):
        self.device = device
        self.model = PaperHeatmapFrontend(joints).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg["scheduler_step"], cfg["scheduler_gamma"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.runtime = {}

    @contextmanager
    def measure(self, key):
        start = time.monotonic()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.runtime[key] = time.monotonic() - start

    def update(self, sample, image_size):
        self.model.train()
        with self.measure("frontend_train_input_transfer_wall_seconds"):
            images, xy, mask = (tensor.to(self.device) for tensor in sample)
        with self.measure("frontend_train_compute_wall_seconds"):
            return self._update(images, xy, mask, image_size)

    def _update(self, images, xy, mask, image_size):
        self.optimizer.zero_grad(set_to_none=True)
        if not _inside(xy, mask, image_size).any():
            self.optimizer.step()
            self.scheduler.step()
            return {"supervised_joint_views": 0, "no_supervision_batches": 1}
        logits, _, _ = self.model(images)
        loss, inside = heatmap_loss(logits, xy, mask, image_size, loss_kind="gaussian_cross_entropy")
        if not torch.isfinite(loss):
            raise ValueError("rewind_v1_frontend_nonfinite_loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("rewind_v1_frontend_nonfinite_gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return {"cross_entropy": float(loss.detach()), "supervised_joint_views": int(inside.sum())}


def _inside(xy, valid, size):
    return valid & (xy[..., 0] >= 0) & (xy[..., 0] <= size[0]-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= size[1]-1)


def accumulate(destination, values, prefix="", namespace=""):
    for name, value in values.items():
        if name.startswith(prefix):
            key = namespace + name
            destination[key] = destination.get(key, 0.) + value


def frame_batch(data, keys, split, size, budget, timing):
    result = []
    for key in keys:
        budget.check()
        sample, stats = data.frame(key, split, size)
        accumulate(timing, stats)
        result.append(sample)
    return tuple(torch.stack(values) for values in zip(*result))


@torch.no_grad()
def evaluate_frontend(stage, data, config, budget, timing):
    stage.model.eval()
    size = config["size"]
    destination = IMAGE_SIZE if data.identity["data_kind"] == "dataset_sample" else size
    keys = data.frames["validation"]
    sums = {}
    for start in range(0, len(keys), config["batch_size"]):
        batch = frame_batch(data, keys[start:start+config["batch_size"]], "validation", size, budget, timing)
        with stage.measure("frontend_evaluation_input_transfer_wall_seconds"):
            images, xy, valid = (v.to(stage.device) for v in batch)
        with stage.measure("frontend_evaluation_compute_wall_seconds"):
            _, pred, _ = stage.model(images)
            inside = _inside(xy, valid, size)
            error = torch.linalg.vector_norm(scale_xy(pred - xy, size, destination), dim=-1)
        accumulate(timing, stage.runtime, "frontend_evaluation_")
        if not torch.isfinite(error[inside]).all():
            raise ValueError("rewind_v1_frontend_validation_nonfinite")
        accumulate(sums, {"inside_error": float(error[inside].double().sum()), "inside_count": int(inside.sum()),
                          "valid_count": int(valid.sum())})
    if not sums["inside_count"]:
        raise ValueError("rewind_v1_frontend_validation_no_inside_joints")
    return {"score": sums["inside_error"] / sums["inside_count"], "unit": "native_px" if size != destination else "input_px",
            "frames": len(keys), "inside_joint_views": sums["inside_count"], "valid_joint_views": sums["valid_count"]}


class CascadeStages:
    def __init__(self, data, clips, cfg, device, feature_bytes):
        self.data, self.clips, self.cfg, self.device = data, clips, cfg, device
        self.spec = data.spec
        self.body_names = self.spec["body_joint_names"]
        self.body_idx = [self.spec["joint_names"].index(n) for n in self.body_names]
        self.upper_idx = [self.body_names.index(n) for n in self.spec["upper_joint_names"]]
        self.fingers = [name for side in ("left", "right") for name in self.spec["hands"][side][1:]]
        self.finger_idx = [self.spec["joint_names"].index(n) for n in self.fingers]
        self.input_sha256 = next(iter(clips["train"].values())).input_sha256
        self.body_validation = {}
        self.retained_bytes = feature_bytes

    def make(self, role, body_pin=None):
        names = self.body_names if role == "body" else self.fingers
        edges = [(names.index(a), names.index(b)) for a, b in self.spec["edges"] if a in names and b in names]
        kwargs = {} if role == "body" else dict(upper_joints=len(self.upper_idx), body_checkpoint_sha256=body_pin)
        return TeacherRun(self.cfg["teacher"], joints=len(names), edges=edges, source="predicted_2d",
                          input_sha256=self.input_sha256, device=self.device, **kwargs)

    def train(self, role, stage, key):
        clip = self.clips["train"][key]
        if role == "body":
            return stage.update(clip.select_joints(self.body_idx))
        return stage.update(hand_clip(clip, self.spec), upper=gt_upper_train(clip, self.spec))

    def _noise(self, key, joints, seed_offset):
        sequence, start = key
        return torch.stack([keyed_frame_noise(sequence, i, joints, self.cfg["seed"] + seed_offset)
                            for i in range(start, start + self.data.width)])[None]

    @torch.no_grad()
    def evaluate(self, role, stage, budget, timing, frozen_body=None):
        stage.model.eval()
        sums = {}
        for key, clip in self.clips["validation"].items():
            budget.check()
            body_clip = clip.select_joints(self.body_idx)
            if role == "body":
                predicted = stage.predict(body_clip, self._noise(key, len(self.body_names), 100)).cpu()
                self._errors(sums, "body", predicted, body_clip)
            else:
                if key not in self.body_validation:
                    reserve = self.data.width * len(self.body_names) * 3 * 4
                    if self.retained_bytes + reserve > self.cfg["limits"]["feature_bytes"]:
                        raise ValueError("rewind_v1_run_feature_limit")
                    body = frozen_body.predict(body_clip, self._noise(key, len(self.body_names), 100)).cpu()
                    self.body_validation[key] = body
                    self.retained_bytes += reserve
                    timing["retained_feature_and_body_tensor_bytes"] = self.retained_bytes
                    accumulate(timing, frozen_body.runtime, "evaluation_", "hand_frozen_body_")
                body = self.body_validation[key]
                values = body[..., self.upper_idx, :]
                valid = clip.time_valid[..., None].expand(values.shape[:-1])
                upper = UpperCondition(values, valid, "predicted_frozen_body", frozen_body.checkpoint_sha256)
                local = hand_clip(clip, self.spec)
                predicted = stage.predict(local, self._noise(key, len(self.fingers), 200), upper=upper).cpu()
                self._errors(sums, "hand_local", predicted, local)
                self._errors(sums, "hand_global", attach_predicted_hands(body, predicted, self.spec),
                             clip.select_joints(self.finger_idx))
                self._errors(sums, "frozen_body", body, body_clip)
            accumulate(timing, stage.runtime, "evaluation_", role + "_")
        metrics = {}
        for name, (total, count) in sums.items():
            metrics[name] = {"mpjpe_mm": total / count if count else None, "valid_joints": count}
        score = metrics["body" if role == "body" else "hand_global"]["mpjpe_mm"]
        if score is None:
            raise ValueError("rewind_v1_teacher_validation_no_targets")
        return {"score": score, "unit": "mm", "frames": len(self.clips["validation"]) * self.data.width,
                "clips": len(self.clips["validation"]), "joint_metrics": metrics,
                "upper_source": "predicted_frozen_body" if role == "hand" else None,
                "body_checkpoint_sha256": frozen_body.checkpoint_sha256 if frozen_body else None}

    @staticmethod
    def _errors(sums, label, predicted, clip):
        valid = clip.target_valid & clip.time_valid[..., None]
        distances = torch.linalg.vector_norm(predicted[valid] - clip.target_mm[valid], dim=-1)
        if not torch.isfinite(distances).all():
            raise ValueError("rewind_v1_validation_nonfinite")
        previous, count = sums.get(label, (0., 0))
        sums[label] = previous + float(distances.double().sum()), count + int(valid.sum())

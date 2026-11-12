"""Paired body/hand updates on one GT2D+GT-upper oracle clip, with fixed-noise diagnostics."""

from contextlib import suppress
from dataclasses import replace
import json
from pathlib import Path
import platform
import time

from PIL import Image, ImageDraw, ImageFont
import torch

from rewind_archive.contracts import canonical
from rewind_m1.training import seed_run
from rewind_m1.local_rgb import _regular_read
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_prepare.converter import sha
from rewind_prepare.temporal_cache import _atomic, _check_part, _locked
from scripts.visualize_archvis_v1 import _draw_joints, _shared_3d_groups, DARK
from .data import attach_predicted_hands, gt_upper_train, hand_clip
from .oracle_data import ORACLE, ROOT, RUN_ID, source_identity
from .run_stages import CascadeStages, preserve_rng
from .runner import Budget, _fit_state, _publish, _read_checkpoint, _restore_state
from .teacher import V1Body, V1Hand, motion_losses
from .training import gradient_l2


FORMAT = "rewind_v14_oracle_overfit_v1"


def validate_config(config):
    expected = json.loads((ROOT / "configs/rewind_v14_oracle_overfit.json").read_bytes())
    if config != expected:
        raise ValueError("v14_fixed_overfit_config")


class OracleStage:
    def __init__(self, cfg, names, edges, upper_joints, device):
        self.cfg, self.device, self.step = cfg, device, 0
        self.upper_joints = upper_joints
        self.model = (V1Hand(cfg, len(names), edges, upper_joints) if upper_joints else V1Body(cfg, len(names), edges)).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg["scheduler_step"], cfg["scheduler_gamma"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.diffusion = DiffusionSchedule(cfg["diffusion_steps"], cfg["beta_start"], cfg["beta_end"])

    def condition(self, upper):
        if not self.upper_joints and upper is not None:
            raise ValueError("v14_body_rejects_gt_upper")
        if self.upper_joints and upper is None:
            raise ValueError("v14_hand_requires_gt_upper_oracle")
        if upper is None:
            return self.model
        upper.require("gt_upper_train", {})
        return BoundHand(self.model, upper.values_mm / self.cfg["target_scale_mm"], upper.valid)

    def update(self, clip, upper, measure_gradient):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        mask = clip.target_valid & clip.time_valid[..., None]
        clean = torch.where(mask[..., None], clip.target_mm / self.cfg["target_scale_mm"], 0.)
        k = torch.randint(1, self.cfg["diffusion_steps"]+1, (len(clean),), device=self.device)
        noisy = self.diffusion.q_sample(clean, torch.randn_like(clean), k)
        prediction, frame = self.condition(upper)(noisy, clip.features, clip.camera, k, clip.time_valid)
        loss, components = motion_losses(prediction, frame, clip.target_mm, clip.target_valid, clip.time_valid,
            scale_mm=self.cfg["target_scale_mm"], frame_weight=self.cfg["frame_loss_weight"],
            velocity_weight=self.cfg["velocity_loss_weight"], time_seconds=clip.seconds)
        if not torch.isfinite(loss):
            raise ValueError("v14_nonfinite_loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("v14_nonfinite_gradient")
        metrics = {key: float(value.detach()) for key, value in components.items()}
        if measure_gradient:
            metrics["gradient_l2"] = gradient_l2(self.model)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return metrics


def _metrics(predicted, clip, edges):
    valid = clip.target_valid & clip.time_valid[..., None]
    if not valid.any() or not torch.isfinite(predicted[valid]).all():
        raise ValueError("v14_evaluation_validity")
    distances = (predicted-clip.target_mm).norm(dim=-1)
    shape = CascadeStages._shape(predicted, clip, edges)
    frames, bones = shape["frames"], shape["bones"]
    indices = sorted({0, predicted.shape[1]//2, predicted.shape[1]-1})
    result = {"mpjpe_mm": float(distances[valid].double().mean()), "valid_joints": int(valid.sum()),
              "frame_mpjpe_mm": {str(i): float(distances[:, i][valid[:, i]].mean()) if valid[:, i].any() else None for i in indices},
              **{f"{label}_spread_mm": shape[f"{label}_spread"]/frames if frames else None for label in ("predicted", "target")},
              **{f"{label}_bone_mm": shape[f"{label}_bone"]/bones if bones else None for label in ("predicted", "target")}}
    endpoints = valid[:, 0] & valid[:, -1]
    for label, xyz in (("predicted", predicted), ("target", clip.target_mm)):
        displacement = (xyz[:, -1]-xyz[:, 0]).norm(dim=-1)
        result[f"{label}_first_last_displacement_mm"] = float(displacement[endpoints].mean()) if endpoints.any() else None
    return result


def _constant_pose(clip):
    valid = clip.target_valid & clip.time_valid[..., None]
    total = torch.where(valid[..., None], clip.target_mm, 0.).sum(1, keepdim=True)
    count = valid.sum(1, keepdim=True)[..., None].clamp_min(1)
    return (total/count).expand_as(clip.target_mm)


class OracleExperiment:
    def __init__(self, config, clip, input_receipt, device, output, budget, input_tensor_path=None):
        self.cfg, self.root, self.budget, self.device = config, Path(output), budget, torch.device(device)
        self.spec, self.input = input_receipt["spec"], input_receipt
        self.input_tensor_path = input_tensor_path
        if clip.source != ORACLE["source"] or clip.split != "train":
            raise ValueError("v14_oracle_train_clip_required")
        self.clip = replace(clip, **{name: getattr(clip, name).to(self.device) for name in
            ("features", "camera", "target_mm", "target_valid", "time_valid", "seconds")})
        names = self.spec["joint_names"]
        self.body_names = self.spec["body_joint_names"]
        self.hand_names = [name for side in ("left", "right") for name in self.spec["hands"][side][1:]]
        self.body_clip = self.clip.select_joints([names.index(name) for name in self.body_names])
        self.hand_clip = hand_clip(self.clip, self.spec)
        self.upper = gt_upper_train(self.clip, self.spec)
        self.edges = {role: [(group.index(a), group.index(b)) for a, b in self.spec["edges"] if a in group and b in group]
                      for role, group in (("body", self.body_names), ("hand", self.hand_names))}
        self.stages = {}
        for offset, (role, group) in enumerate((("body", self.body_names), ("hand", self.hand_names))):
            seed_run(config["seed"]+offset)
            self.stages[role] = OracleStage(config["teacher"], group, self.edges[role],
                                           len(self.spec["upper_joint_names"]) if role == "hand" else 0, self.device)
        environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(self.device),
                       "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                       "vram_bytes": torch.cuda.get_device_properties(self.device).total_memory if self.device.type == "cuda" else None,
                       "precision": "fp32", "cuda": torch.version.cuda, "threads": torch.get_num_threads()}
        self.binding = {"format": FORMAT, "run_id": RUN_ID, "config": config, "input_sha256": clip.input_sha256,
                        "data_kind": input_receipt["data_kind"],
                        "source_code_sha256": source_identity(), "environment": environment, "oracle": ORACLE}
        self.step, self.last, self.durable_step = 0, None, None
        self.evaluations, self.history = {}, []
        self.snapshots = {}

    def report(self, status, failure=None):
        result = {**self.binding, "status": status, "observed_paired_updates": self.step,
                  "optimizer_updates": 2*self.step, "durable_paired_updates": self.durable_step,
                  "last_checkpoint": self.last, "snapshots": self.snapshots, "history": self.history,
                  "evaluations": self.evaluations, "failure": failure,
                  "elapsed_this_invocation_seconds": time.monotonic()-self.budget.start,
                  "metric_scope": "same train clip, not validation or generalization"}
        if self.device.type == "cuda":
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
        raw = canonical(result)+b"\n"
        self.budget.storage(self.root, len(raw))
        _atomic(self.root / "run.json", raw)
        return result

    def save(self):
        self.last = _publish(self.root / "last.pt", {"binding": self.binding, "step": self.step,
            "input_receipt": self.input,
            "fits": {role: _fit_state(stage) for role, stage in self.stages.items()},
            "evaluations": self.evaluations, "history": self.history, "snapshots": self.snapshots}, self.budget)
        self.durable_step = self.step

    def resume(self, expected_last_sha):
        state = _read_checkpoint(self.root / "last.pt", expected_last_sha, self.cfg["limits"]["checkpoint_bytes"])
        if (state["binding"] != self.binding or type(state["step"]) is not int
                or not 0 <= state["step"] <= self.cfg["updates_per_model"]):
            raise ValueError("v14_resume_binding")
        for role, stage in self.stages.items():
            _restore_state(stage, state["fits"][role], state["step"])
        self.step = self.durable_step = state["step"]
        self.evaluations, self.history, self.snapshots = state["evaluations"], state["history"], state["snapshots"]
        for artifact_pin in self.snapshots.values():
            _check_part(self.root / artifact_pin["name"], artifact_pin)
        for evaluation in self.evaluations.values():
            for artifact_pin in evaluation["artifacts"]:
                _check_part(self.root / artifact_pin["name"], artifact_pin)
        self.last = {"name": "last.pt", "bytes": (self.root / "last.pt").stat().st_size, "sha256": expected_last_sha}

    @torch.no_grad()
    def evaluate(self):
        self.budget.check()
        predictions, metrics, baselines = {}, {}, {}
        with preserve_rng():
            for offset, (role, clip) in enumerate((("body", self.body_clip), ("hand", self.hand_clip))):
                self.budget.check()
                stage = self.stages[role]
                stage.model.eval()
                model = stage.condition(self.upper if role == "hand" else None).eval()
                noise = torch.stack([keyed_frame_noise(sequence, frame, clip.target_mm.shape[-2], self.cfg["seed"]+100+offset)
                                     for sequence, frame in self.input["keys"]])[None].to(self.device)
                prediction = stage.diffusion.sample(model, clip.features, clip.camera, clip.time_valid, noise, 10)*100.
                k = torch.ones((1,), dtype=torch.long, device=self.device)
                mask = clip.target_valid & clip.time_valid[..., None]
                clean = torch.where(mask[..., None], clip.target_mm/100., 0.)
                noisy_gt = stage.diffusion.q_sample(clean, noise, k)
                direct, _ = model(noisy_gt, clip.features, clip.camera, k, clip.time_valid)
                predictions[role] = prediction
                metrics[role] = {"ddim10_pure_noise": _metrics(prediction, clip, self.edges[role]),
                                 "k1_target_noised_oracle_reconstruction": _metrics(direct*100., clip, self.edges[role])}
                baselines[role] = _metrics(_constant_pose(clip), clip, self.edges[role])
            gt_body = self.body_clip.target_mm
            hand_global_oracle = attach_predicted_hands(gt_body, predictions["hand"], self.spec)
            cascade_hand = attach_predicted_hands(predictions["body"], predictions["hand"], self.spec)
            fingers = [self.spec["joint_names"].index(name) for name in self.hand_names]
            global_clip = self.clip.select_joints(fingers)
            oracle_global_clip = replace(global_clip, target_valid=self.hand_clip.target_valid)
            metrics["hand_global_gt_wrist_oracle"] = _metrics(hand_global_oracle, oracle_global_clip, self.edges["hand"])
            metrics["hand_global_predicted_wrist_gt_upper_oracle"] = _metrics(cascade_hand, global_clip, self.edges["hand"])
            points = dict(zip(self.body_names+self.hand_names, torch.cat((predictions["body"], cascade_hand), dim=-2).unbind(-2)))
            combined = torch.stack([points[name] for name in self.spec["joint_names"]], dim=-2)
            self.visualize(combined.cpu())
            payload = {"oracle": ORACLE, "step": self.step, "input_sha256": self.clip.input_sha256,
                       "body_mm": predictions["body"].cpu().tolist(),
                       "hand_local_mm": predictions["hand"].cpu().tolist(), "combined_predicted_wrist_mm": combined.cpu().tolist(),
                       "target_mm": self.clip.target_mm.cpu().tolist(), "target_valid": self.clip.target_valid.cpu().tolist(),
                       "hand_local_valid": self.hand_clip.target_valid.cpu().tolist()}
            prediction_name = f"predictions-{self.step:05d}.json"
            _atomic(self.root / prediction_name, canonical(payload)+b"\n")
        criteria = self.cfg["reference_criteria"]
        passes = {}
        for role in ("body", "hand"):
            m = metrics[role]["ddim10_pure_noise"]
            limit = criteria["body_mpjpe_mm" if role == "body" else "hand_local_mpjpe_mm"]
            ratios = {key: m[f"predicted_{key}_mm"]/m[f"target_{key}_mm"]
                      if m[f"target_{key}_mm"] else None for key in ("bone", "spread")}
            passes[role] = {"shape_ratios": ratios, "spatial_reference_pass": m["mpjpe_mm"] <= limit
                and all(value is not None and criteria["shape_ratio_min"] <= value <= criteria["shape_ratio_max"] for value in ratios.values()),
                "better_than_constant_clip_mean": m["mpjpe_mm"] < baselines[role]["mpjpe_mm"]}
            passes[role]["first_middle_last_below_error_limit"] = all(
                value is not None and value <= limit for value in m["frame_mpjpe_mm"].values())
        name = f"models-{self.step:05d}.pt"
        self.snapshots[str(self.step)] = _publish(self.root / name, {"binding": self.binding, "step": self.step,
            "models": {role: stage.model.state_dict() for role, stage in self.stages.items()}}, self.budget)
        self.evaluations[str(self.step)] = {"metrics": metrics, "constant_pose_gt_clip_mean_baseline": baselines,
            "reference_criteria": passes, "prediction_file": prediction_name,
            "artifacts": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                          for path in (self.root / prediction_name, self.root / f"comparison-{self.step:05d}.png")],
            "interpretation": "Provisional memorization criteria only. Budget-limited failure does not establish an implementation bug."}

    def visualize(self, predicted):
        target = self.clip.target_mm.cpu()
        valid = self.clip.target_valid.cpu()
        frames = sorted({0, predicted.shape[1]//2, predicted.shape[1]-1})
        size = 320
        image = Image.new("RGB", (size*2+36, 90+len(frames)*(size+25)), DARK)
        draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=16)
        draw.text((12, 8), f"v1.4 GT2D + GT upper oracle / update {self.step}", fill="white", font=font)
        draw.text((12, 31), "Same train clip / predicted-wrist attachment / shared device-mm scale", fill="white", font=font)
        for row, frame in enumerate(frames):
            gt = [p if ok else None for p, ok in zip(target[0, frame].tolist(), valid[0, frame].tolist())]
            groups = _shared_3d_groups((predicted[0, frame].tolist(), gt), size)
            for col, (label, group) in enumerate(zip(("oracle model", "GT annotation"), groups)):
                x, y = 12+col*(size+12), 64+row*(size+25)
                draw.text((x, y), f"frame {self.input['keys'][frame][1]} / {label}", fill="white", font=font)
                tile = Image.new("RGB", (size, size), "#202a38")
                _draw_joints(tile, group, self.spec)
                image.paste(tile, (x, y+22))
        image.save(self.root / f"comparison-{self.step:05d}.png")

    def run(self, resume=None, stop_after=None):
        with _locked(self.root):
            if resume:
                self.resume(resume)
            elif any(p.name != ".lock" for p in self.root.iterdir()):
                raise ValueError("v14_fresh_run_or_explicit_resume")
            target = min(self.cfg["updates_per_model"], self.step+stop_after) if stop_after else self.cfg["updates_per_model"]
            try:
                if self.last is None:
                    if self.input_tensor_path is not None:
                        raw = _regular_read(self.input_tensor_path, 4*1024**2)
                        if sha(raw) != self.input["tensor"]["sha256"]:
                            raise ValueError("v14_run_input_tensor_pin")
                        _atomic(self.root / "oracle_clip.pt", raw)
                        spec_raw = _regular_read(self.input_tensor_path.parent / "spec.json", 1024**2)
                        if sha(spec_raw) != self.input["spec_sha256"]:
                            raise ValueError("v14_run_spec_pin")
                        _atomic(self.root / "spec.json", spec_raw)
                    _atomic(self.root / "oracle_input.json", canonical(self.input)+b"\n")
                    self.save()
                while True:
                    self.budget.check()
                    if self.step in self.cfg["milestones"] and str(self.step) not in self.evaluations:
                        self.evaluate()
                        self.save()
                    if self.step >= target:
                        self.save()
                        return self.report("completed" if self.step == self.cfg["updates_per_model"] else "paused")
                    started = time.monotonic()
                    log = (self.step+1) % self.cfg["log_every"] == 0 or self.step == 0
                    metrics = {"body": self.stages["body"].update(self.body_clip, None, log),
                               "hand": self.stages["hand"].update(self.hand_clip, self.upper, log)}
                    self.step += 1
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    if log:
                        self.history.append({"paired_update": self.step, "metrics": metrics,
                                             "paired_step_wall_seconds": time.monotonic()-started})
                    if self.step % self.cfg["checkpoint_every"] == 0:
                        self.save()
                        self.report("running")
            except BaseException as error:
                failure = {"type": type(error).__name__, "reason": str(error)[:512]}
                if isinstance(error, TimeoutError) and str(error) == "rewind_v1_run_time_limit":
                    self.save()
                    return self.report("paused", failure)
                with suppress(Exception):
                    self.report("stopped" if isinstance(error, KeyboardInterrupt) else "failed", failure)
                raise

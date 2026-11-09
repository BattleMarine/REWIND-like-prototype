"""Bounded staged training with a single atomic cursor and optimizer checkpoint."""

from contextlib import suppress
import copy
import io
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
import zipfile

import torch

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _hash, _regular_read
from rewind_m1.training import restore_rng, rng_state, seed_run
from rewind_m2.training import validate_config as validate_teacher
from rewind_prepare.converter import sha
from rewind_prepare.temporal_cache import _atomic, _hash_file, _locked
from .frontend import frontend_fingerprint
from .run_stages import CascadeStages, FrameStage, accumulate, evaluate_frontend, frame_batch, preserve_rng
from .training import MAX_CHECKPOINT_BYTES, _source


STAGES = ("frontend", "body", "hand")
FORMAT = "rewind_v1_pipeline_v1"
REPORT = "run.json"
LAST = "last.pt"
MIB = 1024**2


def validate_config(config):
    if set(config) != {"seed", "epochs", "frontend", "teacher", "limits"}:
        raise ValueError("rewind_v1_run_config_fields")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32-1000:
        raise ValueError("rewind_v1_run_seed")
    epochs, frontend, limits = (config[key] for key in ("epochs", "frontend", "limits"))
    if set(epochs) != set(STAGES) or any(type(n) is not int or not 1 <= n <= 100 for n in epochs.values()):
        raise ValueError("rewind_v1_run_epoch_limits")
    if (set(frontend) != {"size", "batch_size", "learning_rate", "scheduler_step", "scheduler_gamma"}
            or not isinstance(frontend["size"], list) or len(frontend["size"]) != 2
            or any(type(n) is not int or n % 16 or not 32 <= n <= 256 for n in frontend["size"])
            or type(frontend["batch_size"]) is not int or not 1 <= frontend["batch_size"] <= 8
            or type(frontend["scheduler_step"]) is not int or frontend["scheduler_step"] < 1
            or not 0 < frontend["learning_rate"] <= .01 or not 0 < frontend["scheduler_gamma"] <= 1):
        raise ValueError("rewind_v1_run_frontend_config")
    teacher = config["teacher"]
    velocity = teacher.get("velocity_loss_weight")
    if type(velocity) not in (float, int) or not 0 <= velocity <= 100000:
        raise ValueError("rewind_v1_run_velocity_weight")
    validate_teacher({**{k: v for k, v in teacher.items() if k != "velocity_loss_weight"},
                      "window_size": 50, "schedule": "cosine_beta_endpoints_v1", "ddim_eta": 0.,
                      "batch_size": 1, "seed": config["seed"], "amp": False})
    if teacher["hidden"] > 512 or teacher["graph_layers"] > 3 or teacher["temporal_layers"] > 3 or teacher["sampling_steps"] > 10:
        raise ValueError("rewind_v1_run_teacher_resource_shape")
    if (set(limits) != {"max_seconds", "max_updates", "output_bytes", "cache_restore_bytes",
                        "feature_bytes", "checkpoint_bytes", "checkpoint_every"}
            or any(type(n) is not int or n <= 0 for n in limits.values())
            or limits["max_seconds"] > 86400 or limits["max_updates"] > 1_000_000
            or limits["checkpoint_bytes"] > MAX_CHECKPOINT_BYTES or limits["feature_bytes"] > 1024*MIB):
        raise ValueError("rewind_v1_run_limits")
    return config


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.start = time.monotonic()

    def remaining(self):
        remaining = self.limits["max_seconds"] - (time.monotonic() - self.start)
        if remaining <= 0:
            raise TimeoutError("rewind_v1_run_time_limit")
        return remaining

    def check(self):
        self.remaining()

    def storage(self, root, reserve):
        used = sum(item.stat().st_size for item in root.iterdir() if item.is_file())
        if used + reserve > self.limits["output_bytes"] or shutil.disk_usage(root).free < reserve:
            raise TimeoutError("rewind_v1_run_output_limit")


class _CappedWriter:
    def __init__(self, handle, limit):
        self.handle, self.remaining = handle, limit

    def write(self, data):
        if len(data) > self.remaining:
            raise ValueError("rewind_v1_run_checkpoint_size")
        written = self.handle.write(data)
        self.remaining -= written
        return written

    def flush(self):
        self.handle.flush()


def _publish(path, value, budget):
    cap = budget.limits["checkpoint_bytes"]
    budget.storage(path.parent, cap)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".save-", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(value, _CappedWriter(handle, cap))
            handle.flush()
            os.fsync(handle.fileno())
        size = temporary.stat().st_size
        pin = _hash_file(temporary, size)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"name": path.name, "sha256": pin, "bytes": size}


def _read_checkpoint(path, expected, maximum):
    raw = _regular_read(path, maximum)
    if not _hash(expected) or sha(raw) != expected:
        raise ValueError("rewind_v1_run_checkpoint_pin")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def _fit_state(stage):
    return {"model": stage.model.state_dict(), "optimizer": stage.optimizer.state_dict(),
            "scheduler": stage.scheduler.state_dict(), "scaler": stage.scaler.state_dict(), "rng": rng_state()}


def _restore_state(stage, state, step):
    if state["scheduler"]["last_epoch"] != step:
        raise ValueError("rewind_v1_run_optimizer_cursor")
    stage.model.load_state_dict(state["model"], strict=True)
    stage.optimizer.load_state_dict(state["optimizer"])
    stage.scheduler.load_state_dict(state["scheduler"])
    stage.scaler.load_state_dict(state["scaler"])
    if hasattr(stage, "step"):
        stage.step = step
    restore_rng(state["rng"])


class Pipeline:
    def __init__(self, config, config_sha256, data, device, output, budget):
        self.cfg, self.data, self.device, self.root, self.budget = config, data, device, Path(output), budget
        self.frontend_size = config["frontend"]["size"]
        self.batch_sizes = {"frontend": config["frontend"]["batch_size"], "body": 1, "hand": 1}
        self.keys = {role: data.frames["train"] if role == "frontend" else data.clips["train"] for role in STAGES}
        self.updates = {role: math.ceil(len(self.keys[role])/self.batch_sizes[role]) for role in STAGES}
        self.total = sum(self.updates[role]*config["epochs"][role] for role in STAGES)
        if self.total > config["limits"]["max_updates"]:
            raise ValueError("rewind_v1_run_update_budget")
        self.binding = {"format": FORMAT, "config": config, "config_sha256": config_sha256, "input": data.identity,
                        "source_code_sha256": _source(), "coverage": data.coverage(),
                        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                        "pillow": __import__("PIL").__version__, "device": str(device),
                                        "device_name": torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                                        "device_memory_bytes": torch.cuda.get_device_properties(device).total_memory
                                        if device.type=="cuda" else None,
                                        "cuda_runtime": torch.version.cuda, "precision": "fp32", "threads": torch.get_num_threads()}}
        self.progress = {"stage": "frontend", "epoch": 0, "batch": 0, "updates": 0, "history": [],
                         "best": {}, "partial_loss_sums": {}}
        self.last, self.cascade, self.frozen_body = None, None, None
        self.durable_progress = None
        self.timing = {}

    def _prefix(self, role):
        return sum(self.updates[r]*self.cfg["epochs"][r] for r in STAGES[:STAGES.index(role)])

    def _step(self):
        cursor = self.progress
        return cursor["epoch"] * self.updates[cursor["stage"]] + cursor["batch"]

    def _report(self, status, failure=None):
        report = {**self.binding, "status": status, "cursor": self.progress, "checkpoint_cursor": self.durable_progress,
                  "last_checkpoint": self.last,
                  "planned_updates": self.total, "timing_this_invocation": self.timing,
                  "elapsed_this_invocation_seconds": time.monotonic() - self.budget.start,
                  "cache_restore": getattr(self.data, "restore_metrics", None), "failure": failure}
        payload = canonical(report) + b"\n"
        self.budget.storage(self.root, len(payload))
        _atomic(self.root / REPORT, payload)
        return report

    def _save_last(self, stage):
        start = time.monotonic()
        self.last = _publish(self.root / LAST, {"binding": self.binding, "progress": self.progress,
                                               "fit": _fit_state(stage)}, self.budget)
        self.durable_progress = copy.deepcopy(self.progress)
        accumulate(self.timing, {"checkpoint_wall_seconds": time.monotonic()-start})
        self._report("running")

    def _save_best(self, role, stage, metrics):
        epoch = self.progress["epoch"]
        old = self.progress["best"].get(role)
        if old is None or metrics["score"] < old["score"]:
            start = time.monotonic()
            name = f"best-{role}-e{epoch+1:03d}.pt"
            pin = _publish(self.root / name, {"binding": self.binding, "stage": role, "epoch": epoch+1,
                                            "score": metrics["score"], "model": stage.model.state_dict(),
                                            "dependencies": self._dependencies(role)}, self.budget)
            accumulate(self.timing, {"checkpoint_wall_seconds": time.monotonic()-start})
            self.progress["best"][role] = {**pin, "score": metrics["score"], "epoch": epoch+1}

    def _dependencies(self, role):
        return {stage: self.progress["best"][stage]["sha256"] for stage in STAGES[:STAGES.index(role)]}

    def _load_best(self, role, stage):
        pin = self.progress["best"][role]
        name = f"best-{role}-e{pin['epoch']:03d}.pt"
        if pin["name"] != name:
            raise ValueError("rewind_v1_run_best_name")
        state = _read_checkpoint(self.root / name, pin["sha256"], self.cfg["limits"]["checkpoint_bytes"])
        if (state["binding"], state["stage"], state["epoch"], state["score"], state["dependencies"]) != (
            self.binding, role, pin["epoch"], pin["score"], self._dependencies(role)
        ):
            raise ValueError("rewind_v1_run_best_identity")
        stage.model.load_state_dict(state["model"], strict=True)
        stage.model.eval().requires_grad_(False)
        if hasattr(stage, "checkpoint_sha256"):
            stage.checkpoint_sha256 = pin["sha256"]

    def _make_frontend(self):
        return FrameStage(self.cfg["frontend"], len(self.data.spec["joint_names"]), self.device)

    def _make_stage(self, role):
        if role == "frontend":
            return self._make_frontend()
        if self.cascade is None:
            self._features()
        if role == "hand" and self.frozen_body is None:
            with preserve_rng():
                self.frozen_body = self.cascade.make("body")
                self._load_best("body", self.frozen_body)
        pin = self.progress["best"].get("body", {}).get("sha256") if role == "hand" else None
        return self.cascade.make(role, pin)

    def _features(self):
        with preserve_rng():
            frontend = self._make_frontend()
            self._load_best("frontend", frontend)
            pin = frontend_fingerprint(frontend.model)
            clips = {}
            bytes_used = 0
            joints = len(self.data.spec["joint_names"])
            estimate = self.data.width * ((14+3)*joints*4 + joints + 2*9*4 + 8 + 1)
            for split, keys in self.data.clips.items():
                clips[split] = {}
                for key in keys:
                    self.budget.check()
                    if bytes_used + estimate > self.cfg["limits"]["feature_bytes"]:
                        raise ValueError("rewind_v1_run_feature_limit")
                    clip = self.data.clip(key, split, frontend.model, self.frontend_size, pin)
                    tensors = (clip.features, clip.camera, clip.target_mm, clip.target_valid, clip.time_valid, clip.seconds)
                    bytes_used += sum(t.numel()*t.element_size() for t in tensors)
                    if bytes_used > self.cfg["limits"]["feature_bytes"]:
                        raise ValueError("rewind_v1_run_feature_limit")
                    clips[split][key] = clip
                    accumulate(self.timing, clip.timings)
            identities = {clip.input_sha256 for group in clips.values() for clip in group.values()}
            if len(identities) != 1:
                raise ValueError("rewind_v1_run_feature_identity")
            self.timing["retained_feature_tensor_bytes"] = bytes_used
            self.cascade = CascadeStages(self.data, clips, self.cfg, self.device, bytes_used)

    def _order(self, role, epoch):
        order = list(self.keys[role])
        random.Random(f"{self.cfg['seed']}:{role}:{epoch}:v1").shuffle(order)
        return order

    def _epoch(self, role, stage, stop_at):
        p = self.progress
        size = self.batch_sizes[role]
        order = self._order(role, p["epoch"])
        while p["batch"] < self.updates[role]:
            if p["updates"] >= stop_at:
                return False
            self.budget.check()
            batch = p["batch"]
            start = time.monotonic()
            if role == "frontend":
                tensors = frame_batch(self.data, order[batch*size:(batch+1)*size], "train", self.frontend_size,
                                      self.budget, self.timing)
                metrics = stage.update(tensors, self.frontend_size)
                accumulate(self.timing, stage.runtime, "frontend_train_")
            else:
                metrics = self.cascade.train(role, stage, order[batch])
                accumulate(self.timing, stage.runtime, "train_", role + "_")
            accumulate(self.timing, {role + "_training_step_wall_seconds": time.monotonic()-start})
            accumulate(p["partial_loss_sums"], metrics)
            p["batch"] += 1
            p["updates"] += 1
            if p["updates"] % self.cfg["limits"]["checkpoint_every"] == 0 or p["updates"] == stop_at or p["batch"] == self.updates[role]:
                self._save_last(stage)
        start = time.monotonic()
        with preserve_rng():
            if role == "frontend":
                metrics = evaluate_frontend(stage, self.data, self.cfg["frontend"], self.budget, self.timing)
            else:
                metrics = self.cascade.evaluate(role, stage, self.budget, self.timing, self.frozen_body)
        accumulate(self.timing, {role + "_validation_wall_seconds": time.monotonic()-start})
        self._save_best(role, stage, metrics)
        p["history"].append({"stage": role, "epoch": p["epoch"]+1, "updates": p["updates"],
                             "train_sums": p["partial_loss_sums"], "validation": metrics})
        p.update(epoch=p["epoch"]+1, batch=0, partial_loss_sums={})
        self._save_last(stage)
        return True

    def _resume(self, expected):
        path = self.root / LAST
        state = _read_checkpoint(path, expected, self.cfg["limits"]["checkpoint_bytes"])
        if not isinstance(state, dict) or state.get("binding") != self.binding:
            raise ValueError("rewind_v1_run_resume_binding")
        self.progress = p = state["progress"]
        role = p["stage"]
        if (role not in STAGES or type(p["epoch"]) is not int or not 0 <= p["epoch"] <= self.cfg["epochs"][role]
                or type(p["batch"]) is not int or not 0 <= p["batch"] <= self.updates[role]
                or p["epoch"] == self.cfg["epochs"][role] and p["batch"]
                or p["updates"] != self._prefix(role) + self._step()):
            raise ValueError("rewind_v1_run_cursor")
        self.last = {"name": LAST, "sha256": expected, "bytes": path.stat().st_size}
        self.durable_progress = copy.deepcopy(self.progress)
        return state["fit"]

    def run(self, resume_sha256=None, stop_after_updates=None):
        if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 1):
            raise ValueError("rewind_v1_run_stop_updates")
        with _locked(self.root):
            fit = self._resume(resume_sha256) if resume_sha256 else None
            if fit is None and any(entry.name != ".lock" for entry in self.root.iterdir()):
                raise ValueError("rewind_v1_run_output_requires_resume")
            stop_at = self.total if stop_after_updates is None else min(self.total, self.progress["updates"]+stop_after_updates)
            try:
                for index in range(STAGES.index(self.progress["stage"]), len(STAGES)):
                    self.budget.check()
                    role = STAGES[index]
                    seed_run(self.cfg["seed"] + index)
                    stage = self._make_stage(role)
                    if fit is not None:
                        _restore_state(stage, fit, self._step())
                        fit = None
                    elif index or self.last is None:
                        self.progress.update(stage=role, epoch=0, batch=0, partial_loss_sums={})
                        self._save_last(stage)
                    while self.progress["epoch"] < self.cfg["epochs"][role]:
                        if not self._epoch(role, stage, stop_at):
                            return self._report("paused")
                    del stage
                    if self.progress["updates"] >= stop_at and self.progress["updates"] < self.total:
                        return self._report("paused")
                return self._report("completed")
            except BaseException as exc:
                with suppress(BaseException):
                    self._report("stopped" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed",
                                 {"type": type(exc).__name__, "reason": str(exc)[:512]})
                raise

"""Bounded CLI for a pinned direct-extracted RGB2D diagnostic run."""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import io
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_prepare.extracted import ExtractedStereoT1, _local
from rewind_prepare.converter import sha
from .extracted_rgb_training import (
    DirectRGBTrainer, PILOT_TRAIN_COUNT, RUN_SCHEMA, SCOPE, comparison_config_sha256,
)
from .frontend import StereoHeatmapNet, weights_hash
from .local_rgb import _regular_read
from .rgb_frontend_artifacts import preprocessing_version, resize_images, scale_xy
from .shards import _hash, _require


MIB = 1024**2
REPORT_MAX = 4 * MIB
CHECKPOINT_MAX = 4 * MIB
WEIGHTS_MAX = 2 * MIB
WEIGHTS_FORMAT = "direct_extracted_rgb2d_weights_v1"


@dataclass(frozen=True)
class RunLimits:
    max_seconds: float
    max_attempts: int
    max_read_bytes: int
    max_rss_bytes: int
    max_disk_bytes: int
    max_cuda_allocated_bytes: int
    max_cuda_reserved_bytes: int

    def __post_init__(self):
        _require(type(self.max_seconds) in (int, float) and 0 < self.max_seconds <= 86400
                 and all(type(value) is int and 0 < value < 2**63
                         for key, value in asdict(self).items() if key != "max_seconds")
                 and self.max_disk_bytes >= 3 * REPORT_MAX, "direct_rgb_run_limits")


class _Budget:
    def __init__(self, limits, output, device):
        import psutil

        self.limits, self.output, self.device = limits, output, torch.device(device)
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.read_bytes, self.peak_rss, self.peak_disk = 0, 0, 0
        self.peak_allocated, self.peak_reserved = 0, 0
        if self.device.type == "cuda":
            _require(torch.cuda.is_available(), "direct_rgb_cuda_unavailable")
            torch.cuda.reset_peak_memory_stats(self.device)

    def disk(self):
        return sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())

    def check(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.peak_disk = max(self.peak_disk, self.disk())
        if self.device.type == "cuda":
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.device))
        observations = ((time.monotonic() - self.started, self.limits.max_seconds, "elapsed"),
                        (self.peak_rss, self.limits.max_rss_bytes, "rss"),
                        (self.peak_disk, self.limits.max_disk_bytes, "disk"),
                        (self.peak_allocated, self.limits.max_cuda_allocated_bytes, "cuda_allocated"),
                        (self.peak_reserved, self.limits.max_cuda_reserved_bytes, "cuda_reserved"))
        for actual, limit, name in observations:
            if actual > limit:
                raise TimeoutError("direct_rgb_" + name + "_limit")

    def reserve_read(self, size):
        self.check()
        if self.read_bytes + size > self.limits.max_read_bytes:
            raise TimeoutError("direct_rgb_read_byte_limit")
        self.read_bytes += size

    def read(self, path, maximum):
        path = _local(path)
        size = path.stat().st_size
        _require(1 <= size <= maximum, "direct_rgb_input_file_size")
        self.reserve_read(size + 1)
        return _regular_read(path, maximum, size)

    def publish(self, name, payload, maximum):
        _require(len(payload) <= maximum, "direct_rgb_artifact_byte_limit")
        peak = self.disk() + len(payload)
        reserve = REPORT_MAX if name != "report.json" else 0
        if (peak + reserve > self.limits.max_disk_bytes
                or shutil.disk_usage(self.output).free < len(payload) + reserve):
            raise TimeoutError("direct_rgb_artifact_disk_limit")
        with tempfile.NamedTemporaryFile(dir=self.output, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                handle.close()
                temporary.unlink()
                raise
        try:
            self.peak_disk = max(self.peak_disk, peak)
            temporary.replace(self.output / name)
        finally:
            temporary.unlink(missing_ok=True)
        return sha(payload)

    def summary(self):
        return {"elapsed_seconds": time.monotonic() - self.started,
                "input_bytes_reserved": self.read_bytes, "peak_observed_rss_bytes": self.peak_rss,
                "peak_artifact_bytes": self.peak_disk, "cuda_peak_allocated_bytes": self.peak_allocated,
                "cuda_peak_reserved_bytes": self.peak_reserved, "limits": asdict(self.limits),
                "measurement_scope": "cooperative host samples and PyTorch CUDA allocator peaks"}


def _serialized_state(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _pinned(guard, path, expected_sha256, maximum):
    _require(_hash(expected_sha256), "direct_rgb_required_input_pin")
    raw = guard.read(path, maximum)
    _require(sha(raw) == expected_sha256, "direct_rgb_input_pin_mismatch")
    return raw


def _weights(trainer, checkpoint_sha256):
    config = trainer.config
    metadata = {"training_scope": SCOPE, "diagnostic_only": True, "train_ready": False,
                "manifest_sha256": trainer.binding["manifest_sha256"], "checkpoint_sha256": checkpoint_sha256,
                "source": trainer.source, "step": trainer.step, "loss_kind": config["loss_kind"],
                "training_image_size": config["training_image_size"],
                "preprocessing_version": preprocessing_version(config["training_image_size"]),
                "frontend_version": config["frontend_version"], "joint_names": trainer.contract.joint_names,
                "config_sha256": digest(config)}
    return {"format_version": WEIGHTS_FORMAT, "metadata": metadata, "metadata_sha256": digest(metadata),
            "model_sha256": weights_hash(trainer.model),
            "model": {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}}


class DirectRGBInference:
    """Read a pinned diagnostic weights file and predict one native stereo pair."""

    def __init__(self, path, *, expected_sha256):
        raw = _regular_read(_local(path), WEIGHTS_MAX)
        _require(_hash(expected_sha256) and sha(raw) == expected_sha256, "direct_rgb_weights_file_pin")
        artifact = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        meta = artifact["metadata"]
        _require(artifact["format_version"] == WEIGHTS_FORMAT and artifact["metadata_sha256"] == digest(meta)
                 and meta["training_scope"] == SCOPE and meta["diagnostic_only"] is True
                 and meta["train_ready"] is False and len(set(meta["joint_names"])) == 62
                 and meta["preprocessing_version"] == preprocessing_version(meta["training_image_size"]),
                 "direct_rgb_weights_identity")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.model = StereoHeatmapNet(62).float()
        _require(all(v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in artifact["model"].values()),
                 "direct_rgb_weights_tensor")
        self.model.load_state_dict(artifact["model"], strict=True)
        _require(weights_hash(self.model) == artifact["model_sha256"], "direct_rgb_weights_state")
        self.model.eval()
        self.metadata = meta

    def predict(self, images):
        _require(images.shape == (1, 2, 3, 1024, 1024) and images.device.type == "cpu", "direct_rgb_native_pair")
        size = self.metadata["training_image_size"]
        with torch.no_grad():
            _, xy, confidence = self.model(resize_images(images, size))
        return {"coordinates": scale_xy(xy, size, [1024, 1024]), "confidence": confidence,
                "diagnostic_only": True, "input_source": "predicted_2d"}


def run(*, manifest, manifest_sha256, rgb_root, pose_root, spec, config, output, limits,
        device="cpu", epochs=None, steps=None, checkpoint_every=25, resume=None, resume_sha256=None,
        pilot_report=None, pilot_report_sha256=None, promotion_review=None, promotion_review_sha256=None):
    _require((epochs is None) != (steps is None), "direct_rgb_one_run_target")
    target = epochs if epochs is not None else steps
    _require(type(target) is int and target > 0 and type(checkpoint_every) is int and checkpoint_every > 0,
             "direct_rgb_run_target")
    _require(isinstance(limits, RunLimits), "direct_rgb_run_limits")
    _require((resume is None) == (resume_sha256 is None), "direct_rgb_resume_pin")
    evidence = (pilot_report, pilot_report_sha256, promotion_review, promotion_review_sha256)
    _require(all(v is None for v in evidence) or all(v is not None for v in evidence), "direct_rgb_promotion_pins")
    output = Path(output).absolute()
    parent = _local(output.parent, directory=True)
    output = parent / output.name
    _require(not any(output.is_relative_to(_local(root, directory=True)) for root in (rgb_root, pose_root)),
             "direct_rgb_output_outside_inputs")
    output.mkdir(exist_ok=False)
    guard = _Budget(limits, output, device)
    report = {"schema": RUN_SCHEMA, "status": "started", "training_scope": SCOPE,
              "diagnostic_only": True, "train_ready": False, "independent_validation_verified": False,
              "operational_pilot_passed": False, "promotion_decision": "external_review_required",
              "target": {"epochs": epochs, "steps": steps}}
    events = deque(maxlen=32)
    timings = {}

    def report_now():
        report.update(resources=guard.summary(), phase_seconds=timings, recent_updates=list(events))
        guard.publish("report.json", canonical(report) + b"\n", REPORT_MAX)

    def timed(name, call):
        started = time.monotonic()
        try:
            return call()
        finally:
            timings[name] = timings.get(name, 0.0) + time.monotonic() - started

    try:
        config_doc = read_json(guard.read(config, 64 * 1024))
        reader = ExtractedStereoT1(manifest, rgb_root, pose_root, spec,
                                   expected_sha256=manifest_sha256, split="train", read_metadata=guard.read)
        promotion = None
        if pilot_report is not None:
            promotion = (read_json(_pinned(guard, pilot_report, pilot_report_sha256, REPORT_MAX)),
                         pilot_report_sha256,
                         read_json(_pinned(guard, promotion_review, promotion_review_sha256, 64 * 1024)),
                         promotion_review_sha256)
        trainer = DirectRGBTrainer(config_doc, reader, device=device, guard=guard, promotion=promotion)
        report.update(binding=trainer.binding, config=trainer.config, config_sha256=digest(trainer.config),
                      comparison_config_sha256=comparison_config_sha256(trainer.config),
                      promotion_decision="verified_pinned_review" if trainer.binding["phase"] == "full"
                      else "external_review_required",
                      source=trainer.source, environment=trainer.environment,
                      initial_weights_sha256=trainer.initial_weights_sha256)
        if resume is not None:
            state = torch.load(io.BytesIO(_pinned(guard, resume, resume_sha256, CHECKPOINT_MAX)),
                               map_location="cpu", weights_only=True)
            trainer.resume(state)
            initial = state["initial_metrics"]
            report["resumed_checkpoint_sha256"] = resume_sha256
            _require(target >= (trainer.step if steps is not None else trainer.sampler.epoch),
                     "direct_rgb_target_before_resume")
        else:
            initial = timed("initial_evaluation", lambda: {s: trainer.evaluate(s) for s in ("train", "validation")})
        report["initial_metrics"] = initial

        def checkpoint():
            guard.check()
            state = {**trainer.checkpoint(), "initial_metrics": initial}
            identity = guard.publish("last.pt", _serialized_state(state), CHECKPOINT_MAX)
            report.update(checkpoint={"sha256": identity, "step": trainer.step},
                          coverage=trainer.coverage(), diagnostic_progress=trainer.progress(),
                          optimizer_updates=trainer.step, attempts=trainer.attempts,
                          amp_skipped_updates=trainer.skipped_updates)
            report_now()

        checkpoint()
        starting_attempts = trainer.attempts
        while (trainer.step if steps is not None else trainer.sampler.epoch) < target:
            if trainer.attempts - starting_attempts >= limits.max_attempts:
                raise TimeoutError("direct_rgb_attempt_limit")
            guard.check()
            event = timed("training", trainer.train_step)
            guard.check()
            events.append(event)
            if not event["updated"] or trainer.step % checkpoint_every == 0:
                checkpoint()
        checkpoint()
        report["final_metrics"] = timed("final_evaluation", lambda: {s: trainer.evaluate(s) for s in ("train", "validation")})
        guard.check()
        report["final_weights_sha256"] = weights_hash(trainer.model)
        report["weights_file_sha256"] = guard.publish(
            "weights.pt", _serialized_state(_weights(trainer, report["checkpoint"]["sha256"])), WEIGHTS_MAX)
        trained_seconds = timings.get("training", 0.0)
        report["this_run_attempts_per_training_second"] = (
            (trainer.attempts - starting_attempts) / trained_seconds if trained_seconds else None)
        passed = (trainer.binding["phase"] == "pilot" and trainer.binding["train_count"] == PILOT_TRAIN_COUNT
                  and trainer.last_completed_unique_train == PILOT_TRAIN_COUNT
                  and trainer.nonzero_gradient_updates > 0
                  and report["initial_weights_sha256"] != report["final_weights_sha256"])
        if passed:
            report["pilot_train_records"] = [[*key, identity] for key, identity in trainer.train_records.items()]
        report.update(status="completed", operational_pilot_passed=passed)
        report_now()
    except BaseException as exc:
        report.update(status="bounded_stop" if isinstance(exc, TimeoutError) else
                      "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      operational_pilot_passed=False,
                      failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
        try:
            report_now()
        except (OSError, TimeoutError, ValueError):
            pass
        if not isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "rgb-root", "pose-root", "spec", "config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--epochs", type=int)
    target.add_argument("--steps", type=int)
    for key in RunLimits.__dataclass_fields__:
        parser.add_argument("--" + key.replace("_", "-"), required=True,
                            type=float if key == "max_seconds" else int)
    for key in ("resume", "pilot-report", "promotion-review"):
        parser.add_argument("--" + key, type=Path)
        parser.add_argument("--" + key + "-sha256")
    args = vars(parser.parse_args(argv))
    try:
        args["limits"] = RunLimits(**{key: args.pop(key) for key in RunLimits.__dataclass_fields__})
        report = run(**args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(report["status"])
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Full-cache V2Pipeline adapter with common, resumable validation across seeds."""

from collections import OrderedDict
from copy import deepcopy
import math
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_cache import _atomic
from rewind_v1 import v2_contract as contract
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.run_stages import CascadeStages, FrameStage, evaluate_frontend, keyed_frame_noise, preserve_rng
from rewind_v1.runner import Pipeline, REPORT
from rewind_v1.v2_cache import Cache, CHUNK
from rewind_v1.v2_training import Data, ScenarioMap, TrainingBudget, V2Pipeline, runtime_config
from scripts.v2_gpu_training_features import PersistentFeatures
from scripts.v2_gpu_training_storage import EVAL_SEED, publish_json, read_sealed, restore_cache


class FullCache(Cache):
    def __init__(self, root, journal, anchor, entries, spec, cfg, ledger):
        self.root, self.spec, self.ledger = root, spec, ledger
        self.journal, self.doc, self.anchor, self.entries, self.cfg = journal, journal.doc, anchor, entries, cfg
        self.root.mkdir(exist_ok=True)
        self.loaded, self.verified = OrderedDict(), set()
        self.index = {(r["sequence_id"], r["frame_index"]): i for i, r in enumerate(self.doc["records"])}
        self.identity = digest({"journal": journal.journal_id, "final": entries[-1]["content_sha256"]})
        self.metrics = {"seconds": 0, "drive_read_payload_bytes": 0, "reused_shards": 0, "verified_shards": 0}

    def _ensure(self, entries):
        if entries:
            started = time.monotonic()
            result = restore_cache(self.root, self.journal, self.anchor, entries, self.cfg, self.ledger)
            for k in ("drive_read_payload_bytes", "reused_shards"):
                self.metrics[k] += result[k]
            self.verified.update(e["index"] for e in entries)
            self.metrics.update(seconds=self.metrics["seconds"]+time.monotonic()-started, verified_shards=len(self.verified))

    def prepare_frontend(self):
        self._ensure([e for e in self.entries if e["index"] not in self.verified])

    def frame(self, key, split):
        number = self.index[tuple(key)] // CHUNK
        if number not in self.verified:
            self._ensure([self.entries[number]])
        return super().frame(key, split)


class FullData(Data):
    def __init__(self, doc, spec, cfg, plan, cache, ledger, source):
        self.spec, self.doc, self.cfg, self.plan, self.ledger = spec, doc, cfg, plan, ledger
        self.cache, self.budget = cache, ledger
        self.frames = {s: sorted((r["sequence_id"], r["frame_index"]) for r in doc["records"] if r["split"] == s)
                       for s in ("train", "validation")}
        self.clip_rows = {(r["sequence"], r["start_frame"]): r for r in plan["clips"]}
        self.clips = {s: sorted(k for k, r in self.clip_rows.items() if r["split"] == s) for s in self.frames}
        self.identity = {"data_kind": "dataset_sample", "run_kind": "nominal_25fps_diagnostic",
                         "manifest": doc["content_sha256"], "cache": cache.identity, "camera": cfg["camera"],
                         "source": source, "spec": digest(spec), "padding": "post_zero_invalid",
                         "clock": "nominal25fps diagnostic"}
        self.restore_metrics = cache.metrics


class CommonEvaluation(CascadeStages):
    def _noise(self, key, joints, seed_offset):
        sequence, start = key
        return torch.stack([keyed_frame_noise(sequence, frame, joints, EVAL_SEED + seed_offset)
                            for frame in range(start, start + self.data.width)])[None]


class FirstUpdateStage(FrameStage):
    def update(self, sample, image_size):
        self.measure_gradient |= self.require_proof()
        return super().update(sample, image_size)


def mean(rows, field, weight):
    denom = sum(r[weight] for r in rows)
    return sum(r[field] * r[weight] for r in rows if r[weight]) / denom if denom else None


def merge_metrics(role, rows):
    if role == "frontend":
        return {"score": mean(rows, "score", "inside_joint_views"), "unit": rows[0]["unit"],
                **{k: sum(r[k] for r in rows) for k in ("frames", "inside_joint_views", "valid_joint_views")}}
    metrics = {}
    for name in rows[0]["joint_metrics"]:
        group = [r["joint_metrics"][name] for r in rows]
        metrics[name] = {"mpjpe_mm": mean(group, "mpjpe_mm", "valid_joints"),
                         "valid_joints": sum(r["valid_joints"] for r in group)}
    shapes = [r["collapse_diagnostics"] for r in rows]
    length, spread = "valid_bone_instances", "frames_with_two_valid_joints"
    diagnostics = {"scope": shapes[0]["scope"], length: sum(r[length] for r in shapes), spread: sum(r[spread] for r in shapes)}
    for label in ("predicted", "target"):
        for metric, weight in (("mean_bone_length_mm", length), ("mean_joint_spread_mm", spread)):
            field = f"{label}_{metric}"
            diagnostics[field] = mean(shapes, field, weight)
    return {"score": metrics["body" if role == "body" else "hand_global"]["mpjpe_mm"],
            "unit": "mm", "frames": sum(r["frames"] for r in rows), "clips": sum(r["clips"] for r in rows),
            "joint_metrics": metrics, "collapse_diagnostics": diagnostics, "upper_source": rows[0]["upper_source"],
            "body_checkpoint_sha256": rows[0]["body_checkpoint_sha256"]}


class Training(V2Pipeline):
    def __init__(self, cfg, data, features, device, output, drive, ledger, owner, identity):
        runtime = runtime_config(cfg, data.plan)
        self.v2cfg, self.ledger, self.owner, self.drive = cfg, ledger, owner, drive
        self.feature_root, self.identity = features, identity
        Pipeline.__init__(self, runtime, digest(cfg), data, device, output, TrainingBudget(runtime["limits"], ledger))
        self.epoch_expansion = True
        self.progress["milestones"] = {}
        self.environment = deepcopy(self.binding["environment"])
        # Keep exact software, CUDA runtime and one-device RNG contract across resumed VMs.
        # The physical CUDA device is recorded per receipt and may change across invocations.
        for key in ("device_name", "device_memory_bytes"):
            del self.binding["environment"][key]
        self.binding.update(format="rewind_v2_gpu_training_v1", v2_config=cfg,
                            source_code_sha256=identity["source"], experiment=identity)
        self.checkpoints = None
        self.first_required = identity["seed"] == 42
        self.backup_at = time.monotonic()
        self.invocation_start_updates = 0

    def _make_frontend(self):
        stage = FirstUpdateStage(self.cfg["frontend"], len(self.data.spec["joint_names"]), self.device)
        stage.require_proof = lambda: self.first_required and self.progress["stage"] == "frontend"
        return stage

    def _make_stage(self, role):
        if role == "frontend" and self.progress["epoch"] < self.cfg["epochs"][role]:
            self.data.cache.prepare_frontend()
        return super()._make_stage(role)

    def _features(self):
        with preserve_rng():
            frontend = self._make_frontend()
            self._load_best("frontend", frontend)
            store = PersistentFeatures(self.feature_root, self.data, frontend_fingerprint(frontend.model),
                                       self.drive / "features", self.v2cfg, self.ledger, self.owner)
            clips, stats = store.build(frontend.model, self.frontend_size)
            self.timing.update(stats)
            self.feature_store_identity = stats["feature_identity"]
            self.cascade = CommonEvaluation(self.data, clips, self.cfg, self.device, 0)

    def _save_last(self, stage):
        Pipeline._save_last(self, stage)
        archive = self.v2cfg["resources"]["archive_bytes"]
        if (not self.first_required and time.monotonic()-self.backup_at >= 3600
                and all(self.ledger.current[k+"_bytes_reserved"] + 4*archive <= self.v2cfg["resources"][k+"_bytes"]
                        for k in ("read", "write"))):
            self.checkpoints.save()
            self.backup_at = time.monotonic()

    def _after_update(self, role, stage):
        proof = self.progress["partial_loss_sums"]
        gradient = proof.get("first_update_gradient_l2", 0)
        if self.first_required and role == "frontend" and proof.get("supervised_joint_views", 0) > 0:
            if not 0 < gradient < math.inf or not stage.optimizer.state:
                raise ValueError("v2_gpu_A_first_update_nonzero_gradient_required")
            self._save_last(stage)
            self.checkpoints.save(first=True)
            self.first_required = False
            self.backup_at = time.monotonic()
        if self.progress["updates"] % 100 == 0:
            print(f"Training {role} epoch={self.progress['epoch']+1} updates={self.progress['updates']}/{self.total}", flush=True)

    def _validate_epoch(self, role, stage):
        frame_phase = role == "frontend"
        folder = self.drive / "validation"
        folder.mkdir(exist_ok=True)
        schema = "rewind_v2_gpu_validation_chunk_v1"
        original = self.data.frames["validation"] if frame_phase else self.cascade.clips["validation"]
        task = {"experiment": self.identity, "model_checkpoint": self.last["sha256"], "role": role,
                "epoch": self.progress["epoch"], "eval_seed": EVAL_SEED}
        stage_path = folder / digest(task)
        stage_path.mkdir(exist_ok=True)
        scenario_results = {}
        try:
            for scenario in contract.SCENARIOS:
                keys = [k for k in original if k[0].startswith(scenario+"/")]
                rows, step = [], 512 if frame_phase else 24
                for begin in range(0, len(keys), step):
                    self.ledger.check()
                    chunk = keys[begin:begin+step]
                    marker = stage_path / f"{scenario}-{begin:05d}.json"
                    binding = {"task": task, "keys_sha256": digest(chunk)}
                    if marker.is_file():
                        record = read_sealed(marker, schema)
                        if record["binding"] != binding:
                            raise ValueError("v2_gpu_validation_binding")
                        metrics = record["metrics"]
                    else:
                        if frame_phase:
                            self.data.frames["validation"] = chunk
                            metrics = evaluate_frontend(stage, self.data, self.cfg["frontend"], self.budget, self.timing)
                        else:
                            group = ScenarioMap(original, scenario)
                            group.keys = chunk
                            self.cascade.clips["validation"] = group
                            metrics = self.cascade.evaluate(role, stage, self.budget, self.timing, self.frozen_body,
                                                            diagnostics=True)
                        publish_json(marker, schema, {"binding": binding, "metrics": metrics}, self.owner)
                    rows.append(metrics)
                summary = merge_metrics(role, rows)
                if not frame_phase:
                    summary["real_frames"] = sum(self.data.real_frames(k, "validation") for k in keys)
                    summary["padding_slots"] = summary["frames"] - summary["real_frames"]
                scenario_results[scenario] = summary
        finally:
            if frame_phase:
                self.data.frames["validation"] = original
            else:
                self.cascade.clips["validation"] = original
        metrics = merge_metrics(role, list(scenario_results.values()))
        if not frame_phase:
            metrics["real_frames"] = sum(r["real_frames"] for r in scenario_results.values())
            metrics["padding_slots"] = metrics["frames"] - metrics["real_frames"]
        metrics["per_scenario_reference"] = scenario_results
        return metrics

    def _save_epoch_model(self, role, stage, metrics):
        Pipeline._save_epoch_model(self, role, stage, metrics)

    def _report(self, status, failure=None):
        report = super()._report(status, failure)
        report["actual_environment"] = self.environment
        _atomic(self.root / REPORT, canonical(report))
        return report

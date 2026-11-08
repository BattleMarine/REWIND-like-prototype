"""Direct extracted RGB2D diagnostic training and split evaluation."""

from collections import defaultdict
import math
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from rewind_archive.contracts import digest
from rewind_prepare.extracted import ExtractedStereoT1, _code_identity
from rewind_prepare.converter import sha
from .frontend import heatmap_loss, weights_hash
from .rgb_frontend_artifacts import scale_xy
from .rgb_training import RGBTrainer, validate_config
from .shards import EpochSampler, _hash, _require
from .training import restore_rng, rng_state


FORMAT = "direct_extracted_rgb2d_checkpoint_v1"
SCOPE = "direct_extracted_rgb2d_diagnostic_v1"
RUN_SCHEMA = "direct_extracted_rgb2d_run_v1"
PILOT_TRAIN_COUNT = 1000
CHUNK_SIZE = 128
LOSSES = ("mse", "gaussian_cross_entropy")


def direct_config(config):
    _require(isinstance(config, dict), "direct_rgb_config")
    loss = config.get("loss_kind", "mse")
    _require(loss in LOSSES, "direct_rgb_loss_kind")
    return {**validate_config({k: v for k, v in config.items() if k != "loss_kind"}), "loss_kind": loss}


def comparison_config_sha256(config):
    return digest({k: v for k, v in config.items()
                   if k not in ("max_native_tensor_bytes", "max_heatmap_elements")})


def _training_contract_sha256():
    directory = Path(__file__).parent
    names = ("frontend.py", "rgb_training.py", "rgb_frontend_artifacts.py",
             "shard_training.py", "shards.py", "extracted_rgb_training.py")
    return digest({name: sha((directory / name).read_bytes()) for name in names})


def _key(row):
    return row["sequence_id"], row["frame_index"]


def _record_hash(row):
    return digest({k: row[k] for k in ("sequence_id", "frame_index", "split", "motion_id", "files")})


def verify_promotion(report, report_sha256, review, config, binding, train_records):
    """Require an externally pinned review in addition to operational checks."""
    _require(isinstance(report, dict) and isinstance(review, dict) and _hash(report_sha256),
             "direct_full_pilot_evidence_required")
    _require(review.get("schema") == "direct_rgb2d_promotion_review_v1"
             and review.get("pilot_report_sha256") == report_sha256
             and review.get("decision") == "approve_full_train"
             and all(isinstance(review.get(k), str) and review[k].strip() and len(review[k]) <= 2048
                     for k in ("reviewer", "learning_signal_evidence", "representative_prediction_evidence")),
             "direct_full_review_required")
    prior = report.get("binding", {})
    _require(report.get("schema") == RUN_SCHEMA and report.get("status") == "completed"
             and report.get("operational_pilot_passed") is True and prior.get("phase") == "pilot"
             and prior.get("train_count") == PILOT_TRAIN_COUNT
             and report.get("comparison_config_sha256") == comparison_config_sha256(config)
             and report.get("diagnostic_only") is True and report.get("train_ready") is False,
             "direct_full_successful_pilot_required")
    _require(all(prior.get(key) == binding[key] for key in
                 ("validation_keys_sha256", "validation_payloads_sha256", "data_identity",
                  "training_contract_sha256")),
             "direct_full_validation_or_data_changed")
    progress = report.get("diagnostic_progress", {})
    checkpoint = report.get("checkpoint", {})
    _require(progress.get("last_completed_unique_train") == PILOT_TRAIN_COUNT
             and progress.get("nonzero_gradient_updates", 0) > 0
             and report.get("coverage", {}).get("completed_epochs", 0) >= 1
             and _hash(checkpoint.get("sha256")) and _hash(report.get("weights_file_sha256"))
             and checkpoint.get("step", 0) > 0
             and _hash(report.get("initial_weights_sha256")) and _hash(report.get("final_weights_sha256"))
             and report["initial_weights_sha256"] != report["final_weights_sha256"],
             "direct_full_actual_coverage_or_update_missing")
    for phase in ("initial_metrics", "final_metrics"):
        for split in ("train", "validation"):
            metric = report.get(phase, {}).get(split, {})
            expected = PILOT_TRAIN_COUNT if split == "train" else binding["validation_count"]
            expected_step = 0 if phase == "initial_metrics" else checkpoint["step"]
            _require(metric.get("records") == expected and metric.get("representative_predictions")
                     and metric.get("step") == expected_step and metric.get("loss_kind") == config["loss_kind"]
                     and set(metric.get("groups", {})) == {"whole", "body", "hands"}
                     and all(type(metric.get(k)) in (int, float) and math.isfinite(metric[k])
                             for k in ("loss", "native_mean_error_pixels", "center_mean_error_pixels",
                                       "mean_confidence", "mean_entropy")),
                     "direct_full_pilot_metrics_missing")
    rows = report.get("pilot_train_records", [])
    _require(isinstance(rows, list) and len(rows) == PILOT_TRAIN_COUNT
             and all(isinstance(row, list) and len(row) == 3 for row in rows),
             "direct_full_pilot_keys_missing")
    pilot = {(seq, frame): sha for seq, frame, sha in rows}
    _require(len(pilot) == PILOT_TRAIN_COUNT and all(train_records.get(key) == sha for key, sha in pilot.items()),
             "direct_full_pilot_train_payload_changed")


class _Sampler(EpochSampler):
    POLICY = "direct_rgb2d_chunked_epoch_v1"

    def __init__(self, catalog, seed):
        self.catalog, self.seed, self.split, self.shuffle = catalog, seed, "train", True
        self.sizes = {i: row["sequences"][0]["records"] for i, row in enumerate(catalog["shards"])
                      if row["sequences"][0]["split"] == "train"}
        self.epoch = self.shard_position = self.offset = 0

    def _policy(self):
        return self.POLICY


class _Chunk:
    def __init__(self, reader, start, guard):
        self.reader, self.start, self.guard = reader, start, guard

    def load_batch(self, indices):
        indices = [self.start + index for index in indices]
        if self.guard is None:
            return self.reader.load_batch(indices)
        size = sum(item["size_bytes"] + 1 for index in indices
                   for item in self.reader.metadata(index)["files"].values())
        self.guard.reserve_read(size)
        return self.reader.load_batch(indices)


class _Dataset:
    def __init__(self, readers, shards, guard):
        self.readers, self.shards, self.guard = readers, shards, guard

    def verify_all(self, check_budget):
        check_budget()
        return {"metadata_records": {split: len(reader) for split, reader in self.readers.items()}}

    def load(self, shard, split):
        row = self.shards[shard]
        _require(row["sequences"][0]["split"] == split, "direct_rgb_shard_split")
        return _Chunk(self.readers[split], row["start"], self.guard)


class DirectRGBTrainer(RGBTrainer):
    CHECKPOINT_FORMAT = FORMAT
    TRAINING_SCOPE = SCOPE

    def __init__(self, config, reader, device="cpu", *, guard=None, promotion=None):
        config = direct_config(config)
        self.loss_kind = config["loss_kind"]
        self.guard, self.promotion = guard, promotion
        super().__init__({k: v for k, v in config.items() if k != "loss_kind"}, reader, None, device,
                         check_budget=guard.check if guard else None)
        self.config = config
        self.source["code_sha256"].update(_code_identity())
        self.initial_weights_sha256 = weights_hash(self.model)
        self.epoch_bitmap = bytearray((self.binding["train_count"] + 7) // 8)
        self.epoch_unique = self.last_completed_unique_train = self.nonzero_gradient_updates = 0

    def _open_data(self, reader, base):
        _require(isinstance(reader, ExtractedStereoT1) and reader.split == "train", "direct_rgb_extracted_reader_required")
        info = reader.identity
        readers = {"train": reader, "validation": reader.for_split("validation")}
        train_count, validation_count = len(reader), len(readers["validation"])
        self.train_records = {}
        for i in range(train_count):
            if i % CHUNK_SIZE == 0:
                self.check_budget()
            row = reader.metadata(i)
            self.train_records[_key(row)] = _record_hash(row)
        validation_hashes = []
        for i in range(validation_count):
            if i % CHUNK_SIZE == 0:
                self.check_budget()
            validation_hashes.append(_record_hash(readers["validation"].metadata(i)))
        self.train_indices = {key: i for i, key in enumerate(self.train_records)}
        limit = info["selection"]["train_limit"]
        phase = "full" if limit is None else "pilot"
        pins = info["provenance"]
        self.binding = {
            "phase": phase, "manifest_sha256": reader.manifest_sha256, "train_count": train_count,
            "validation_count": validation_count, "validation_keys_sha256": info["selection"]["validation_keys_sha256"],
            "validation_payloads_sha256": digest(validation_hashes),
            "training_contract_sha256": _training_contract_sha256(),
            "data_identity": {"dataset_version": info["dataset_version"], "environment": info["environment"],
                              **{key: pins[key] for key in ("spec_sha256", "calibration_sha256", "split_files",
                                                           "motion_map_sha256", "inventory_sha256")}},
            "original_motion_status": info["audit"]["original_motion_status"],
            "diagnostic_only": True, "independent_validation_verified": False,
        }
        if phase == "full":
            _require(self.promotion is not None, "direct_full_pilot_evidence_required")
            report, report_sha, review, review_sha = self.promotion
            _require(_hash(review_sha), "direct_full_review_pin_required")
            verify_promotion(report, report_sha, review, {**self.config, "loss_kind": self.loss_kind},
                             self.binding, self.train_records)
            self.binding.update(pilot_report_sha256=report_sha, promotion_review_sha256=review_sha)
        else:
            _require(type(limit) is int and 1 <= limit <= PILOT_TRAIN_COUNT and train_count == limit
                     and self.promotion is None, "direct_rgb_pilot_selection")
        contract = {"joint_names": info["joint_names"], "body_joint_names": info["body_joint_names"],
                    "image_sizes": [[1024, 1024], [1024, 1024]], "data_kind": "dataset_sample"}
        shards = []
        for split, split_reader in readers.items():
            for start in range(0, len(split_reader), CHUNK_SIZE):
                shards.append({"start": start, "sequences": [{"split": split,
                               "records": min(CHUNK_SIZE, len(split_reader) - start)}]})
        body = {"schema_version": "direct_rgb2d_runtime_catalog_v1", "binding": self.binding,
                "contract": contract, "limits": {"max_records": 4, "max_batch_tensor_bytes": 128 * 1024**2},
                "shards": shards}
        return {**body, "content_sha256": digest(body)}, SimpleNamespace(**contract), _Dataset(readers, shards, self.guard)

    def _new_sampler(self):
        return _Sampler(self.catalog, self.config["seed"])

    def _pose_metadata(self, metadata):
        return metadata

    def _loss(self, logits, gt_xy, valid):
        return heatmap_loss(logits.float(), gt_xy.float(), valid, self.config["training_image_size"],
                            loss_kind=self.loss_kind)

    def _update_observation(self):
        return {"gradient_l2": math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                             for p in self.model.parameters() if p.grad is not None))}

    def train_step(self):
        epoch = self.sampler.epoch
        event = super().train_step()
        if event["updated"]:
            try:
                _require(math.isfinite(event["gradient_l2"]), "direct_rgb_gradient_l2")
                self.nonzero_gradient_updates += event["gradient_l2"] > 0
                for key in event["keys"]:
                    byte, bit = divmod(self.train_indices[tuple(key)], 8)
                    _require(not self.epoch_bitmap[byte] & (1 << bit), "direct_rgb_repeated_epoch_sample")
                    self.epoch_bitmap[byte] |= 1 << bit
                    self.epoch_unique += 1
                if self.sampler.epoch != epoch:
                    _require(self.epoch_unique == self.binding["train_count"], "direct_rgb_epoch_coverage")
                    self.last_completed_unique_train = self.epoch_unique
                    self.epoch_bitmap = bytearray(len(self.epoch_bitmap))
                    self.epoch_unique = 0
            except BaseException:
                self.poisoned = True
                raise
        return event

    def progress(self):
        return {"current_unique_train": self.epoch_unique,
                "last_completed_unique_train": self.last_completed_unique_train,
                "nonzero_gradient_updates": self.nonzero_gradient_updates}

    def checkpoint(self):
        return {**super().checkpoint(), "diagnostic_progress": self.progress(),
                "epoch_bitmap": self.epoch_bitmap.hex(), "initial_weights_sha256": self.initial_weights_sha256}

    def resume(self, checkpoint):
        progress = checkpoint["diagnostic_progress"]
        bitmap = bytearray.fromhex(checkpoint["epoch_bitmap"])
        current, last, nonzero = (progress[k] for k in
                                  ("current_unique_train", "last_completed_unique_train", "nonzero_gradient_updates"))
        count = self.binding["train_count"]
        sampler = self._new_sampler()
        sampler.load_state_dict(checkpoint["cursor"])
        expected = bytearray(len(self.epoch_bitmap))
        for i, shard in enumerate(sampler._order()[:sampler.shard_position + 1]):
            indices = list(range(sampler.sizes[shard]))
            if i == sampler.shard_position:
                random.Random(f"{sampler.seed}:{sampler.epoch}:{shard}:anchors").shuffle(indices)
                indices = indices[:sampler.offset]
            for index in indices:
                byte, bit = divmod(self.catalog["shards"][shard]["start"] + index, 8)
                expected[byte] |= 1 << bit
        _require(all(type(n) is int for n in (current, last, nonzero))
                 and bitmap == expected
                 and sum(x.bit_count() for x in bitmap) == current == checkpoint["samples_seen"] % count
                 and last == (count if checkpoint["cursor"]["epoch"] else 0)
                 and 0 <= nonzero <= checkpoint["step"]
                 and checkpoint["initial_weights_sha256"] == self.initial_weights_sha256,
                 "direct_rgb_checkpoint_coverage")
        super().resume(checkpoint)
        self.epoch_bitmap, self.epoch_unique = bitmap, current
        self.last_completed_unique_train, self.nonzero_gradient_updates = last, nonzero

    def evaluate(self, split="validation", final_test=False):
        _require(split in ("train", "validation") and not final_test and not self.poisoned,
                 "direct_rgb_diagnostic_split")
        mode, rng = self.model.training, rng_state()
        cursor = self.sampler.state_dict()
        counts = defaultdict(lambda: defaultdict(float))
        samples, representatives, loss_sum, loss_count = 0, [], 0.0, 0
        names = self.contract.joint_names
        body = torch.tensor([n in self.contract.body_joint_names for n in names], device=self.device)
        groups = {"whole": torch.ones_like(body), "body": body, "hands": ~body}
        width, height = self.config["training_image_size"]
        try:
            self.model.eval()
            with torch.no_grad():
                for shard, descriptor in enumerate(self.catalog["shards"]):
                    group = descriptor["sequences"][0]
                    if group["split"] != split:
                        continue
                    for start in range(0, group["records"], self.config["batch_size"]):
                        self.check_budget()
                        batch = self._batch(shard, split, list(range(
                            start, min(group["records"], start + self.config["batch_size"]))))
                        logits, xy, confidence = self.model(batch["images"].to(self.device))
                        valid, target = batch["valid"].to(self.device), batch["gt_xy"].to(self.device)
                        inside = valid & (target[..., 0] >= 0) & (target[..., 0] <= width - 1)
                        inside &= (target[..., 1] >= 0) & (target[..., 1] <= height - 1)
                        n = int(inside.sum())
                        if n:
                            loss, _ = self._loss(logits, target, valid)
                            loss_sum += float(loss) * n
                            loss_count += n
                        native = scale_xy(xy, [width, height], [1024, 1024])
                        gt = batch["native_gt_xy"].to(self.device)
                        error = torch.linalg.vector_norm(native - gt, dim=-1)
                        center = torch.linalg.vector_norm(511.5 - gt, dim=-1)
                        logp = logits.float().flatten(-2).log_softmax(-1)
                        entropy = -(logp.exp() * logp).sum(-1)
                        _require(all(torch.isfinite(x).all().item() for x in (error, center, confidence, entropy)),
                                 "direct_rgb_nonfinite_evaluation")
                        for name, group_mask in groups.items():
                            accum = counts[name]
                            accum["possible"] += len(batch["keys"]) * 2 * int(group_mask.sum())
                            for scope, mask in (("valid", valid), ("inside", inside), ("outside", valid & ~inside)):
                                mask = mask & group_mask
                                accum[scope] += int(mask.sum())
                                for metric, values in (("error", error), ("center", center),
                                                       ("confidence", confidence), ("entropy", entropy)):
                                    accum[scope + "_" + metric] += float(values[mask].sum())
                        for i, key in enumerate(batch["keys"][:max(0, 2 - len(representatives))]):
                            representatives.append({"key": key, "prediction_xy": native[i].cpu().tolist(),
                                "gt_xy": gt[i].cpu().tolist(), "valid": valid[i].cpu().tolist(),
                                "confidence": confidence[i].cpu().tolist()})
                        samples += len(batch["keys"])
                        self.check_budget()
        finally:
            self.model.train(mode)
            restore_rng(rng)
        _require(cursor == self.sampler.state_dict() and samples == self.counts[split], "direct_rgb_eval_coverage")
        _require(loss_count > 0 and math.isfinite(loss_sum), "direct_rgb_eval_loss_coverage")
        report_groups = {}
        for name, accum in counts.items():
            row = {"possible_joint_views": int(accum["possible"]),
                   "annotation_coverage": accum["valid"] / accum["possible"]}
            for scope in ("valid", "inside", "outside"):
                n = accum[scope]
                row[scope] = {"joint_views": int(n), **{metric: accum[scope + "_" + metric] / n if n else None
                                                      for metric in ("error", "center", "confidence", "entropy")}}
            report_groups[name] = row
        overall = report_groups["whole"]["valid"]
        return {"split": split, "records": samples, "step": self.step, "groups": report_groups,
                "loss_kind": self.loss_kind, "loss": loss_sum / loss_count,
                "native_mean_error_pixels": overall["error"], "center_mean_error_pixels": overall["center"],
                "mean_confidence": overall["confidence"], "mean_entropy": overall["entropy"],
                "representative_predictions": representatives, "diagnostic_only": True,
                "independent_validation_verified": False}

"""Fixed-manifest ArchVis 1,000-pair context heatmap training and inference."""

import argparse
import io
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from PIL import Image, ImageDraw
import torch

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_prepare.converter import sha
from rewind_prepare.extracted import ExtractedStereoT1
from .archvis_v0_model import ARCHITECTURE, ContextHeatmapNet
from .extracted_rgb_cli import MIB, RunLimits, _Budget, _serialized_state
from .extracted_rgb_training import DirectRGBTrainer
from .rgb_frontend_artifacts import scale_xy
from .shards import _hash, _read, _require


FORMAT = "archvis_1000_context_checkpoint_v0"
SCOPE = "archvis_1000_v0_2d_development"
MANIFEST = "ac9227722b7c780d7df7b1fb2b64e8e8be4d2e7076bab0b086668bdc4b962595"
VALIDATION = "b4c52d654e772771e553ce887f81f502ea80082e825bab63d02f7c6088ba2dee"
COUNTS = {"train": 1000, "validation": 2236}
EVALUATE = (0, 1, 5, 10, 15, 20)
CHECKPOINT_EVERY = 100
MAX_ATTEMPTS = 10000
PREVIOUS_CE_EPOCH1 = {"updates": 500, "whole_inside_native_error_pixels": 334.707,
                      "whole_valid_native_error_pixels": 359.584, "source": "docs/ARCHVIS1000_RESULTS.md"}
CONFIG = {
    "seed": 71, "batch_size": 2, "learning_rate": 1e-4, "amp": False,
    "training_image_size": [64, 64], "scheduler_step": 51, "scheduler_gamma": 1.0,
    "frontend_version": ARCHITECTURE, "loss_kind": "gaussian_cross_entropy",
    "max_native_tensor_bytes": 64 * MIB, "max_heatmap_elements": 2 * MIB,
}
CACHE_LIMIT = 384 * MIB
CHECKPOINT_LIMIT = 8 * MIB
DEMO_LIMIT = 4 * MIB
ARTIFACT_LIMIT = 96 * MIB
WALL_SECONDS = 7200
RUN_SECONDS = 7080
RSS_LIMIT = 6 * 1024 * MIB
CUDA_ALLOCATED_LIMIT = 4 * 1024 * MIB
CUDA_RESERVED_LIMIT = 6 * 1024 * MIB


def _planned_payload(reader, counts):
    return sum(item["size_bytes"] + 1 for split, count in counts.items()
               for split_reader in [reader.for_split(split)]
               for i in range(count) for item in split_reader.metadata(i)["files"].values())


class _LocalBatchCache:
    """Process-local cache. Hashes are derived only from source-verified preprocessed batches."""

    def __init__(self, path):
        self.path = path
        self.pins = {}
        self.write_bytes = self.read_bytes = 0
        self.build_seconds = self.read_seconds = 0.0

    def build(self, trainer, load):
        started = time.monotonic()
        try:
            for shard, descriptor in enumerate(trainer.catalog["shards"]):
                group = descriptor["sequences"][0]
                split = group["split"]
                for index in range(group["records"]):
                    batch = load(shard, split, [index])
                    payload = _serialized_state(batch)
                    if self.write_bytes + len(payload) > CACHE_LIMIT:
                        raise TimeoutError("archvis_v0_cache_limit")
                    path = self.path / f"{shard}-{index}.pt"
                    with path.open("xb") as handle:
                        handle.write(payload)
                    self.pins[shard, index] = sha(payload)
                    self.write_bytes += len(payload)
        finally:
            self.build_seconds = time.monotonic() - started

    def batch(self, shard, indices):
        started = time.monotonic()
        rows = []
        for index in indices:
            pin = self.pins[shard, index]
            raw = _read(self.path / f"{shard}-{index}.pt", MIB)
            _require(sha(raw) == pin, "archvis_v0_cache_hash")
            self.read_bytes += len(raw)
            rows.append(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
        self.read_seconds += time.monotonic() - started
        batch = {key: torch.cat([row[key] for row in rows])
                 for key in ("images", "gt_xy", "valid", "native_gt_xy")}
        return {**batch, "keys": [row["keys"][0] for row in rows]}


class ContextTrainer(DirectRGBTrainer):
    CHECKPOINT_FORMAT = FORMAT
    TRAINING_SCOPE = SCOPE

    def __init__(self, reader, device, guard, cache):
        self.cache = cache
        super().__init__(CONFIG, reader, device, guard=guard)

    def _make_model(self, joints):
        return ContextHeatmapNet(joints).to(self.device)

    def _batch(self, shard, split, indices):
        self.check_budget()
        _require(self.catalog["shards"][shard]["sequences"][0]["split"] == split,
                 "archvis_v0_cache_split")
        return self.cache.batch(shard, indices)

    def fill_cache(self):
        self.cache.build(self, super()._batch)

    def _update_observation(self):
        gradient_squares = torch.stack([p.grad.detach().double().square().sum()
                                        for p in self.model.parameters() if p.grad is not None])
        return {"gradient_l2": float(gradient_squares.sum().sqrt())}


def read_run_checkpoint(path, expected_sha256):
    raw = _read(path, CHECKPOINT_LIMIT)
    _require(_hash(expected_sha256) and sha(raw) == expected_sha256, "archvis_v0_checkpoint_pin")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _require(state["format_version"] == FORMAT and state["training_scope"] == SCOPE
             and state["full_training_ready"] is False and state["config"] == CONFIG,
             "archvis_v0_checkpoint_contract")
    return state


def _best_payload(directory, info):
    name = info["name"]
    _require(Path(name).name == name and name.startswith("best_e"), "archvis_v0_best_filename")
    raw = _read(directory / name, CHECKPOINT_LIMIT)
    _require(sha(raw) == info["sha256"] and len(raw) == info["bytes"], "archvis_v0_best_pin")
    return raw


def recover_run(work):
    """Return the authoritative journal/checkpoint even if the report lagged an atomic save."""
    work = Path(work)
    output = work / "v0"
    payloads = {}
    for name, maximum in (("manifest/manifest.json", 8 * MIB), ("prepare_report.json", 8 * MIB),
                          ("preflight.json", 8 * MIB), ("restore_report.json", MIB),
                          ("v0_plan.json", MIB), ("v0_launch.json", MIB),
                          ("bootstrap_report.json", MIB), ("v0_failure.json", MIB)):
        if (work / name).is_file():
            payloads[name] = _read(work / name, maximum)
    if (output / "report.json").is_file():
        payloads["report.json"] = _read(output / "report.json", 4 * MIB)
    resume_files = {}
    if (output / "last.pt").is_file():
        last = payloads["last.pt"] = _read(output / "last.pt", CHECKPOINT_LIMIT)
        state = read_run_checkpoint(output / "last.pt", sha(last))
        journal = state["journal"]
        payloads["epoch_log.json"] = canonical(journal) + b"\n"
        resume_files["last.pt"] = "last.pt"
        if journal["best"]:
            for kind, name in (("checkpoint", "best.pt"), ("demo", "predictions.zip")):
                info = journal["best"][kind]
                payloads[name] = _best_payload(output, info)
                resume_files[name] = info["name"]
    receipt = {"schema": SCOPE, "files": {name: {"bytes": len(raw), "sha256": sha(raw)}
                                        for name, raw in payloads.items()}, "resume_files": resume_files}
    target = work / "archvis_1000_v0_recovery.zip"
    with tempfile.NamedTemporaryFile(dir=work, suffix=".zip.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in payloads.items():
                archive.writestr(name, raw)
            archive.writestr("recovery_receipt.json", canonical(receipt) + b"\n")
        with zipfile.ZipFile(temporary) as archive:
            _require(archive.testzip() is None, "archvis_v0_recovery_crc")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, receipt


def restore_run(recovery, output, *, plan_sha256):
    """Copy only pinned durable state from an already archive-verified recovery directory."""
    recovery, output = Path(recovery), Path(output)
    receipt = read_json(_read(recovery / "recovery_receipt.json", MIB))
    plan = read_json(_read(recovery / "v0_plan.json", MIB))
    _require(receipt["schema"] == SCOPE and plan["content_sha256"] == plan_sha256 ==
             digest({key: value for key, value in plan.items() if key != "content_sha256"}),
             "archvis_v0_resume_plan")
    entries = receipt["resume_files"]
    _require(set(entries) in ({"last.pt"}, {"last.pt", "best.pt", "predictions.zip"}),
             "archvis_v0_resume_members")
    last = read_run_checkpoint(recovery / "last.pt", receipt["files"]["last.pt"]["sha256"])
    best = last["journal"]["best"]
    expected = {"last.pt": "last.pt"}
    if best:
        expected.update({"best.pt": best["checkpoint"]["name"], "predictions.zip": best["demo"]["name"]})
    _require(entries == expected, "archvis_v0_resume_member_names")
    payloads = {}
    for name, destination in entries.items():
        _require(Path(destination).name == destination, "archvis_v0_resume_path")
        raw = _read(recovery / name, CHECKPOINT_LIMIT)
        _require(receipt["files"][name] == {"bytes": len(raw), "sha256": sha(raw)},
                 "archvis_v0_resume_file_pin")
        payloads[destination] = raw
    output.mkdir(exist_ok=False)
    for name, raw in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return receipt["files"]["last.pt"]["sha256"]


def load_context_checkpoint(path, expected_sha256):
    """Load the selected inference model without restoring optimizer, data or RNG."""
    state = read_run_checkpoint(path, expected_sha256)
    contract = state["catalog"]["contract"]
    with torch.random.fork_rng(devices=[]):
        model = ContextHeatmapNet(len(contract["joint_names"]))
    model.load_state_dict(state["model"], strict=True)
    _require(all(torch.isfinite(p).all().item() for p in model.parameters()), "archvis_v0_nonfinite_weights")
    return model.eval(), contract


def _image_bytes(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@torch.inference_mode()
def _demo(trainer):
    representatives = [(shard, descriptor["sequences"][0]["records"] // 2)
                       for shard, descriptor in enumerate(trainer.catalog["shards"])
                       if descriptor["sequences"][0]["split"] == "validation"]
    locations = (representatives[0], representatives[len(representatives) // 2], representatives[-1])
    body_names = set(trainer.contract.body_joint_names)
    names = list(trainer.contract.joint_names)
    out = io.BytesIO()
    predictions = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sample, (shard, index) in enumerate(dict.fromkeys(locations)):
            batch = trainer._batch(shard, "validation", [index])
            trainer.model.eval()
            logits, xy, confidence = trainer.model(batch["images"].to(trainer.device))
            native_xy = scale_xy(xy, [64, 64], [1024, 1024])
            predictions.append({"key": batch["keys"][0], "native_prediction_xy": native_xy[0].tolist(),
                                "confidence": confidence[0].tolist(), "native_gt_xy": batch["native_gt_xy"][0].tolist(),
                                "annotation_valid": batch["valid"][0].tolist()})
            probability = logits.float().flatten(-2).softmax(-1).reshape_as(logits)[0]
            for view, side in enumerate(("left", "right")):
                rgb = batch["images"][0, view].permute(1, 2, 0).mul(255).round().to(torch.uint8)
                overlay = Image.frombytes("RGB", (64, 64), bytes(rgb.flatten().tolist())).resize((384, 384))
                draw = ImageDraw.Draw(overlay)
                for name, (x, y) in zip(names, xy[0, view].tolist()):
                    x, y = x * 383 / 63, y * 383 / 63
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                                 fill="#4ee8ff" if name in body_names else "#ffca55")
                peak = probability[view].amax(dim=0).cpu()
                peak = (peak / peak.max().clamp_min(1e-12) * 255).round().to(torch.uint8)
                heatmap = Image.frombytes("L", (32, 32), bytes(peak.flatten().tolist())).resize((384, 384))
                archive.writestr(f"{sample}_{side}_overlay.png", _image_bytes(overlay))
                archive.writestr(f"{sample}_{side}_heatmap.png", _image_bytes(heatmap))
        archive.writestr("predictions.json", canonical({
            "joint_names": names, "step": trainer.step, "source": "model_predictions",
            "overlay": "64px input enlarged; body cyan, hands gold",
            "heatmap": "per-view maximum across joints, normalized to its own peak",
            "records": predictions,
        }) + b"\n")
    return out.getvalue()


def run(reader, output, device, *, resume_sha256=None, seconds=RUN_SECONDS):
    """Run the fixed experiment with atomic last state and immutable best generations."""
    output = Path(output)
    _require(0 < seconds <= RUN_SECONDS and reader.manifest_sha256 == MANIFEST
             and reader.identity["selection"]["validation_keys_sha256"] == VALIDATION
             and all(len(reader.for_split(split)) == n for split, n in COUNTS.items()), "archvis_v0_fixed_selection")
    output.mkdir(exist_ok=True)
    lock = output / ".running"
    lock.mkdir()
    try:
        return _run_locked(reader, output, device, resume_sha256, seconds)
    finally:
        lock.rmdir()


def _run_locked(reader, output, device, resume_sha256, seconds):
    prior = None
    if resume_sha256:
        prior = read_run_checkpoint(output / "last.pt", resume_sha256)
        best = prior["journal"]["best"]
        if best:
            for key in ("checkpoint", "demo"):
                _best_payload(output, best[key])
    else:
        _require(not any(p.name != ".running" for p in output.iterdir()), "archvis_v0_output_requires_resume")
    _require(shutil.disk_usage(output).free >= CACHE_LIMIT + ARTIFACT_LIMIT + 128 * MIB,
             "archvis_v0_local_disk_space")
    payload_limit = _planned_payload(reader, COUNTS)
    limits = RunLimits(seconds, MAX_ATTEMPTS, payload_limit, RSS_LIMIT, ARTIFACT_LIMIT, CUDA_ALLOCATED_LIMIT,
                       CUDA_RESERVED_LIMIT)
    budget = _Budget(limits, output, device)
    journal = {"epochs": [], "evaluations": [], "best": None,
               "in_progress": {"loss_sum": 0.0, "joints": 0, "seconds": 0.0}}
    report = {"schema": SCOPE, "status": "started", "architecture": ARCHITECTURE, "config": CONFIG,
              "manifest_sha256": reader.manifest_sha256, "validation_keys_sha256": VALIDATION,
              "counts": COUNTS, "evaluation_epochs": EVALUATE, "train_ready": False,
              "independent_validation_verified": False, "early_stop": "resource limits only",
              "best_metric": "trained epochs only; official validation whole inside mean native 1024px error; "
                             "earlier epoch wins ties",
              "previous_tiny_ce_epoch1": PREVIOUS_CE_EPOCH1,
              "recovery_window": {"checkpoint_every_updates": CHECKPOINT_EVERY,
                                  "maximum_updates_to_replay": CHECKPOINT_EVERY - 1,
                                  "checkpoint_write_failure_may_replay_boundary_update": True,
                                  "unfinished_evaluation_repeats_at_same_step": True}}
    durable = None
    if prior:
        journal = prior["journal"]
        durable = {"sha256": resume_sha256, "step": prior["step"], "epoch": prior["cursor"]["epoch"],
                   "journal": journal}
        report["resumed_from_sha256"] = resume_sha256
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="archvis_v0_cache_") as directory:
        cache = _LocalBatchCache(Path(directory))
        checkpoint_seconds = demo_seconds = 0.0

        def publish_report():
            report.update(durable=durable, resources=budget.summary(), cache={
                "bytes_written": cache.write_bytes, "bytes_read": cache.read_bytes,
                "build_seconds_including_source_io_decode_resize": cache.build_seconds,
                "read_seconds_including_hash_and_deserialization": cache.read_seconds,
            }, checkpoint_seconds=checkpoint_seconds, demo_seconds=demo_seconds)
            if durable and durable["journal"]["best"]:
                evaluated = durable["journal"]["evaluations"]
                initial = evaluated[0]["metrics"]["groups"]["whole"]["inside"]["error"]
                best = durable["journal"]["best"]["error_pixels"]
                report["best_minus_initial_whole_inside_pixels"] = best - initial
                epoch_one = evaluated[1]["metrics"]
                if epoch_one["step"] == PREVIOUS_CE_EPOCH1["updates"]:
                    error = epoch_one["groups"]["whole"]["inside"]["error"]
                    report["epoch1_minus_previous_tiny_ce_whole_inside_pixels"] = (
                        error - PREVIOUS_CE_EPOCH1["whole_inside_native_error_pixels"])
            budget.publish("report.json", canonical(report) + b"\n", 4 * MIB)

        try:
            publish_report()
            trainer = ContextTrainer(reader, device, budget, cache)
            if prior:
                trainer.resume(prior)
            report.update(binding=trainer.binding, source=trainer.source, environment=trainer.environment,
                          base_source_revision=trainer.source["revision"],
                          v0_module_sha256={name: trainer.source["code_sha256"][name] for name in (
                              "rewind_m1/archvis_v0.py", "rewind_m1/archvis_v0_model.py")},
                          parameters=sum(p.numel() for p in trainer.model.parameters()))

            def persist():
                nonlocal durable, checkpoint_seconds
                started = time.monotonic()
                payload = _serialized_state({**trainer.checkpoint(), "journal": journal})
                pin = budget.publish("last.pt", payload, CHECKPOINT_LIMIT)
                checkpoint_seconds += time.monotonic() - started
                durable = {"sha256": pin, "step": trainer.step, "epoch": trainer.sampler.epoch,
                           "journal": read_json(canonical(journal))}
                publish_report()

            def evaluate(epoch):
                nonlocal checkpoint_seconds, demo_seconds
                started = time.monotonic()
                metrics = trainer.evaluate("validation")
                evaluation_seconds = time.monotonic() - started
                score = metrics["groups"]["whole"]["inside"]["error"]
                if epoch and (journal["best"] is None or score < journal["best"]["error_pixels"]):
                    # An interrupted publish may leave an unreferenced generation. Each generation is keyed by bytes.
                    started = time.monotonic()
                    checkpoint = _serialized_state(trainer.checkpoint())
                    checkpoint_seconds += time.monotonic() - started
                    started = time.monotonic()
                    demo = _demo(trainer)
                    demo_seconds += time.monotonic() - started
                    selection = {"epoch": epoch, "step": trainer.step, "error_pixels": score}
                    for kind, raw, extension, limit in (("checkpoint", checkpoint, "pt", CHECKPOINT_LIMIT),
                                                         ("demo", demo, "zip", DEMO_LIMIT)):
                        started = time.monotonic()
                        pin = sha(raw)
                        name = f"best_e{epoch}_{pin[:16]}.{extension}"
                        budget.publish(name, raw, limit)
                        elapsed = time.monotonic() - started
                        if kind == "checkpoint":
                            checkpoint_seconds += elapsed
                        else:
                            demo_seconds += elapsed
                        selection[kind] = {"name": name, "sha256": pin, "bytes": len(raw)}
                    journal["best"] = selection
                journal["evaluations"].append({"epoch": epoch, "seconds": evaluation_seconds, "metrics": metrics})
                persist()
                print(f"validation epoch={epoch} step={trainer.step} inside_error_px={score:.3f}", flush=True)

            if prior is None:
                persist()
            trainer.fill_cache()
            _require(budget.read_bytes == payload_limit, "archvis_v0_verified_payload_coverage")
            if not journal["evaluations"] or (trainer.sampler.epoch in EVALUATE and
                    journal["evaluations"][-1]["epoch"] < trainer.sampler.epoch):
                evaluate(trainer.sampler.epoch)
            while trainer.sampler.epoch < EVALUATE[-1]:
                started = time.monotonic()
                budget.check()
                _require(trainer.attempts < MAX_ATTEMPTS, "archvis_v0_attempt_limit")
                event = trainer.train_step()
                _require(event["updated"], "archvis_v0_fp32_update_missing")
                progress = journal["in_progress"]
                progress["loss_sum"] += event["loss"] * event["inside_joints"]
                progress["joints"] += event["inside_joints"]
                progress["seconds"] += time.monotonic() - started
                epoch = trainer.sampler.epoch
                completed = trainer.epoch_unique == 0
                if completed:
                    journal["epochs"].append({"epoch": epoch, "step": trainer.step,
                        "unique_train": trainer.last_completed_unique_train,
                        "train_loss": progress["loss_sum"] / progress["joints"],
                        "train_seconds": progress["seconds"],
                        "pairs_per_second": COUNTS["train"] / progress["seconds"],
                        "nonzero_gradient_updates": trainer.nonzero_gradient_updates})
                    journal["in_progress"] = {"loss_sum": 0.0, "joints": 0, "seconds": 0.0}
                    print(f"train epoch={epoch} step={trainer.step} pairs_per_second="
                          f"{journal['epochs'][-1]['pairs_per_second']:.2f}", flush=True)
                if completed or trainer.step % CHECKPOINT_EVERY == 0:
                    persist()
                if completed and epoch in EVALUATE:
                    evaluate(epoch)
            report["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="bounded_stop" if isinstance(exc, TimeoutError) else
                          "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
            if not isinstance(exc, TimeoutError):
                raise
        finally:
            publish_report()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "rgb-root", "pose-root", "spec", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args(argv)
    reader = ExtractedStereoT1(args.manifest, args.rgb_root, args.pose_root, args.spec,
                               expected_sha256=MANIFEST, split="train")
    run(reader, args.output, args.device, resume_sha256=args.resume_sha256)


if __name__ == "__main__":
    main()

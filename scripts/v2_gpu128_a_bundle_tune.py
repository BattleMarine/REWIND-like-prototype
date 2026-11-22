"""Run a bounded, stage-specific warmstart from the final v2 128 A bundle."""

import argparse
from copy import copy, deepcopy
import math
import os
from pathlib import Path
import platform
import random
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.training import seed_run
from rewind_prepare.temporal_cache import _atomic
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.run_stages import FrameStage, evaluate_frontend, frame_batch
from rewind_v1.runner import Budget, _fit_state, _publish
from rewind_v1.v2_contract import SCENARIOS
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import EVAL_SEED


ROLES = ("frontend", "body", "hand")
SPLITS = ("train", "validation")
SCHEMA = "rewind_v2_gpu128_stage_warmstart_v1"


def sample_keys(keys, count, *, role, split, seed):
    """Round-robin across scenarios after an independent deterministic shuffle."""
    groups = {name: [] for name in SCENARIOS}
    for key in keys:
        groups[key[0].partition("/")[0]].append(key)
    if any(not group for group in groups.values()) or sum(map(len, groups.values())) < count:
        raise ValueError(f"warmstart_insufficient_{split}")
    for scenario, group in groups.items():
        group.sort()
        random.Random(f"warmstart-v1:{role}:{split}:{scenario}:{seed}").shuffle(group)
    selected = []
    for index in range(max(map(len, groups.values()))):
        for group in groups.values():
            if index < len(group):
                selected.append(group[index])
                if len(selected) == count:
                    return selected
    raise ValueError(f"warmstart_insufficient_{split}")


def select(pipeline, role, train_items, validation_items, seed):
    use_frames = role == "frontend"
    keys = pipeline.data.frames if use_frames else pipeline.data.clips
    return {"train": sample_keys(keys["train"], train_items, role=role, split="train", seed=seed),
            "validation": sample_keys(keys["validation"], validation_items, role=role,
                                      split="validation", seed=EVAL_SEED)}


def _prepare_stage(pipeline, role, samples, cfg):
    data, device, size = pipeline.data, pipeline.device, pipeline.frontend_size
    if role == "frontend":
        stage = FrameStage(cfg["frontend"], len(data.spec["joint_names"]), device)
        view = copy(data)
        view.frames = samples

        def update(batch, budget, timing):
            return stage.update(frame_batch(view, batch, "train", size, budget, timing), size)

        def evaluate(budget, timing):
            return evaluate_frontend(stage, view, cfg["frontend"], budget, timing)
    else:
        frontend = pipeline._make_frontend()
        pipeline._load_best("frontend", frontend)
        frontend_pin = frontend_fingerprint(frontend.model)
        input_sha = digest({"input": data.identity, "frontend_weights": frontend_pin, "size": size,
                            "features": "predicted-2d-v2", "samples": samples})
        clips = {split: {key: data.clip(key, split, frontend.model, size, input_sha) for key in samples[split]}
                 for split in SPLITS}
        del frontend
        cascade = CommonEvaluation(data, clips, cfg, device, 0)
        body = None
        body_pin = None
        if role == "hand":
            body = cascade.make("body")
            pipeline._load_best("body", body)
            body_pin = pipeline.progress["best"]["body"]["sha256"]
        stage = cascade.make(role, body_pin)

        def update(batch, budget, timing):
            return cascade.train(role, stage, batch[0])

        def evaluate(budget, timing):
            return cascade.evaluate(role, stage, budget, timing, body, diagnostics=True)

    pipeline._load_best(role, stage)
    if stage.optimizer.state:
        raise ValueError("warmstart_requires_fresh_optimizer")
    return stage, update, evaluate


def _updates(batches, maximum, seed):
    rng = random.Random(f"warmstart-batches-v1:{seed}")
    done = 0
    while done < maximum:
        order = list(batches)
        rng.shuffle(order)
        for batch in order:
            yield batch
            done += 1
            if done == maximum:
                return


def warmstart(bundle, output, *, role, train_items, validation_items, updates, learning_rate, seed):
    """Tune a single role, retaining the original frontend/body dependencies and split contract."""
    pipeline, ledger = bundle.pipeline, bundle.ledger
    ledger.check()
    seed_run(seed)
    cfg = deepcopy(pipeline.cfg)
    cfg["seed"] = seed
    model_cfg = cfg["frontend" if role == "frontend" else "teacher"]
    if learning_rate is not None:
        model_cfg["learning_rate"] = learning_rate
    samples = select(pipeline, role, train_items, validation_items, seed)
    stage, update, evaluate = _prepare_stage(pipeline, role, samples, cfg)
    upstream = {name: pipeline.progress["best"][name]["sha256"] for name in ROLES[:ROLES.index(role)]}
    baseline_stage = pipeline.progress["best"][role]
    batch_size = cfg["frontend"]["batch_size"] if role == "frontend" else 1
    batches = [samples["train"][i:i+batch_size] for i in range(0, len(samples["train"]), batch_size)]
    timing = {}
    before = evaluate(ledger, timing.setdefault("validation_before", {}))
    before_weights = {name: param.detach().cpu().clone() for name, param in stage.model.named_parameters()}
    stage.model.requires_grad_(True)
    train_steps = []
    for index, batch in enumerate(_updates(batches, updates, seed), 1):
        ledger.check()
        stage.measure_gradient = index == 1
        train_steps.append({"update": index, "items": batch, **update(batch, ledger, timing.setdefault("train", {}))})
    changed = sum(not torch.equal(param.detach().cpu(), before_weights[name]) for name, param in stage.model.named_parameters())
    del before_weights
    if not stage.optimizer.state or not changed:
        raise ValueError("warmstart_no_parameters_updated")
    after = evaluate(ledger, timing.setdefault("validation_after", {}))
    ledger.check()
    provenance = {"parent_identity": pipeline.identity, "parent_receipt_sha256": bundle.diagnostic["checkpoint"]["content_sha256"],
                  "parent_stage_checkpoint": baseline_stage, "upstream_checkpoints": upstream,
                  "stage": role, "updates": updates, "seed": seed, "eval_seed": EVAL_SEED,
                  "config": cfg, "sample_sha256": digest(samples), "sample_unit": "frames" if role == "frontend" else "clips",
                  "samples": samples, "original_optimizer_loaded": False,
                  "upper_policy": "gt_train_predicted_validation" if role == "hand" else None}
    checkpoint = _publish(output / "warmstart.pt", {"schema": SCHEMA, "binding": provenance, "fit": _fit_state(stage)},
                          Budget(cfg["limits"]))
    return {"schema": SCHEMA, "status": "completed", "binding": provenance, "checkpoint": checkpoint,
            "changed_parameter_tensors": changed,
            "optimizer_steps_with_supervision": sum(not step.get("no_supervision_batches", 0) for step in train_steps),
            "training": train_steps, "validation_before": before, "validation_after": after, "timing": timing,
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(pipeline.device),
                            "device_name": torch.cuda.get_device_name(pipeline.device) if pipeline.device.type == "cuda" else None},
            "result_scope": "stage experiment on a deterministic validation subset; preserve upstream checkpoint pins"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path, help="New local scratch directory")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--stage", choices=ROLES, default="frontend")
    parser.add_argument("--train-items", type=int, default=6, help="Frames for frontend, clips for body and hand")
    parser.add_argument("--validation-items", type=int, default=3, help="Frames for frontend, clips for body and hand")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, help="Override the selected stage learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not (3 <= args.train_items <= 24 and 3 <= args.validation_items <= 12 and 1 <= args.updates <= 128
            and 0 <= args.seed < 2**32-1000 and 60 <= args.seconds <= 7200
            and (args.learning_rate is None or math.isfinite(args.learning_rate) and 0 < args.learning_rate <= .01)):
        parser.error("limits: train 3..24, validation 3..12, updates 1..128, seconds 60..7200, seed 0..4294966295, LR (0, .01]")
    bundle = args.bundle.resolve(strict=True)
    output, workspace = args.output.resolve(), args.work.resolve()
    if (not (bundle.is_file() or bundle.is_dir()) or any(path.exists() or path.is_symlink() for path in (args.output, args.work))
            or any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in
                   ((output, workspace), (workspace, bundle), (output, bundle)))):
        parser.error("bundle ZIP or directory and separate new output and scratch directories required")
    output.mkdir()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from v2_gpu128_a_bundle_local import open_bundle

    started = time.monotonic()
    report = {"schema": SCHEMA, "status": "failed"}
    try:
        with open_bundle(bundle, workspace, seconds=args.seconds, device=args.device) as session:
            report = warmstart(session, output, role=args.stage, train_items=args.train_items,
                               validation_items=args.validation_items, updates=args.updates,
                               learning_rate=args.learning_rate, seed=args.seed)
    except BaseException as exc:
        report = {"schema": SCHEMA, "status": "failed", "error": type(exc).__name__, "message": str(exc)[:512]}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        _atomic(output / "warmstart.json", canonical(report))
        print(canonical(report).decode(), flush=True)


if __name__ == "__main__":
    main()

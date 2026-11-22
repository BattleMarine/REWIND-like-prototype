"""Full official-split paces for a new local v2 128 experiment and exact manual resume."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_cache import _atomic, _locked
from rewind_v1.runner import Pipeline
from rewind_v1 import v2_contract as contract
from scripts.run_v2_gpu_training import training_config
from scripts.v2_gpu_training_pipeline import Training

if __package__:
    from . import v2_gpu128_a_bundle_edit as edit
    from .v2_gpu128_a_bundle_format import SOURCE_SHA
else:
    import v2_gpu128_a_bundle_edit as edit
    from v2_gpu128_a_bundle_format import SOURCE_SHA


ROLES = ("frontend", "body", "hand")
GIB = 1024**3


class LocalTraining(Training):
    """Use the same epoch, validation and checkpoint implementation with local storage."""

    def __init__(self, cfg, session, output, identity, initialize, owner):
        parent = session.pipeline
        self.parent_model = parent if initialize == "best" else None
        super().__init__(cfg, parent.data, output/"features", parent.device, output/"run", output, session.ledger, owner, identity)
        self.first_required = False

    def _make_stage(self, role):
        stage = super()._make_stage(role)
        if self.parent_model is not None:
            self.parent_model._load_best(role, stage)
            stage.model.requires_grad_(True)
        return stage

    def _save_last(self, stage):
        Pipeline._save_last(self, stage)
        self.ledger.persist()


def run_config(epochs, seed, run_gib):
    cfg = training_config(seed)
    cfg["run_id"] = "V2-FULL-128-PORTABLE-v1"
    cfg["training"]["epochs"] = dict(zip(ROLES, epochs, strict=True))
    cfg["resources"]["run_bytes"] = run_gib*GIB
    return cfg


def experiment(cfg, session, initialize, editable=None):
    root = Path(__file__).resolve().parent
    pieces = (Path(__file__).name, "v2_gpu128_a_bundle_local.py", "v2_gpu128_a_bundle_format.py", "v2_gpu128_a_bundle_edit.py")
    source = session.pipeline.identity["source"] if editable is None else editable["source_sha256"]
    return {"run_id": cfg["run_id"], "source": source, "seed": cfg["training"]["seed"],
            "parent": session.pipeline.identity, "parent_receipt": session.diagnostic["checkpoint"]["content_sha256"],
            "training_config": digest(cfg), "initialization": initialize, "editable_source": editable,
            "consumer_core_sha256": digest(contract.source()),
            "tool_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in pieces}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "work", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    for role in ROLES:
        parser.add_argument(f"--{role}-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialize", choices=("fresh", "best"), default="fresh")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seconds", type=int, default=28800)
    parser.add_argument("--run-gib", type=int, default=8)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--source-identity-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--editable-origin", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    epochs = tuple(getattr(args, f"{role}_epochs") for role in ROLES)
    if (any(not 1 <= n <= 100 for n in epochs) or not 0 <= args.seed < 2**32-1000 or not 60 <= args.seconds <= 28800
            or not 2 <= args.run_gib <= 32 or args.stop_after_updates is not None and args.stop_after_updates <= 0
            or args.resume_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.resume_sha256)):
        parser.error("epochs 1–100, seed 0–4294966295, seconds 60–28800, run GiB 2–32 and a positive update limit required")
    bundle, work, output = (getattr(args, n).resolve() for n in ("bundle", "work", "output"))
    if (any(getattr(args, n).is_symlink() for n in ("bundle", "work", "output"))
            or any(a.is_relative_to(b) or b.is_relative_to(a) for a,b in ((bundle,output),(bundle,work),(work,output)))
            or output.exists() != bool(args.resume_sha256) or work.exists()):
        parser.error("use separate paths, new work each call, and --resume-sha256 exactly when reusing an output")
    if (args.source_identity_file is None) != (args.editable_origin is None):
        parser.error("use v2_bundle.py train-full --editable-source")
    editable = None
    source_sha = SOURCE_SHA
    if args.source_identity_file is not None:
        origin = args.editable_origin.resolve(strict=True)
        if args.initialize != "fresh" or any(p.is_relative_to(origin) or origin.is_relative_to(p) for p in (work, output, bundle)):
            parser.error("editable source needs fresh initialization and separate bundle, source, work and output paths")
        editable = json.loads(args.source_identity_file.read_bytes())
        source_sha = edit.validate_pin(editable, contract.ROOT, SOURCE_SHA)
    from v2_gpu128_a_bundle_local import open_bundle
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output.mkdir(exist_ok=bool(args.resume_sha256))
    signature = output/"experiment.json"
    with _locked(output):
        previous = json.loads(signature.read_bytes()) if args.resume_sha256 else None
        if previous is not None:
            edit.require_resume_source(previous, source_sha)
        cfg = run_config(epochs, args.seed, args.run_gib)
        with open_bundle(bundle, work, seconds=args.seconds, device=args.device,
                         full_cache=output/"frame-cache", restore_models=args.initialize == "best",
                         full_cache_headroom=(args.run_gib+3)*GIB, editable_manifest=editable) as session:
            identity = experiment(cfg, session, args.initialize, editable)
            training = LocalTraining(cfg, session, output, identity, args.initialize, lambda: None)
            if args.resume_sha256:
                if previous != identity:
                    raise ValueError("portable_full_run_requires_original_settings")
                if not training.root.is_dir():
                    raise ValueError("portable_full_run_checkpoint_missing")
            else:
                training.root.mkdir()
                _atomic(signature, canonical(identity))
            report = training.run(resume_sha256=args.resume_sha256, stop_after_updates=args.stop_after_updates)
            print(canonical({"status": report["status"], "planned_updates": training.total, "cursor": report["cursor"],
                             "last": training.last, "experiment": identity, "result": str(training.root)}).decode(), flush=True)


if __name__ == "__main__":
    main()

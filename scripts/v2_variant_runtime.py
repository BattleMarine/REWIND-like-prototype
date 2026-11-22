"""Read B/C weights and official evidence while using only A's verified common inputs."""

import argparse
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from rewind_archive.contracts import digest
from rewind_v1 import v2_contract as contract, v2_retention as retention
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_gpu_training import identity_for, load_inputs, training_config
from scripts.v2_gpu_training_comparison import shared_clips
from scripts.v2_gpu_training_pipeline import FullData, Training

if __package__:
    from . import v2_gpu128_a_bundle_format as fmt
    from . import v2_gpu128_a_bundle_local as common
else:
    import v2_gpu128_a_bundle_format as fmt
    import v2_gpu128_a_bundle_local as common


BASE_INDEX_SHA256 = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
VARIANT_SCHEMA = "rewind_v2_gpu128_variant_v1"
ROLES = {"frontend", "body", "hand"}
FINAL_CURSOR = {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}
EVIDENCE = {
    "B": {"seed": 71, "diagnostic.json": "31b76e298140aed080f135755f3a09f6f4a95d420d470d5224e6707b5314ae43",
          "comparison.json": "b153fc42f1e3f0f628ade52fc44e0a45794dd0f13e71b6b089feacb0023e46c0",
          "step": "a006e4d796bec1fdc3e94045ac75e538d8e7ad358fb91c31f779a95ea9424613"},
    "C": {"seed": 103, "diagnostic.json": "e786a21d9c00f384df9eb6389d405103e2ae30882dee6002014ec7dc2f40a83b",
          "comparison.json": "4b2f404d1b02ceb75e280172b072c9f2a6fefa2b4502c97f7f2a8d67536297ad",
          "step": "21fef0af03e3a12a2f4d070abca226a8e81fb145e9113c7539f3b1fd3aa079c6"},
}


def base_inputs(view):
    raw = view.read(fmt.INDEX, 2*fmt.MIB)
    if fmt.sha(raw) != BASE_INDEX_SHA256:
        raise ValueError("variant_base_A_index_sha256")
    index = json.loads(raw)
    fmt.check_json_pin(index, fmt.SCHEMA)
    pin = index["members"][fmt.SOURCE_RECEIPT]
    if pin["sha256"] != fmt.SOURCE_RECEIPT_SHA:
        raise ValueError("variant_base_source_receipt_sha256")
    return index, json.loads(view.read(fmt.SOURCE_RECEIPT, 64*1024, pin))["pins"]


def selected_evidence(view, pins):
    manifest = json.loads(view.read("variant.json", 128*1024))
    fmt.check_json_pin(manifest, VARIANT_SCHEMA)
    slot, members = manifest["slot"], manifest["members"]
    if slot not in EVIDENCE or type(manifest["seed"]) is not int or manifest["seed"] != EVIDENCE[slot]["seed"]:
        raise ValueError("variant_slot_seed")
    expected = EVIDENCE[slot]

    def read(name, schema, cap):
        p = members[name]
        if p["sha256"] != expected[name]:
            raise ValueError(f"variant_original_evidence_sha256: {name}")
        doc = json.loads(view.read(name, cap, p))
        fmt.check_json_pin(doc, schema)
        return doc

    diagnostic = read("diagnostic.json", "rewind_v2_gpu_invocation_v1", 192*1024)
    comparison = read("comparison.json", "rewind_v2_gpu_comparison_v1", 64*1024)
    step = diagnostic["checkpoint"]
    fmt.check_json_pin(step, "rewind_v2_gpu_training_checkpoint_v1", expected["step"])
    identity = identity_for(slot, pins)
    if (manifest["seed"] != identity["seed"] or diagnostic["identity"] != identity or diagnostic["slot"] != slot
            or diagnostic["status"] != "completed" or step["identity"] != identity or step["completed"] is not True
            or step["cursor"] != FINAL_CURSOR or diagnostic["comparison"] != comparison
            or comparison["binding"]["experiment"] != identity or comparison["binding"]["eval_seed"] != identity["eval_seed"]
            or set(comparison["binding"]["model"]) != ROLES):
        raise ValueError("variant_final_identity_or_comparison")
    for name, p in (("checkpoint.zip", step["archive"]), ("comparison.zip", comparison["archive"])):
        if members[name] != {key: p[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"variant_original_archive_pin: {name}")
        # Ensure the declared artifact exists with the authorized length even on an explicit fresh run.
        with view.open(name, p["bytes"]):
            pass
    return diagnostic, step, comparison


def restore_selected(view, workspace, model, step, comparison, limits, ledger):
    model.binding["environment"] = {key: value for key, value in step["environment"].items()
                                    if key not in {"device_name", "device_memory_bytes"}}
    package = step["archive"]
    local = view.copy("checkpoint.zip", workspace/"trained-original.zip", package, ledger.check)
    last = retention.restore_run(local, package, model.root, limits, ledger, model.binding)
    local.unlink()
    if last != step["last"]["sha256"]:
        raise ValueError("variant_last_sha256")
    model._resume(last)
    if model.last != step["last"] or any(model.progress[k] != v for k, v in step["cursor"].items()):
        raise ValueError("variant_final_cursor")
    if {role: pin["sha256"] for role, pin in model.progress["best"].items()} != comparison["binding"]["model"]:
        raise ValueError("variant_selected_model_scope")
    return view.copy("comparison.zip", workspace/"official-comparison.zip", comparison["archive"], ledger.check)


@contextmanager
def open_variant_bundle(bundle, workspace, *, base, seconds, device, full_cache=None, restore_models=True,
                        full_cache_headroom=6*fmt.GIB, editable_manifest=None):
    bundle, base, workspace = Path(bundle).resolve(strict=True), Path(base).resolve(strict=True), Path(workspace).absolute()
    if (workspace.exists() or workspace.is_symlink() or not workspace.parent.is_dir()
            or any(workspace.resolve().is_relative_to(root) or root.is_relative_to(workspace.resolve()) for root in (bundle, base))
            or shutil.disk_usage(workspace.parent).free < 4*fmt.GIB):
        raise ValueError("variant_new_separate_workspace_with_4GiB_free")
    if type(seconds) is not int or not 1 <= seconds <= 28800 or device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("variant_runtime_time_or_device")
    if editable_manifest is not None and restore_models:
        raise ValueError("variant_editable_source_requires_fresh")
    torch.set_num_threads(1)
    with fmt.BundleView(base) as base_view, fmt.BundleView(bundle) as variant_view:
        index, source_pins = base_inputs(base_view)
        diagnostic, step, comparison = selected_evidence(variant_view, source_pins)
        identity = step["identity"]
        workspace.mkdir()
        consumer = source_pins if editable_manifest is None else {**source_pins, "training_core_sha256": digest(contract.source())}
        doc, spec, plan = load_inputs(SimpleNamespace(invocation=workspace, workspace=workspace, slot=identity["slot"]), consumer)
        if shared_clips(plan) != {k: tuple(v) for k, v in comparison["binding"]["clips"].items()}:
            raise ValueError("variant_official_clips")
        cfg = training_config(identity["seed"])
        limits = {**cfg, "seconds_per_invocation": seconds, "external_timeout_seconds": seconds,
                  "terminal_reserve_seconds": 0, "copy_seconds": min(seconds, 600)}
        ledger = Ledger(workspace/"usage", identity, limits)
        with ledger.invocation("portable_v2_128_variant"):
            journal, anchor, entries = common.pinned_journal(base_view, index, source_pins, doc, spec, ledger, workspace)
            cache_root = workspace/"cache" if full_cache is None else full_cache
            cap = common.CACHE_SCRATCH_CAP
            if full_cache is not None:
                if full_cache.is_symlink() or any(full_cache.resolve().is_relative_to(p) for p in (bundle, base)):
                    raise ValueError("variant_full_cache_separate_from_bundles")
                missing = sum(e["package"]["bytes"] for e in entries if not (full_cache/f"shard-{e['index']:05d}").is_dir())
                if shutil.disk_usage(full_cache.parent).free < missing+full_cache_headroom:
                    raise OSError(f"variant_full_cache_space: need {missing+full_cache_headroom} free bytes")
                cap = max(cap, index["cache_package_bytes"])
            images = common.BundleCache(cache_root, journal, anchor, entries, spec, cfg, ledger, base_view, cap=cap)
            try:
                inputs = FullData(doc, spec, cfg, plan, images, ledger, fmt.SOURCE_SHA)
                model = Training(cfg, inputs, workspace/"features", torch.device("cuda:0" if device == "cuda" else "cpu"),
                                 workspace/"run", workspace, ledger, lambda: None, identity)
                comparison_path = restore_selected(variant_view, workspace, model, step, comparison, limits, ledger) if restore_models else None
                yield SimpleNamespace(pipeline=model, ledger=ledger, diagnostic=diagnostic, comparison=comparison,
                                      bundle=bundle, root=workspace, comparison_path=comparison_path)
            finally:
                images.close()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("command", choices=("predict", "tune", "train-full"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args, tail = parser.parse_known_args(argv)
    base = args.base.resolve(strict=True)
    for flag in ("--work", "--output"):
        for i, value in enumerate(tail):
            name, equals, path = value.partition("=")
            if name == flag:
                picked = Path(path if equals else tail[i+1]).resolve()
                if picked.is_relative_to(base) or base.is_relative_to(picked):
                    raise ValueError(f"variant_{flag[2:]}_separate_from_base")
    common.open_bundle = partial(open_variant_bundle, base=base)  # This worker is isolated; A's files are never changed.
    if args.command == "predict":
        from v2_gpu128_a_bundle_predict import main as invoke
    elif args.command == "tune":
        from v2_gpu128_a_bundle_tune import main as invoke
    else:
        from v2_gpu128_a_bundle_train_full import main as invoke
    invoke(["--bundle", str(args.bundle), *tail])


if __name__ == "__main__":
    main()

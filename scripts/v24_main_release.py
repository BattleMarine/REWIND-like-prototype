"""Build the default-off v2.4 main candidate against locally recovered pilot evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from rewind_archive.contracts import canonical, digest
from scripts.v210_tail_build import cell
from scripts.v24_main_plan import R2_RECEIPT_SHA, from_pilot
from scripts.v24_main_run import (BOOTSTRAP_AND_PUBLICATION_CAP, BOUNDARIES, GIB, LEARNING_RATE, MIB,
                                  PUBLISH_RESERVE, READ_CAP, REPORT_CAP, SCRATCH_FREE, SECONDS, SEEDS,
                                  STATE_CAP, TRAJECTORY, WORKER_RESERVE, WRITE_CAP)


ROOT = Path(__file__).resolve().parents[1]
PILOT_PLAN_SHA = "d2976020b429550e2868baac65ef3b22db3ee3b97044cfef0e4d462d386b9681"
PILOT_RESULT_SHA = "c2c0f786ffd4032c98c50e1a0b557877bdfa82dde485c00419a897e616670c6d"
NOTEBOOK = "v24_main128_A_paired_candidate_metrics_r2.ipynb"
TOOLKIT = "v24_main128_A_tools_metrics_r2.zip"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def script(name):
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def budget(pins):
    groups = sorted((item["archive"]["bytes"] for item in pins["groups"].values()), reverse=True)
    accesses = 53  # 34 train groups, 18 full-validation groups, one initialization probe
    gross_feature_zip_bound = accesses * groups[0]  # includes repeated groups
    parent = pins["write_budget"]["worker_reservations_bytes"]
    write = (parent["parent_archive_copy"] + parent["parent_extraction"] + pins["base"]["manifest_bytes"]
             + 2*STATE_CAP + gross_feature_zip_bound + accesses*32*MIB)
    if write > WRITE_CAP-BOOTSTRAP_AND_PUBLICATION_CAP:
        raise ValueError("Feature and state reservations exceed the candidate worker cap")
    return {"seconds": SECONDS, "worker_reserve_seconds": WORKER_RESERVE, "publish_reserve_seconds": PUBLISH_RESERVE,
            "worker_logical_read_bytes": READ_CAP, "all_logical_write_bytes": WRITE_CAP,
            "worker_logical_write_bytes": WRITE_CAP-BOOTSTRAP_AND_PUBLICATION_CAP,
            "worker_plan_reservation_bytes": write, "bootstrap_and_publication_bytes": BOOTSTRAP_AND_PUBLICATION_CAP,
            "max_group_accesses": accesses, "max_feature_zip_payload_bytes": gross_feature_zip_bound,
            "state_bytes": STATE_CAP, "report_bytes": REPORT_CAP, "scratch_free_bytes": SCRATCH_FREE,
            "worker_read_evidence": "R2 includes parent restore and 30 or fewer feature groups, main adds feature and resume copy",
            "drive_trial_state_bytes_max": 4*STATE_CAP}


def build(source, pilot, r2, output, gpu=None):
    r2_raw = (r2 / "receipt.json").read_bytes()
    raw = (pilot / "main-plan.json").read_bytes()
    if sha(r2_raw) != R2_RECEIPT_SHA or sha(raw) != PILOT_PLAN_SHA or sha((pilot / "result.json").read_bytes()) != PILOT_RESULT_SHA:
        raise ValueError("Expected the verified R2 release and recovered pilot plan/result")
    measured = from_pilot(pilot.resolve(), r2 / "receipt.json", PILOT_RESULT_SHA)["pilot"]
    prior = json.loads(r2_raw)["pins"]
    if source.name != prior["base"]["source_name"] or sha(source.read_bytes()) != prior["base"]["source_sha256"]:
        raise ValueError("Fixture source ZIP mismatch")
    previous_kit = r2 / prior["toolkit"]["name"]
    if sha(previous_kit.read_bytes()) != prior["toolkit"]["sha256"]:
        raise ValueError("Fixed R2 helper ZIP mismatch")
    with zipfile.ZipFile(previous_kit) as archive:
        modules = {name: archive.read(name) for name in prior["modules"]}
    if {name: sha(data) for name, data in modules.items()} != prior["modules"]:
        raise ValueError("Fixed R2 module mismatch")
    for name in ("v24_body.py", "v24_main_run.py", "v210_metrics.py"):
        modules[name] = script(name).encode()
        compile(modules[name], name, "exec")
    binding = json.loads((pilot / "pilot-plan.json").read_bytes())["binding"]
    protocol = {"schema": "rewind_v24_main_candidate_v1", "status": "awaiting_approval",
                "pilot_plan_sha256": PILOT_PLAN_SHA, "pilot_result_sha256": PILOT_RESULT_SHA,
                "source_sha256": prior["base"]["source_sha256"], "parent_evidence": prior["evidence"],
                "A_checkpoint_receipt_sha256": binding["A_checkpoint_receipt"], "A_weights": binding["A_weights"],
                "feature_identity": binding["feature_identity"], "feature_group_pins_sha256": digest(prior["groups"]),
                "manifest_file_sha256": prior["base"]["manifest_file_sha256"], "max_train_frames": 119586,
                "train_clips": 3097, "validation_clips": 421, "validation_frames": 16425,
                "pilot_validation_keys": binding["validation_keys"], "boundaries": list(BOUNDARIES),
                "interim_validation_updates": list(BOUNDARIES[:-1]), "official_validation_update": BOUNDARIES[-1],
                "initial_seeds": list(SEEDS[:3]), "conditional_seeds": list(SEEDS[3:]), "trajectory": TRAJECTORY,
                "runtime": {key: measured["environment"][key] for key in ("gpu", "precision", "tf32", "deterministic_algorithms")},
                "training": {"learning_rate": LEARNING_RATE, "optimizer": "Adam", "batch": 1, "passes": 1,
                             "loss": "common masked x0 position and x0 frame loss", "eval_seed": 71, "ddim_steps": 10},
                "budgets": budget(prior), "measured_pilot": measured,
                "conditional_extension_proposal": {"min_v_lower_pairs": 2, "min_abs_mean_delta_mm": 3,
                    "max_mean_hand_global_regression_mm": 0, "requires_separate_approval": True},
                "convergence_proposal": {"fixed_final_boundary_selection": True,
                    "flag_if_both_last_subset_intervals_improve": True, "automatic_training_extension": False}}
    if gpu is not None:
        protocol["runtime"]["gpu"] = gpu
    protocol_sha = digest(protocol)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v24-main-") as tmp:
        staging = Path(tmp) / "release"
        staging.mkdir()
        with zipfile.ZipFile(staging / TOOLKIT, "x") as archive:
            for name, data in sorted(modules.items()):
                archive.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
        kit = (staging / TOOLKIT).read_bytes()
        pins = {key: prior[key] for key in ("base", "evidence", "groups")}
        pins.update(protocol=protocol, protocol_sha256=protocol_sha, source_upload_bytes=source.stat().st_size,
                    modules={name: sha(data) for name, data in modules.items()},
                    toolkit={"name": TOOLKIT, "bytes": len(kit), "sha256": sha(kit)})
        settings = '''RUN_V24_MAIN = False
APPROVED_PROTOCOL_SHA256 = ""
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
ARM = "x0"
SEED = 42
START_UPDATE = 0
STOP_UPDATE = 768
RESUME_RESULT = ""
RESUME_RESULT_SHA256 = ""
APPROVED_EXTENSION = False
'''
        support = script("v210_tail_colab.py").split("\ndef run_tail_notebook", 1)[0]
        transfer = script("v24_colab.py").split("\ndef run_v24_notebook", 1)[0]
        code = "\n".join((script("archvis_pilot_bootstrap.py"), support, transfer, script("v24_main_colab.py"))) + f'''
PINS = {pins!r}
if RUN_V24_MAIN:
    run_v24_main(DRIVE_PROJECT, ARM, SEED, START_UPDATE, STOP_UPDATE, RESUME_RESULT,
                 RESUME_RESULT_SHA256, APPROVED_PROTOCOL_SHA256, APPROVED_EXTENSION, PINS)
else:
    print("v2.4 본시험 실행 승인 대기")
'''
        compile(settings, "v24 main settings", "exec")
        compile(code, "v24 main controller", "exec")
        intro = "# v2.4 본시험 후보\n\n승인 전에는 RUN_V24_MAIN을 켜지 않습니다. 구간 1회씩 실행합니다.\n"
        doc = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings), cell("code", "run", code)]})
        files = {NOTEBOOK: doc, TOOLKIT: kit, "candidate-protocol.json": canonical(protocol),
                 "GUIDE_KO.md": (ROOT / "docs/V24_DIFFUSION_MAIN_CANDIDATE.md").read_bytes()}
        if gpu:
            files["GUIDE_KO.md"] = (f"# v2.4 {gpu} 병렬 비교\n\n"
                                   f"이 발급본은 두 arm 모두 {gpu} FP32에서 실행한다. "
                                   "아래 L4 시간은 이전 파일럿 실측이다.\n\n").encode() + files["GUIDE_KO.md"]
            files[NOTEBOOK.replace("candidate", "t4")] = files.pop(NOTEBOOK)
        for name, data in files.items():
            if name != TOOLKIT:
                (staging / name).write_bytes(data)
        receipt = {"schema": "rewind_v24_main_candidate_release_v1", "status": "awaiting_approval", "default_off": True,
                   "protocol_sha256": protocol_sha, "pins": pins,
                   "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}}
        (staging / "receipt.json").write_bytes(canonical(receipt))
        staging.rename(output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "pilot", "r2", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu", choices=["Tesla T4"], help="Override the measured pilot GPU for a matched pair")
    print(json.dumps(build(**vars(parser.parse_args()))["files"]))

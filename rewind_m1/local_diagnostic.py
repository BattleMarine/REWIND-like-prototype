"""기존 M1 모듈을 연결하는 소량 CPU 진단 runner. Runner 재개는 미지원."""
import argparse
import copy
import hashlib
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .data import DataContractError, content_hash
from .evaluate_2d import evaluate_cache, validate_groups
from .frontend_train import run_training, validate_config as validate_frontend
from .local_rgb import LocalStereoRGB
from .local_rgb_predictions import load_weights, build_cache
from .prediction_join import MAX_BYTES, _read_file, _serialized, join_predictions
from .rgb_supervision import LocalRGBSupervision
from .training import Trainer, save_checkpoint, source_identity, validate_config


DEADLINE_SECONDS = 600
MAX_TENSOR_BYTES = 64 * 1024**2
MAX_CHECKPOINT_BYTES = 32 * 1024**2
MAX_ARTIFACT_BYTES = 60 * 1024**2  # 최상위 report 4MiB를 따로 예약한다.


class StateRestoreError(RuntimeError):
    pass


@contextmanager
def _preserve_state():
    python_rng, cpu_rng = random.getstate(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        # 한 항목이 실패해도 나머지 복원은 시도한다.
        failures = []
        actions = [lambda: random.setstate(python_rng), lambda: torch.set_rng_state(cpu_rng),
            lambda: torch.set_num_threads(threads), lambda: torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)]
        if cuda_rng is not None:
            actions.append(lambda: torch.cuda.set_rng_state_all(cuda_rng))
        for action in actions:
            try:
                action()
            except BaseException as exc:
                failures.append(type(exc).__name__)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        if failures:
            raise StateRestoreError("runner state restoration failed")


def _inventory(output, *, partial=False):
    artifacts, total = {}, 0
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == output/"report.json":
            continue
        size = path.stat().st_size
        total += size
        maximum = MAX_CHECKPOINT_BYTES if path.name == "last.pt" else MAX_BYTES
        if not partial and (size > maximum or total > MAX_ARTIFACT_BYTES):
            raise DataContractError("runner 산출물 byte 상한 초과")
        digest = None
        if size <= maximum and total <= MAX_ARTIFACT_BYTES:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        artifacts[path.relative_to(output).as_posix()] = {"size_bytes":size, "sha256":digest}
    return artifacts


def run_diagnostic(loader, pose_doc, joint_names, expected_manifest_sha256, frontend_config,
                   regression_config, groups, output, *, regression_steps=2):
    pose_doc, frontend_config, regression_config, groups, joint_names = copy.deepcopy(
        (pose_doc, frontend_config, regression_config, groups, joint_names))
    validate_frontend(frontend_config)
    config = validate_config(regression_config)
    if type(regression_steps) is not int or not 1 <= regression_steps <= 100:
        raise DataContractError("regression_steps는 1~100 필요")
    if config["amp"] or config["window_size"] > 20 or config["hidden"] > 128 or config["batch_size"] > 8:
        raise DataContractError("CPU 진단 regression config 상한")
    common = dict(expected_manifest_sha256=expected_manifest_sha256, joint_names=joint_names)
    LocalRGBSupervision(loader, pose_doc, split="train", **common)
    LocalRGBSupervision(loader, pose_doc, split="validation", **common)
    validate_groups(groups, joint_names)
    tensor_bytes = len(pose_doc["records"]) * config["window_size"] * (len(joint_names)*69+1)
    if tensor_bytes > MAX_TENSOR_BYTES:
        raise DataContractError("regression tensor 추정 byte 상한")
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"schema_version":"local_m1_diagnostic_v1", "status":"running", "runner_resume_supported":False,
        "input":{"rgb_manifest_sha256":expected_manifest_sha256, "pose_sha256":content_hash(pose_doc),
                 "frontend_config_sha256":content_hash(frontend_config), "regression_config_sha256":content_hash(config), "groups_sha256":content_hash(groups)},
        "frontend_config":frontend_config, "regression_config":config, "regression_steps":regression_steps,
        "groups":groups, "source":None, "stages":[], "artifacts":{}, "partial_artifacts":{}}
    stage = "initialization"

    def persist():
        raw = _serialized(report)
        with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(raw)
            except BaseException:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, output/"report.json")
        finally:
            temporary.unlink(missing_ok=True)

    def deadline():
        if time.monotonic()-started > DEADLINE_SECONDS:
            raise DataContractError("runner cooperative deadline 초과")

    def write_json(document, path):
        raw = _serialized(document)
        with (output/path).open("xb") as handle:
            handle.write(raw)

    def execute(name, action):
        nonlocal stage
        stage = name
        report["active_stage"] = stage
        persist()
        deadline()
        begin = time.monotonic()
        result = action()
        deadline()
        artifacts = _inventory(output)
        report["stages"].append({"name":name, "status":"completed", "elapsed_seconds":time.monotonic()-begin})
        report["artifacts"] = artifacts
        persist()
        return result

    try:
        report["source"] = source_identity()
        persist()
        with _preserve_state():
            execute("frontend", lambda: run_training(loader, pose_doc, joint_names, expected_manifest_sha256,
                                                      frontend_config, output/"frontend"))

            def predict():
                model, checkpoint = load_weights(output/"frontend/frontend_weights.pt", loader, joint_names)
                cache = build_cache(loader, model, checkpoint, joint_names, frontend_config["batch_size"])
                write_json(cache, "predictions.json")
                return cache
            cache = execute("prediction_cache", predict)

            def evaluate():
                metrics = evaluate_cache(pose_doc, cache, expected_manifest_sha256, split="validation", groups=groups)
                write_json(metrics, "evaluation_2d.json")
            execute("evaluation_2d", evaluate)

            def join():
                document = join_predictions(pose_doc, cache, expected_manifest_sha256)
                write_json(document, "predicted_features.json")
                return document
            predicted = execute("prediction_join", join)

            def regress():
                trainer = Trainer(config, predicted, device="cpu")
                losses = []
                for _ in range(regression_steps):
                    deadline()
                    losses.append(trainer.train_step())
                    deadline()
                metrics = trainer.evaluate("validation")
                deadline()
                save_checkpoint(trainer.checkpoint(), output/"regression/last.pt")
                write_json({"step":trainer.step, "losses":losses, "validation":metrics,
                    "manifest_hash":trainer.manifest_hash, "source":trainer.source, "environment":trainer.environment}, "regression/report.json")
            execute("regression", regress)
            stage = "restore_state"
        stage = "final_report"
        deadline()
        report.update(status="completed", active_stage=None, elapsed_seconds=time.monotonic()-started,
            metrics={"2d":"evaluation_2d.json", "3d":"regression/report.json"})
        persist()
        return report
    except BaseException as exc:
        report.update(status="failed", active_stage=None, failed_stage="restore_state" if isinstance(exc, StateRestoreError) else stage,
                      error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
        try:
            inventory = _inventory(output, partial=True)
            report["partial_artifacts"] = {k:v for k,v in inventory.items() if k not in report["artifacts"]}
            persist()
        except BaseException:
            pass  # 보고서를 쓸 수 없으면 기존 partial 파일이 근거다. 완료로 간주하지 않는다.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "root", "features", "frontend-config", "regression-config", "groups", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--regression-steps", type=int, default=2)
    args = parser.parse_args(argv)
    run_diagnostic(LocalStereoRGB(args.manifest, args.root), _read_file(args.features), args.joints, args.manifest_sha256,
        _read_file(args.frontend_config), _read_file(args.regression_config), _read_file(args.groups), args.output,
        regression_steps=args.regression_steps)


if __name__ == "__main__":
    main()

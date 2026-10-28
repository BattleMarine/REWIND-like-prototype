"""작은 CPU 진단 학습과 inference weights export. 재개 기능은 제공하지 않는다."""
import argparse
import math
import platform
import time
from pathlib import Path

import torch
from PIL import __version__ as pillow_version

from .data import DataContractError, content_hash
from .frontend import StereoHeatmapNet, heatmap_loss, save_frontend_weights, weights_hash
from .local_rgb import LocalStereoRGB
from .prediction_join import MAX_BYTES, _read_file, _serialized
from .rgb_supervision import LocalRGBSupervision
from .training import source_identity


MAX_HEATMAP_ELEMENTS = 32 * 1024**2
DEADLINE_SECONDS = 120


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {"seed", "steps", "batch_size", "learning_rate", "frontend_version"}:
        raise DataContractError("frontend config 필드 오류")
    for key, low, high in (("seed", 0, 2**32-1), ("steps", 1, 100), ("batch_size", 1, 4)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise DataContractError(f"frontend config 범위: {key}")
    lr = config["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= .1:
        raise DataContractError("frontend learning_rate 범위")
    version = config["frontend_version"]
    if not isinstance(version, str) or not version.strip() or len(version.encode()) > 128:
        raise DataContractError("frontend_version 필요")


def _inside(batch, size):
    xy = batch["gt_xy"]
    return batch["annotation_valid"] & (xy[..., 0] >= 0) & (xy[..., 0] <= size[0]-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= size[1]-1)


def run_training(loader, pose_doc, joint_names, expected_manifest_sha256, config, output):
    validate_config(config)
    config = dict(config)
    common = dict(expected_manifest_sha256=expected_manifest_sha256, joint_names=joint_names)
    train = LocalRGBSupervision(loader, pose_doc, split="train", **common)
    validation = LocalRGBSupervision(loader, pose_doc, split="validation", **common) if any(r["split"] == "validation" for r in pose_doc["records"]) else None
    size = train.metadata(0)["image_sizes"][0]
    if max(size) > 1024 or len(joint_names) > 256:
        raise DataContractError("진단 영상/관절 상한 초과")
    elements = config["batch_size"] * 2 * len(joint_names) * ((size[1]+1)//2) * ((size[0]+1)//2)
    if elements > MAX_HEATMAP_ELEMENTS:
        raise DataContractError("heatmap element 상한 초과")
    source = source_identity()
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    weights_path, report_path = output/"frontend_weights.pt", output/"report.json"
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > DEADLINE_SECONDS:
            raise DataContractError("CPU 진단 cooperative deadline 초과")

    try:
        # CPU RNG만 격리한다. Thread/default dtype/device 설정은 변경하지 않는다.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            model = StereoHeatmapNet(len(joint_names)).to(device="cpu", dtype=torch.float32)
            initial = weights_hash(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            history, position = [], 0
            model.train()
            for step in range(config["steps"]):
                deadline()
                indices = list(range(position, min(position+config["batch_size"], len(train))))
                batch = train.load_batch(indices)
                optimizer.zero_grad(set_to_none=True)
                loss, inside = heatmap_loss(model(batch["images"])[0], batch["gt_xy"], batch["annotation_valid"], size)
                if not torch.isfinite(loss):
                    raise DataContractError("frontend loss 비유한 값")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise DataContractError("frontend gradient 비유한 값")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise DataContractError("frontend weights 비유한 값")
                history.append({"step":step+1, "loss":float(loss.detach()), "inside_joints":int(inside.sum()),
                    "keys":[[m["pose_record"]["sequence_id"], m["pose_record"]["frame_index"]] for m in batch["metadata"]]})
                position = (indices[-1]+1) % len(train)
                deadline()
            fingerprint = weights_hash(model)
            evaluation = {"status":"not_available", "loss":None, "records":0, "inside_joints":0, "zero_inside_batches":0}
            if validation is not None:
                total, count, skipped = 0., 0, 0
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(validation), config["batch_size"]):
                        deadline()
                        batch = validation.load_batch(list(range(start, min(start+config["batch_size"], len(validation)))))
                        n = int(_inside(batch, size).sum())
                        if n:
                            loss, _ = heatmap_loss(model(batch["images"])[0], batch["gt_xy"], batch["annotation_valid"], size)
                            if not torch.isfinite(loss):
                                raise DataContractError("validation loss 비유한 값")
                            total += float(loss) * n
                            count += n
                        else:
                            skipped += 1
                        deadline()
                evaluation = {"status":"evaluated" if count else "not_evaluable", "loss":total/count if count else None,
                    "records":len(validation), "inside_joints":count, "zero_inside_batches":skipped}
            if weights_hash(model) != fingerprint:
                raise DataContractError("validation weights 변경")
            meta = {"joint_names":joint_names, "image_sizes":[size, size], "data_kind":train.metadata(0)["rgb"]["data_kind"],
                "frontend_version":config["frontend_version"], "frontend_checkpoint_sha256":fingerprint,
                "preprocessing_version":"rgb_0_1_v1", "manifest_hash":expected_manifest_sha256}
            save_frontend_weights(model, meta, weights_path)
            if weights_path.stat().st_size > MAX_BYTES:
                raise DataContractError("weights byte 상한 초과")
            deadline()
            report = {"schema_version":"local_frontend_diagnostic_v1", "evaluation_kind":"diagnostic_code_check",
                "data_kind":meta["data_kind"], "joint_names":joint_names, "image_sizes":meta["image_sizes"], "preprocessing_version":"rgb_0_1_v1",
                "config":config, "resume_supported":False, "scheduler":None, "scaler":None, "batch_order":"sequential_no_shuffle",
                "input":{ "rgb_manifest_sha256":expected_manifest_sha256, "pose_document_sha256":content_hash(pose_doc),
                    "train_keys":[[train.metadata(i)["pose_record"]["sequence_id"], train.metadata(i)["pose_record"]["frame_index"]] for i in range(len(train))]},
                "source":source, "environment":{"python":platform.python_version(), "torch":str(torch.__version__), "pillow":pillow_version,
                    "cpu_threads":torch.get_num_threads(), "device":"cpu", "precision":"float32"},
                "initial_weights_sha256":initial, "final_weights_sha256":fingerprint, "training":history, "validation":evaluation,
                "weights_size_bytes":weights_path.stat().st_size, "elapsed_seconds":time.monotonic()-started}
            raw = _serialized(report)
            deadline()
            with report_path.open("xb") as handle:
                handle.write(raw)
            return report
    except BaseException:
        weights_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        output.rmdir()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_training(LocalStereoRGB(args.manifest, args.root), _read_file(args.features), args.joints,
                 args.manifest_sha256, _read_file(args.config), args.output)


if __name__ == "__main__":
    main()

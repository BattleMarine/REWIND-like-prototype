"""작은 step 기반 학습·평가와 완전한 로컬 checkpoint 재개."""

import copy
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import tempfile
from pathlib import Path

import torch

from .data import content_hash, load_document, synthetic_document
from .metrics import masked_mse, metric_sums
from .models import make_model

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    required = {"model", "window_size", "hidden", "dropout", "batch_size", "learning_rate", "seed", "target_scale_mm", "amp", "scheduler_step", "scheduler_gamma"}
    if set(config) != required:
        raise ValueError(f"Config key 불일치: {sorted(set(config) ^ required)}")
    for key in ("window_size", "hidden", "batch_size", "scheduler_step"):
        if type(config[key]) is not int or config[key] < 1: raise ValueError(f"{key}: 양의 정수 필요")
    if type(config["seed"]) is not int or config["seed"] < 0 or type(config["amp"]) is not bool:
        raise ValueError("seed/amp type 불일치")
    for key in ("dropout", "learning_rate", "target_scale_mm", "scheduler_gamma"):
        value = config[key]
        if type(value) not in (int,float) or not 0 <= value < float("inf"): raise ValueError(f"{key}: 유한한 양수/0 필요")
    if not 0 <= config["dropout"] < 1 or not 0 < config["scheduler_gamma"] <= 1 or config["learning_rate"] <= 0 or config["target_scale_mm"] <= 0:
        raise ValueError("Config 범위 오류")
    if config["model"] not in ("single_frame", "temporal"):
        raise ValueError("지원하지 않는 model")
    return copy.deepcopy(config)


def require_torch_api():
    """필요 API의 최소 버전만 검사. 버전별 실행 검증을 대신하지 않는다."""
    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if not match or tuple(map(int, match.groups())) < (2, 3) or not callable(getattr(getattr(torch, "amp", None), "GradScaler", None)):
        raise ValueError("PyTorch >=2.3 및 torch.amp.GradScaler 필요. 기존 GPU 환경은 자동 교체하지 않습니다.")


def source_identity():
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("rewind_m1", "rewind_data") for p in sorted((ROOT/package).glob("*.py"))}
    revision, dirty = "unversioned", None
    stamp = ROOT / "SOURCE_REVISION.txt"
    if stamp.exists():
        revision = stamp.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision): raise ValueError("SOURCE_REVISION.txt에는 commit SHA 40자리 필요")
    elif (ROOT / ".git").exists():
        try:
            revision = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
            dirty = bool(subprocess.run(["git","status","--porcelain"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip())
        except (OSError,subprocess.CalledProcessError): pass
    return {"revision": revision, "dirty": dirty, "code_sha256": hashes}


def seed_run(seed):
    # CUDA deterministic GEMM 요구. CUDA 연산을 시작하기 전에 설정한다.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]: torch.cuda.set_rng_state_all(state["torch_cuda"])


def read_checkpoint(path):
    # 로컬에서 생성한 checkpoint만 사용. 임의 pickle class를 허용하지 않는다.
    doc = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(doc, dict) or doc.get("format_version") != "m1_checkpoint_v1":
        raise ValueError("지원하지 않는 checkpoint")
    return doc


def save_checkpoint(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(doc, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


class Trainer:
    def __init__(self, config, document=None, device="cpu"):
        require_torch_api()
        self.config = validate_config(config)
        self.document = copy.deepcopy(document if document is not None else synthetic_document())
        self.contract, self.splits, self.manifest_hash = load_document(self.document, config["window_size"])
        if not {"train","validation"}.issubset(self.splits): raise ValueError("학습에는 분리된 train/validation 필요")
        if any(not split.target_valid[:, -1].any(dim=1).all() for split in self.splits.values()):
            raise ValueError("유효 GT가 전부 누락된 anchor를 학습/evaluation sample로 사용하지 않음")
        self.device = torch.device(device)
        if self.device.type not in ("cpu","cuda") or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("요청 device가 사용 가능하지 않음")
        if config["amp"] and self.device.type != "cuda": raise ValueError("AMP는 이 묶음에서 CUDA에만 지원")
        if self.device.type=="cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") not in (":4096:8", ":16:8"):
            raise ValueError("CUDA deterministic cuBLAS workspace 설정 불일치")
        seed_run(config["seed"])
        self.model = make_model(config, self.splits["train"].features.shape[-1], len(self.contract.joint_names)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config["scheduler_step"], config["scheduler_gamma"])
        self.scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        self.step = 0
        self.source = source_identity()
        self.environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(self.device),
            "cuda_runtime": torch.version.cuda, "threads": torch.get_num_threads(), "amp": config["amp"]}
        self.runtime_observation={}
        if self.device.type=="cuda":
            props=torch.cuda.get_device_properties(self.device)
            self.environment.update(gpu_name=props.name,gpu_total_memory_bytes=props.total_memory,gpu_capability=list(torch.cuda.get_device_capability(self.device)),
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
            free,total=torch.cuda.mem_get_info(self.device)
            self.runtime_observation={"gpu_free_memory_bytes_at_init":free,"gpu_memory_total_bytes_at_init":total}

    def train_step(self):
        data, cfg = self.splits["train"], self.config
        indices = torch.randint(len(data.anchors), (cfg["batch_size"],))
        x, time = data.features[indices].to(self.device), data.time_valid[indices].to(self.device)
        y, mask = data.target_mm[indices, -1].to(self.device), data.target_valid[indices, -1].to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=cfg["amp"]):
            prediction = self.model(x, time)[:, -1].float() * cfg["target_scale_mm"]
            loss = masked_mse(prediction, y, mask, cfg["target_scale_mm"])
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError("비유한 gradient; optimizer update 중단")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        return float(loss.detach())

    def evaluate(self, split="validation", final_test=False):
        if split == "test" and not final_test: raise ValueError("Test는 --final-test 명시 후 최종 평가에만 사용")
        if split not in self.splits: raise ValueError(f"{split} split 없음")
        data, total, previous_mode = self.splits[split], {"distance_sum_mm":0.,"squared_sum_mm2":0.,"valid_joints":0}, self.model.training
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(data.anchors), self.config["batch_size"]):
                end = start + self.config["batch_size"]
                prediction = self.model(data.features[start:end].to(self.device), data.time_valid[start:end].to(self.device))[:, -1] * self.config["target_scale_mm"]
                values = metric_sums(prediction, data.target_mm[start:end,-1].to(self.device), data.target_valid[start:end,-1].to(self.device))
                for k in total: total[k] += values[k]
        self.model.train(previous_mode)
        count = total["valid_joints"]
        return {"split":split, "mpjpe_mm":total["distance_sum_mm"]/count, "mse_mm2":total["squared_sum_mm2"]/(3*count),
                "normalized_mse":total["squared_sum_mm2"]/(3*count*self.config["target_scale_mm"]**2), "valid_joints":count,
                "data_kind":self.contract.data_kind, "input_source":self.contract.input_source,
                "evaluation_kind":"synthetic_code_check" if self.contract.data_kind=="synthetic_fixture" else
                    ("gt_input_diagnostic" if self.contract.input_source=="gt_2d" else "predicted_input_baseline")}

    def checkpoint(self):
        return {"format_version":"m1_checkpoint_v1", "model":self.model.state_dict(), "optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict(), "scaler":self.scaler.state_dict(), "rng":rng_state(), "step":self.step,
            "samples_seen":self.step*self.config["batch_size"], "sampler_policy":"random_with_replacement_torch_global_rng",
            "config":self.config, "manifest_hash":self.manifest_hash, "manifest":self.document["manifest"],
            "input_contract":self.document["contract"], "source":self.source, "environment":self.environment,
            "runtime_observation":self.runtime_observation}

    def resume(self, checkpoint):
        if checkpoint["config"] != self.config or checkpoint["manifest_hash"] != self.manifest_hash:
            raise ValueError("Resume config/manifest 불일치")
        if checkpoint["source"]["code_sha256"] != self.source["code_sha256"] or checkpoint["environment"] != self.environment:
            raise ValueError("Resume code/environment 불일치; 동일 환경의 정확한 재개만 지원")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = checkpoint["step"]
        if type(self.step) is not int or self.step < 0 or checkpoint["samples_seen"] != self.step*self.config["batch_size"]:
            raise ValueError("Checkpoint 학습 위치 불일치")
        restore_rng(checkpoint["rng"])

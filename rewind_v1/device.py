"""Inspect CUDA for training, or enforce A100 for latency benchmarks."""

import argparse
import json
import re
import torch


def require_cuda_training(device="cuda", *, report=print):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("rewind_v1_gpu_training_requires_cuda")
    if not torch.cuda.is_available():
        report(json.dumps({"device": str(device), "cuda_available": False}))
        raise RuntimeError("rewind_v1_cuda_required")
    index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    details = {"device": f"cuda:{index}", "gpu_name": props.name, "vram_bytes": props.total_memory}
    report(json.dumps(details))
    return details


def require_a100(device="cuda", *, report=print):
    """Enforce the A100 hardware condition for inference latency benchmarks."""
    details = require_cuda_training(device, report=report)
    if not re.search(r"\bA100\b", details["gpu_name"], flags=re.IGNORECASE):
        raise RuntimeError("rewind_v1_a100_required")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-a100", action="store_true", help="Require A100 for an inference latency benchmark")
    args = parser.parse_args()
    (require_a100 if args.benchmark_a100 else require_cuda_training)(args.device)

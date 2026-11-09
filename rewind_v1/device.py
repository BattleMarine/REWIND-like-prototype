"""Inspect the actual CUDA device before v1 GPU preparation or training."""

import argparse
import json
import re
import torch


def require_a100(device="cuda", *, report=print):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("rewind_v1_gpu_training_requires_cuda")
    if not torch.cuda.is_available():
        report(json.dumps({"device": str(device), "cuda_available": False}))
        raise RuntimeError("rewind_v1_a100_required")
    index = torch.cuda.current_device() if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    details = {"device": f"cuda:{index}", "gpu_name": props.name, "vram_bytes": props.total_memory}
    report(json.dumps(details))
    if not re.search(r"\bA100\b", props.name, flags=re.IGNORECASE):
        raise RuntimeError("rewind_v1_a100_required")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    require_a100(parser.parse_args().device)

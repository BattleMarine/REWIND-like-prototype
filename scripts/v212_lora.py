"""Unmerged LoRA on the three fused body temporal QKV weights."""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize
from rewind_v1.attention import Equation4Block
from rewind_v1.teacher import V1Body
from scripts.v211_trial import TRAINING, UPDATES, load_parameters, reset_optimizer, save_parameters, tensor_digest, trajectory_seed


METHOD = "v2.1.2"
RANK, ALPHA, LAYERS, HIDDEN = 4, 4, 3, 512
BRANCH = {"scope": "body_temporal_qkv", "adapter_layers": LAYERS, "rank": RANK, "alpha": ALPHA,
          "adapter_dropout": 0, "zero_initial_delta": True, "updates": UPDATES, **TRAINING}
ARTIFACT_PREFIX = "lora"
FIT_SCHEMA = "rewind_v212_fit_receipt_v1"
SUMMARY_SCHEMA = "rewind_v212_validation_v1"
STATE = "rewind_v212_body_temporal_qkv_fit_v1"
PROJECTIONS = tuple(f"temporal.{i}.qkv" for i in range(LAYERS))
PARAMETRIZATIONS = tuple(f"{name}.parametrizations.weight" for name in PROJECTIONS)
TRAINABLE = tuple(f"{name}.0.{factor}" for name in PARAMETRIZATIONS for factor in ("down", "up"))
ORIGINAL_WEIGHTS = {f"{name}.original": f"{projection}.weight"
                    for projection, name in zip(PROJECTIONS, PARAMETRIZATIONS, strict=True)}


class LowRankDelta(nn.Module):
    def __init__(self, device, generator):
        super().__init__()
        down = torch.empty(RANK, HIDDEN, dtype=torch.float32, device="cpu")
        nn.init.kaiming_uniform_(down, a=math.sqrt(5), generator=generator)
        self.down = nn.Parameter(down.to(device))
        self.up = nn.Parameter(torch.zeros(3*HIDDEN, RANK, dtype=torch.float32, device=device))

    def forward(self, original):
        return original + (self.up @ self.down) * (ALPHA / RANK)


def adapters(stage):
    parameters = dict(stage.model.named_parameters())
    return {name: parameters[name] for name in TRAINABLE}


def backbone_sha(stage):
    items = ((ORIGINAL_WEIGHTS.get(name, name), value)
             for name, value in stage.model.state_dict().items() if name not in TRAINABLE)
    return tensor_digest(sorted(items, key=lambda item: item[0]))


def begin_fit(stage, seed):
    model = stage.model
    if type(model) is not V1Body or len(model.temporal) != LAYERS:
        raise ValueError("v212_parent_body_temporal_layout")
    for block in model.temporal:
        if (type(block) is not Equation4Block or type(block.qkv) is not nn.Linear
                or block.qkv.weight.shape != (3*HIDDEN, HIDDEN) or block.qkv.weight.dtype != torch.float32
                or block.qkv.bias is None):
            raise ValueError("v212_parent_fused_qkv_layout")
    base_sha = backbone_sha(stage)
    model.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(trajectory_seed(seed, "lora-initialization"))
    for block in model.temporal:
        projection = block.qkv
        parametrize.register_parametrization(projection, "weight", LowRankDelta(projection.weight.device, generator))
    if (tuple(name for name, p in model.named_parameters() if p.requires_grad) != TRAINABLE
            or backbone_sha(stage) != base_sha):
        raise ValueError("v212_adapter_scope_or_base_changed")
    reset_optimizer(stage, adapters(stage).values())


def save_state(stage, binding, path, last_loss):
    save_parameters(stage, binding, path, last_loss, adapters(stage), STATE, "adapters")


def load_state(stage, binding, raw, expected_step):
    return load_parameters(stage, binding, raw, expected_step, adapters(stage), STATE, "adapters")

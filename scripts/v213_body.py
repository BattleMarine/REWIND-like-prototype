"""Full A-body FP32 fit with native Adam state and bounded Drive checkpoints."""

from rewind_v1.teacher import V1Body
from scripts.v211_trial import TRAINING, UPDATES, load_parameters, reset_optimizer, save_parameters, tensor_digest
from scripts.v2_gpu_training_storage import MIB


METHOD = "v2.1.3"
BRANCH = {"scope": "body_all", "updates": UPDATES, **TRAINING}
TRAINABLE = ("all_body_parameters",)
PARAMETER_TENSORS = 99
PARAMETER_VALUES = 15_010_822
ARTIFACT_PREFIX = "body"
FIT_SCHEMA = "rewind_v213_fit_receipt_v1"
SUMMARY_SCHEMA = "rewind_v213_validation_v1"
STATE = "rewind_v213_body_full_fit_v1"
CHECKPOINT_CAP = 192*MIB
CHECKPOINT_INTERVAL = 1550
CHECKPOINT_COPY_SECONDS = 210


def body_parameters(stage):
    parameters = dict(stage.model.named_parameters())
    if (type(stage.model) is not V1Body or len(parameters) != PARAMETER_TENSORS
            or sum(parameter.numel() for parameter in parameters.values()) != PARAMETER_VALUES):
        raise ValueError("v213_parent_body_parameter_layout")
    return parameters


def begin_fit(stage, seed):
    parameters = body_parameters(stage)
    stage.model.requires_grad_(True)
    reset_optimizer(stage, parameters.values())


def backbone_sha(stage):
    # All parameters train. Fixed buffers are reconstructed from the pinned A
    # architecture and fingerprinted on both sides of resume.
    return tensor_digest(sorted(stage.model.named_buffers()))


def save_state(stage, binding, path, last_loss):
    save_parameters(stage, binding, path, last_loss, body_parameters(stage), STATE, "body")


def load_state(stage, binding, raw, expected_step):
    return load_parameters(stage, binding, raw, expected_step, body_parameters(stage), STATE, "body")

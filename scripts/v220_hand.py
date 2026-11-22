"""Deterministic full-hand fit under one of the two declared upper conditions."""

import tempfile
import time
import uuid
from pathlib import Path

from rewind_archive.contracts import digest
from rewind_m3.cascade import UpperCondition
from rewind_v1.data import hand_clip
from rewind_v1.teacher import V1Hand
from rewind_v1.training import TeacherRun
from scripts.v211_progress import checkpoint_raw, fit_tip, publish_artifact, require_io_room
from scripts.v211_trial import TRAINING, UPDATES, load_parameters, reset_optimizer, save_parameters, tensor_digest, update
from scripts.v220_upper import CONTROL, POLICIES, condition
from scripts.v2_gpu_training_storage import MIB, publish_json


METHOD = "v2.2"
BRANCH = {"scope": "hand_all", "conditions": list(POLICIES), "updates": UPDATES, **TRAINING}
FIT = "rewind_v220_hand_fit_receipt_v1"
STATE = "rewind_v220_hand_full_state_v1"
CAP = 192*MIB
INTERVAL = 1550
COPY_SECONDS = 210
PARAMETER_TENSORS = 103
PARAMETER_VALUES = 15_571_974


class HandTeacher(TeacherRun):
    def _condition(self, upper, *, training):
        if training and isinstance(upper, UpperCondition) and upper.source == "predicted_frozen_body":
            if self.upper_policy not in POLICIES or self.upper_policy == CONTROL:
                raise ValueError("v220_predicted_upper_disallowed_in_control")
            # The original prediction token enforces the exact frozen body pin.
            return super()._condition(upper, training=False)
        return super()._condition(upper, training=training)

    def update(self, item):
        clip, upper = item
        return super().update(clip, upper=upper)


def make_hand(cascade, pipe, policy, body_pin):
    if policy not in POLICIES:
        raise ValueError("v220_upper_policy")
    hand = HandTeacher(cascade.cfg["teacher"], joints=len(cascade.fingers), edges=cascade._edges(cascade.fingers),
                       source="predicted_2d", input_sha256=cascade.input_sha256, upper_joints=len(cascade.upper_idx),
                       body_checkpoint_sha256=body_pin, device=cascade.device)
    hand.upper_policy = policy
    pipe._load_best("hand", hand)
    return hand


def parameters(stage):
    weights = dict(stage.model.named_parameters())
    if (type(stage.model) is not V1Hand or len(weights) != PARAMETER_TENSORS
            or sum(x.numel() for x in weights.values()) != PARAMETER_VALUES):
        raise ValueError("v220_hand_parameter_layout")
    return weights


def source_counts(ordered, selected, step):
    predicted = sum(key in selected for key in ordered[:step])
    return {"gt": step-predicted, "predicted": predicted}


def fit(stage, ordered, selected, target, work, binding, store, cache, cascade, ledger, owner, environment, timing):
    start = time.monotonic()
    previous = fit_tip(target, binding, FIT)
    weights = parameters(stage)
    stage.model.requires_grad_(True)
    reset_optimizer(stage, weights.values())
    buffers = tensor_digest(sorted(stage.model.named_buffers()))
    last_loss = None
    if previous:
        if (previous["buffer_sha256"] != buffers
                or previous["upper_counts"] != source_counts(ordered, selected, previous["step"])):
            raise ValueError("v220_fit_buffer_or_upper_cursor")
        last_loss = load_parameters(stage, binding, checkpoint_raw(target, previous, ledger, CAP), previous["step"],
                                    weights, STATE, "hand")
    timing.update(resumed_from=stage.step, train_seconds=0., fit_publish_seconds=0., fit_prepare_seconds=time.monotonic()-start)

    def save():
        nonlocal previous
        if stage.step <= (previous["step"] if previous else 0):
            return
        start = time.monotonic()
        with tempfile.TemporaryDirectory(dir=work, prefix="hand-fit-") as tmp:
            path = Path(tmp)/f"hand-{stage.step:05d}-{uuid.uuid4().hex}.pt"
            save_parameters(stage, binding, path, last_loss, weights, STATE, "hand")
            artifact = publish_artifact(path, target, ledger, owner, CAP, COPY_SECONDS)
            previous = publish_json(target/f"fit-{stage.step:05d}.json", FIT,
                                    {"step": stage.step, "binding_sha256": digest(binding), "artifact": artifact,
                                     "buffer_sha256": buffers, "last_loss": last_loss,
                                     "upper_counts": source_counts(ordered, selected, stage.step),
                                     "previous": previous["content_sha256"] if previous else None,
                                     "environment": environment}, owner)
        timing["fit_publish_seconds"] += time.monotonic()-start
        print(f"v2.2 {stage.upper_policy} seed {binding['seed']} fit {stage.step}/{UPDATES}", flush=True)

    while stage.step < UPDATES:
        if stage.step == (previous["step"] if previous else 0):
            require_io_room(ledger, read=2*CAP, write=CAP)
        start = time.monotonic()
        try:
            ledger.check()
            key = ordered[stage.step]
            clip = store.load(key, "train")
        except TimeoutError:
            save()
            raise
        upper = condition(clip, key, cascade, cache, selected, binding["body"])
        last_loss = update(stage, (hand_clip(clip, cascade.spec), upper), binding["seed"])
        timing["train_seconds"] += time.monotonic()-start
        if stage.step in (1, INTERVAL, UPDATES):
            if stage.step == UPDATES and tensor_digest(sorted(stage.model.named_buffers())) != buffers:
                raise ValueError("v220_hand_buffers_changed")
            save()
    stage.optimizer.zero_grad(set_to_none=True)
    stage.model.eval().requires_grad_(False)
    stage.checkpoint_sha256 = previous["artifact"]["sha256"]
    return previous

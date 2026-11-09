"""Opt-in body and hand candidates with REWIND Eq. (4) and masked motion losses."""

import torch
from torch import nn

from rewind_m2.models import BodyDenoiser
from rewind_m2.training import position_losses
from rewind_m3.models import HandDenoiser
from .attention import Equation4Block


WINDOW_FRAMES = 50


def _temporal(config):
    return nn.ModuleList(Equation4Block(config["hidden"], config["heads"], config["attention_past_frames"],
                                       config["dropout"]) for _ in range(config["temporal_layers"]))


class V1Body(BodyDenoiser):
    def __init__(self, config, joints, edges):
        super().__init__(config, joints, edges)
        self.temporal = _temporal(config)


class V1Hand(HandDenoiser):
    def __init__(self, config, joints, edges, upper_joints):
        super().__init__(config, joints, edges, upper_joints, conditioned=True)
        self.temporal = _temporal(config)


def _timestamps(values, time_valid, pair_valid):
    if values.shape != time_valid.shape or not torch.is_floating_point(values):
        raise ValueError("rewind_v1_timestamp_shape")
    if not torch.isfinite(values[time_valid]).all():
        raise ValueError("rewind_v1_timestamp_nonfinite")
    dt = values[:, 1:] - values[:, :-1]
    if not (dt[pair_valid] > 0).all():
        raise ValueError("rewind_v1_timestamp_nonincreasing")
    return torch.where(pair_valid, dt, 1.)


def _adjacent_delta(values, mask):
    clean = torch.where(mask[..., None], values, 0.)
    return clean[:, 1:] - clean[:, :-1]


def motion_losses(prediction, intermediate, target_mm, valid, time_valid, *,
                  scale_mm, frame_weight, velocity_weight=0., time_seconds=None,
                  foot_weight=0., world_prediction_mm=None, foot_contact=None):
    """Losses in normalized position units and normalized units per second.

    Contact is a reviewed [B,T,J] bool signal; true joints must be feet. World pose
    must share the same joint order. No world/contact inference is performed here.
    """
    if any(not 0 <= weight < float("inf") for weight in (frame_weight, velocity_weight, foot_weight)):
        raise ValueError("rewind_v1_loss_weight")
    loss, position, frame = position_losses(prediction.float(), intermediate.float(), target_mm,
                                            valid, time_valid, scale_mm, frame_weight)
    components = {"position": position, "frame": frame}
    if not (velocity_weight or foot_weight):
        return loss, components
    if time_seconds is None:
        raise ValueError("rewind_v1_verified_time_required")
    time_pair = time_valid[:, 1:] & time_valid[:, :-1]
    dt = _timestamps(time_seconds, time_valid, time_pair)
    dt = dt.float()[..., None, None]
    mask = valid & time_valid[..., None]
    pairs = mask[:, 1:] & mask[:, :-1]
    if velocity_weight:
        pred = _adjacent_delta(prediction.float(), mask)
        target = _adjacent_delta(target_mm.float() / scale_mm, mask)
        velocity = ((pred - target)[pairs] / dt.expand_as(pred)[pairs]).square().sum() / (3 * pairs.sum().clamp_min(1))
        components["velocity"] = velocity
        loss = loss + velocity_weight * velocity
    if foot_weight:
        if (world_prediction_mm is None or world_prediction_mm.shape != prediction.shape or foot_contact is None
                or foot_contact.shape != valid.shape or foot_contact.dtype != torch.bool):
            raise ValueError("rewind_v1_verified_world_and_contact_required")
        contact = foot_contact & mask
        stationary = contact[:, 1:] & contact[:, :-1]
        world_delta = _adjacent_delta(world_prediction_mm.float() / scale_mm, contact)
        foot = (world_delta[stationary] / dt.expand_as(world_delta)[stationary]).square().sum() / (
            3 * stationary.sum().clamp_min(1))
        components["foot"] = foot
        loss = loss + foot_weight * foot
    if not torch.isfinite(loss):
        raise ValueError("rewind_v1_nonfinite_motion_loss")
    return loss, components

"""현재 anchor의 유효 관절만 집계하는 loss/MPJPE 계약."""

import torch
import math


def masked_mse(prediction_mm, target_mm, valid, scale_mm):
    if type(scale_mm) not in (int, float) or not math.isfinite(scale_mm) or scale_mm <= 0:
        raise ValueError("양의 유한한 고정 scale_mm 필요")
    if prediction_mm.shape != target_mm.shape or valid.shape != target_mm.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Loss shape/bool mask 불일치")
    if not valid.any():
        raise ValueError("유효 GT가 없는 batch로 학습하지 않습니다.")
    residual = (prediction_mm[valid] - target_mm[valid]) / scale_mm
    if not torch.isfinite(residual).all():
        raise ValueError("유효 관절 loss에 비유한 값")
    return residual.square().mean()  # 유효 joint 및 xyz 전체 평균, dimensionless


def metric_sums(prediction_mm, target_mm, valid):
    if prediction_mm.shape != target_mm.shape or valid.shape != target_mm.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Metric shape 불일치")
    residual = prediction_mm[valid] - target_mm[valid]
    if not torch.isfinite(residual).all():
        raise ValueError("Metric 비유한 값")
    return {"distance_sum_mm": float(torch.linalg.vector_norm(residual, dim=-1).sum()),
            "squared_sum_mm2": float(residual.square().sum()), "valid_joints": int(valid.sum())}

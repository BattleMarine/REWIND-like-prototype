"""개발용 작은 regression 비교 모델. 논문 크기 모델이 아니다."""

import torch
from torch import nn


class SingleFrameRegressor(nn.Module):
    def __init__(self, feature_dim, joints, hidden=32, dropout=0.):
        super().__init__()
        self.joints = joints
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, joints*3))

    def forward(self, features, time_valid):
        output = self.net(features).reshape(*features.shape[:2], self.joints, 3)
        return output * time_valid[..., None, None]


class TemporalRegressor(nn.Module):
    def __init__(self, feature_dim, joints, hidden=32, dropout=0.):
        super().__init__()
        self.joints, self.hidden = joints, hidden
        self.cell = nn.GRUCell(feature_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, joints*3)

    def forward(self, features, time_valid):
        h = features.new_zeros(features.shape[0], self.hidden)
        outputs = []
        for index in range(features.shape[1]):
            valid = time_valid[:, index, None]
            h = torch.where(valid, self.cell(features[:, index], h), h)
            out = self.head(self.dropout(h)).reshape(-1, self.joints, 3)
            outputs.append(out * valid[..., None])
        return torch.stack(outputs, dim=1)


def make_model(config, feature_dim, joints):
    model = {"single_frame": SingleFrameRegressor, "temporal": TemporalRegressor}.get(config["model"])
    if model is None:
        raise ValueError("model은 single_frame/temporal이어야 합니다.")
    return model(feature_dim, joints, config["hidden"], config["dropout"])

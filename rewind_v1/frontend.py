"""Paper-inspired stereo heatmap frontend, independent from the v0 checkpoint."""

import hashlib
import torch
from torch import nn
from rewind_m1.frontend import StereoHeatmapNet


ARCHITECTURE = "rewind_v1_mbconv_heatmap"


def _conv(in_channels, out_channels, size, *, stride=1, groups=1, activation=nn.Hardswish):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, size, stride=stride, padding=size // 2,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(out_channels), activation())


class _MBConv(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        expanded = input_channels * 4
        self.spatial = nn.Sequential(_conv(input_channels, expanded, 1),
                                     _conv(expanded, expanded, 3, stride=stride, groups=expanded))
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(expanded, input_channels, 1),
                                  nn.SiLU(), nn.Conv2d(input_channels, expanded, 1), nn.Sigmoid())
        self.project = _conv(expanded, output_channels, 1, activation=nn.Identity)
        self.residual = stride == 1 and input_channels == output_channels

    def forward(self, x):
        spatial = self.spatial(x)
        out = self.project(spatial * self.gate(spatial))
        return x + out if self.residual else out


class PaperHeatmapFrontend(StereoHeatmapNet):
    def __init__(self, joints):
        nn.Module.__init__(self)
        self.joints = joints
        blocks = [_conv(3, 16, 3, stride=2)]
        channels = 16
        # Four stacks. Widths/repeats are read as consecutive groups within each stack.
        for stack in range(4):
            for width, repeats in ((16, 1), (24, 2), (40, 2)):
                for repeat in range(repeats):
                    stride = 2 if stack < 3 and width == 16 and repeat == 0 else 1
                    blocks.append(_MBConv(channels, width, stride))
                    channels = width
        for width in (40, 24, 16):
            blocks.extend((nn.Upsample(scale_factor=2, mode="nearest"),
                           _conv(channels, width, 3, activation=nn.ReLU),
                           _conv(width, width, 3, activation=nn.ReLU)))
            channels = width
        self.encoder = nn.Sequential(*blocks, nn.Conv2d(channels, joints, 1))

    def forward(self, images):
        if any(size < 32 or size % 16 for size in images.shape[-2:]):
            raise ValueError("rewind_v1_frontend_spatial_size")
        return super().forward(images)


@torch.no_grad()
def predicted_features(frontend, images):
    """Convert [T,2,3,H,W] to M2's joint-major [T,14J] features, without GT fallback."""
    if frontend.training:
        raise ValueError("rewind_v1_frontend_requires_eval_mode")
    _, xy, confidence = frontend(images)
    h, w = images.shape[-2:]
    xy = 2 * xy / xy.new_tensor([w - 1, h - 1]) - 1
    one, zero = torch.ones_like(confidence), torch.zeros_like(confidence)
    views = torch.cat((xy, one[..., None], confidence[..., None], one[..., None],
                       zero[..., None], zero[..., None]), dim=-1)
    return views.transpose(1, 2).flatten(2).flatten(1)


def frontend_fingerprint(frontend):
    if not isinstance(frontend, PaperHeatmapFrontend):
        raise ValueError("rewind_v1_frontend_architecture")
    digest = hashlib.sha256()
    for name, value in sorted(frontend.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(repr((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()

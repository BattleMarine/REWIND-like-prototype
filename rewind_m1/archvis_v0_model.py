"""64px stereo heatmap frontend for the fixed ArchVis v0 development run."""

from torch import nn

from .data import DataContractError
from .frontend import StereoHeatmapNet


ARCHITECTURE = "context_heatmap_v0"


def _block(in_channels, out_channels, *, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(),
    )


def _up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(),
    )


class _ContextEncoderDecoder(nn.Module):
    def __init__(self, joints):
        super().__init__()
        self.down32 = _block(3, 16, stride=2)
        self.down16 = _block(16, 24, stride=2)
        self.down8 = _block(24, 32, stride=2)
        self.down4 = _block(32, 48, stride=2)
        self.context = nn.Sequential(nn.Conv2d(48, 48, 4, bias=False), nn.GroupNorm(4, 48), nn.ReLU())
        self.up8 = _up(48, 32)
        self.up16 = _up(32, 24)
        self.up32 = _up(24, 16)
        self.head = nn.Conv2d(16, joints, 1)

    def forward(self, images):
        at32 = self.down32(images)
        at16 = self.down16(at32)
        at8 = self.down8(at16)
        at4 = self.down4(at8)
        decoded = self.up8(at4 + self.context(at4)) + at8
        decoded = self.up16(decoded) + at16
        return self.head(self.up32(decoded) + at32)


class ContextHeatmapNet(StereoHeatmapNet):
    def __init__(self, joints):
        nn.Module.__init__(self)
        self.joints = joints
        self.encoder = _ContextEncoderDecoder(joints)

    def forward(self, images):
        if images.shape[-2:] != (64, 64):
            raise DataContractError("context_heatmap_v0 requires 64x64 inputs")
        return super().forward(images)

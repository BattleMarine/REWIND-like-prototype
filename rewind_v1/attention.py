"""Candidate for REWIND Eq. (4), with explicit feature-map and FP32 assumptions."""

import torch
from torch.nn import functional as F
from rewind_m2.models import TemporalBlock


def kernel_attention(q, k, v, valid, rotary, positions, past_frames, epsilon=1e-6):
    """Inputs [B,H,T,D], valid [B,T]. The denominator uses unrotated features."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        valid_tokens = valid[:, None, :, None]
        q = F.elu(torch.where(valid_tokens, q.float(), 0.0)) + 1
        k = F.elu(torch.where(valid_tokens, k.float(), 0.0)) + 1
        v = torch.where(valid_tokens, v.float(), 0.0)
        distance = torch.arange(q.shape[-2], device=q.device)
        distance = distance[:, None] - distance[None, :]
        mask = (distance >= 0) & (distance <= past_frames)
        allowed = mask[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
        numerator_weights = (rotary(q, positions) @ rotary(k, positions).transpose(-2, -1)).masked_fill(~allowed, 0.)
        normalizer = (q @ k.transpose(-2, -1)).masked_fill(~allowed, 0.).sum(-1, keepdim=True)
        result = (numerator_weights @ v) / normalizer.clamp_min(epsilon)
        return torch.where(valid_tokens, result, 0.)


class Equation4Block(TemporalBlock):
    def forward(self, x, valid, positions):
        b, t, j, d = x.shape
        clean = torch.where(valid[..., None, None], x, 0.)
        h = self.norm(clean).permute(0, 2, 1, 3).reshape(b * j, t, d)
        q, k, v = self.qkv(h).reshape(b * j, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        key_valid = valid[:, None, :].expand(b, j, t).reshape(b * j, t)
        out = kernel_attention(q, k, v, key_valid, self.rotary, positions, self.past_frames)
        out = out.to(x.dtype).transpose(1, 2).reshape(b, j, t, d).permute(0, 2, 1, 3)
        x = clean + self.drop(self.out(out))
        x = x + self.drop(self.ff(self.norm_f(x)))
        return torch.where(valid[..., None, None], x, 0.)

"""
Satellite semantic prior head for residual logit correction.

Input:
    sat_tokens: (B, N, C), where N = gs * gs
Output:
    delta_logits: (B, num_classes, H, W)
    u_sat: (B, 1, H, W) in [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class SatSemanticPriorHead(nn.Module):
    """Predict satellite prior residual logits and confidence map."""

    def __init__(self,
                 in_channels=256,
                 hidden_channels=256,
                 num_classes=3,
                 delta_scale_init=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.conf_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init)))

    def forward(self, sat_tokens, sat_grid_size=None, target_hw=None):
        if sat_tokens is None:
            raise ValueError('sat_tokens must not be None when SatSemanticPriorHead is enabled.')
        if target_hw is None:
            raise ValueError('target_hw must be provided as (H, W).')

        gs = sat_grid_size
        if gs is None:
            n_tokens = sat_tokens.shape[1]
            gs = int(round(n_tokens ** 0.5))
        if gs * gs != sat_tokens.shape[1]:
            raise ValueError(
                f'sat_tokens length ({sat_tokens.shape[1]}) does not match grid_size^2 ({gs * gs}).')

        bs, _, ch = sat_tokens.shape
        sat_spatial = sat_tokens.permute(0, 2, 1).reshape(bs, ch, gs, gs)
        sat_spatial = F.interpolate(sat_spatial, size=target_hw, mode='bilinear', align_corners=False)

        feat = self.trunk(sat_spatial)
        delta_logits = self.delta_scale * self.delta_head(feat)
        u_sat = torch.sigmoid(self.conf_head(feat))
        return delta_logits, u_sat

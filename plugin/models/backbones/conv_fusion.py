"""
Conv-based late fusion of BEV features and satellite tokens.

Applied after BEVFormer + neck, before seg/det heads.
Complements the early cross-attention inside BEVFormer encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SatConvFusion(nn.Module):
    """Fuse BEV features with upsampled satellite tokens via conv layers + gated residual.

    Flow:
        sat_tokens (bs, N, C) -> reshape (bs, C, gs, gs) -> upsample (bs, C, H, W)
        concat [bev, sat_up] (bs, 2C, H, W) -> conv -> fused (bs, C, H, W)
        output = (1-g)*bev + g*fused
    """

    def __init__(self,
                 in_channels=256,
                 sat_grid_size=14,
                 bev_size=(50, 100),
                 gate_init=-1.0):
        super().__init__()
        self.sat_grid_size = sat_grid_size
        self.bev_size = bev_size

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, bev_feats, sat_tokens, sat_grid_size=None):
        """
        Args:
            bev_feats: (bs, C, bev_h, bev_w)
            sat_tokens: (bs, N, C) where N = sat_grid_size^2
            sat_grid_size: int
        Returns:
            fused: (bs, C, bev_h, bev_w)
        """
        if sat_tokens is None:
            return bev_feats

        gs = sat_grid_size or self.sat_grid_size
        bs, N, C = sat_tokens.shape

        if gs * gs != N:
            raise ValueError(
                f'sat_tokens length ({N}) does not match grid_size^2 ({gs * gs}).')

        sat_spatial = sat_tokens.permute(0, 2, 1).reshape(bs, C, gs, gs)
        sat_up = F.interpolate(sat_spatial, size=bev_feats.shape[-2:],
                               mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([bev_feats, sat_up], dim=1))

        g = torch.sigmoid(self.gate)
        return (1 - g) * bev_feats + g * fused

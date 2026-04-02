"""Satellite-Camera BEV Fusion modules.

Config-switchable fusion strategies for combining camera and satellite BEV features.
Used in Stage 3 joint finetuning of SatMapTracker.
"""
import torch
import torch.nn as nn
from mmdet.models import NECKS


@NECKS.register_module()
class SatCamGatedFusion(nn.Module):
    """Learnable spatial gating fusion.

    fused = gate * cam_bev + (1 - gate) * sat_bev

    Args:
        in_channels (int): Input feature channels.
        gate_hidden_dim (int): Hidden dim for gating network.
        gate_bias (float): Initial bias for gate sigmoid.
    """

    def __init__(self, in_channels=256, gate_hidden_dim=64, gate_bias=1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden_dim, 1, 1, bias=True),
        )
        # Initialize to favor camera features
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, cam_bev, sat_bev):
        concat = torch.cat([cam_bev, sat_bev], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * cam_bev + (1 - gate) * sat_bev
        info = {'gate_mean': gate.mean().item(), 'gate_std': gate.std().item()}
        return fused, info


@NECKS.register_module()
class SatCamModulation(nn.Module):
    """FiLM/SPADE-style satellite-conditioned modulation.

    fused = gamma(sat) * BN(cam) + beta(sat) + cam  (with residual)

    Satellite acts as structural prior that modulates camera features,
    avoiding gate collapse and handling feature norm imbalance.

    Args:
        in_channels (int): Feature channels.
        hidden_dim (int): Hidden dim for modulation parameter generation.
        use_residual (bool): Add cam_bev residual.
    """

    def __init__(self, in_channels=256, hidden_dim=128, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        self.norm = nn.BatchNorm2d(in_channels)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma_conv = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        self.beta_conv = nn.Conv2d(hidden_dim, in_channels, 3, padding=1)

        # Init: gamma=1, beta=0 → camera passthrough at start
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.ones_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, cam_bev, sat_bev):
        h = self.shared(sat_bev)
        gamma = self.gamma_conv(h)
        beta = self.beta_conv(h)

        modulated = gamma * self.norm(cam_bev) + beta
        if self.use_residual:
            modulated = modulated + cam_bev

        info = {
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std().item(),
            'beta_norm': beta.norm(dim=1).mean().item(),
        }
        return modulated, info


@NECKS.register_module()
class SatCamConvFusion(nn.Module):
    """Concat + Conv fusion (BEVFusion-style) with camera residual.

    fused = Conv(cat(proj_cam, proj_sat)) + cam_bev

    Camera residual ensures compatibility with downstream heads trained
    on cam_bev distribution (Stage 2). Fusion output acts as additive delta.

    Args:
        in_channels (int): Input channels per modality.
        hidden_channels (int): Hidden channels in conv blocks.
    """

    def __init__(self, in_channels=256, hidden_channels=256):
        super().__init__()
        self.proj_cam = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.proj_sat = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, cam_bev, sat_bev):
        concat = torch.cat([self.proj_cam(cam_bev), self.proj_sat(sat_bev)], dim=1)
        fused = self.fuse(concat) + cam_bev
        return fused, {}


@NECKS.register_module()
class SatelliteConvFuser(nn.Module):
    """SatMap-style ConvFuser: concat + conv fusion (BEVFusion baseline).

    No residual connection. Projects both modalities, concatenates,
    then processes through conv blocks.

    Reference: SatMap (arXiv:2601.10512), BEVFusion (ICRA 2023)

    Args:
        in_channels (int): Input channels per modality.
        hidden_channels (int): Hidden channels in conv blocks.
    """

    def __init__(self, in_channels=256, hidden_channels=256):
        super().__init__()
        self.proj_cam = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.proj_sat = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cam_bev, sat_bev):
        cam_proj = self.proj_cam(cam_bev)
        sat_proj = self.proj_sat(sat_bev)
        concat = torch.cat([cam_proj, sat_proj], dim=1)
        fused = self.fuse(concat)
        return fused, {}

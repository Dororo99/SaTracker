"""Dual-Path Deformable Attention Fusion.

Camera와 satellite BEV feature를 deformable cross-attention으로 융합.
두 modality가 spatially aligned (같은 ego-centric BEV 좌표)이므로
reference point = identity grid, offset만 학습.

Dual-path:
  - cam→sat path: camera query가 satellite에서 보완 정보를 가져옴
  - sat→cam path: satellite query가 camera에서 fine structure 가져옴
  - 두 path 결과를 spatial gate로 pixel-wise 조합
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner.base_module import BaseModule
from mmdet.models import NECKS

from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)

try:
    from plugin.models.transformer_utils.fp16_dattn import (
        MultiScaleDeformableAttnFunctionFp32,
    )
    HAS_FP32_DATTN = True
except ImportError:
    HAS_FP32_DATTN = False


class BEVDeformableCrossAttention(BaseModule):
    """Single-direction deformable cross-attention between two aligned BEV maps.

    Query comes from one modality, Key/Value from the other.
    Reference points are an identity grid (same position) with learned offsets.

    Args:
        embed_dims (int): Feature dimension.
        num_heads (int): Number of attention heads.
        num_points (int): Sampling points per query per head.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dims=256, num_heads=8, num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = 1  # single-scale BEV

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * self.num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * self.num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query_feat, kv_feat, bev_h, bev_w):
        """
        Args:
            query_feat: (B, C, H, W) — query modality BEV features
            kv_feat: (B, C, H, W) — key/value modality BEV features
            bev_h, bev_w: spatial dimensions
        Returns:
            attended: (B, C, H, W)
        """
        B, C, H, W = query_feat.shape
        N = H * W

        query = query_feat.flatten(2).permute(0, 2, 1)
        value = kv_feat.flatten(2).permute(0, 2, 1)

        ref_y = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=query.device, dtype=query.dtype)
        ref_x = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=query.device, dtype=query.dtype)
        ref_y, ref_x = torch.meshgrid(ref_y, ref_x, indexing='ij')
        ref_points = torch.stack([ref_x.flatten(), ref_y.flatten()], dim=-1)
        ref_points = ref_points[None, :, None, :].expand(B, -1, self.num_levels, -1)

        spatial_shapes = torch.tensor([[H, W]], device=query.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=query.device, dtype=torch.long)

        value = self.value_proj(value)
        value = value.view(B, N, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            B, N, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            B, N, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            B, N, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        reference_points_expanded = ref_points[:, :, None, :, None, :].expand(
            -1, -1, self.num_heads, -1, self.num_points, -1)
        sampling_locations = reference_points_expanded + \
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        if torch.cuda.is_available() and value.is_cuda and HAS_FP32_DATTN:
            output = MultiScaleDeformableAttnFunctionFp32.apply(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        output = self.dropout(output)

        return output.permute(0, 2, 1).reshape(B, C, H, W)


@NECKS.register_module()
class SatCamDeformAttnFusion(nn.Module):
    """Dual-Path Deformable Attention Fusion.

    Two cross-attention paths exchange information between cam and sat BEV,
    then a spatial gate combines the two attended results pixel-wise.

    Args:
        in_channels (int): BEV feature channels.
        num_heads (int): Attention heads.
        num_points (int): Deformable sampling points per query.
        num_classes (int): Unused, kept for config compatibility.
        ffn_dim (int): FFN hidden dimension.
        dropout (float): Dropout rate.
    """

    def __init__(self,
                 in_channels=256,
                 num_heads=8,
                 num_points=4,
                 num_classes=3,
                 ffn_dim=512,
                 dropout=0.1):
        super().__init__()
        self.in_channels = in_channels

        # ── Dual-path deformable cross-attention ──
        self.cam2sat_attn = BEVDeformableCrossAttention(
            embed_dims=in_channels, num_heads=num_heads,
            num_points=num_points, dropout=dropout)
        self.sat2cam_attn = BEVDeformableCrossAttention(
            embed_dims=in_channels, num_heads=num_heads,
            num_points=num_points, dropout=dropout)

        # Layer norms
        self.norm_cam2sat = nn.LayerNorm(in_channels)
        self.norm_sat2cam = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)

        # FFN after gated combination
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, in_channels),
            nn.Dropout(dropout),
        )

        # ── Spatial gate ──
        # Learns pixel-wise weight for cam_attended vs sat_attended.
        # Input: concatenation of the two attended features.
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=True),
        )
        # Init: sigmoid(0)=0.5 → equal cam/sat weight at start
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, cam_bev, sat_bev):
        """
        Args:
            cam_bev: (B, C, H, W) camera BEV features
            sat_bev: (B, C, H, W) satellite BEV features
        Returns:
            fused: (B, C, H, W) fused BEV features
            info: dict with monitoring metrics
        """
        B, C, H, W = cam_bev.shape

        # ── Path 1: Camera queries Satellite ──
        cam2sat_out = self.cam2sat_attn(cam_bev, sat_bev, H, W)
        cam_attended = cam_bev + cam2sat_out
        cam_attended = self.norm_cam2sat(
            cam_attended.flatten(2).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(B, C, H, W)

        # ── Path 2: Satellite queries Camera ──
        sat2cam_out = self.sat2cam_attn(sat_bev, cam_bev, H, W)
        sat_attended = sat_bev + sat2cam_out
        sat_attended = self.norm_sat2cam(
            sat_attended.flatten(2).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(B, C, H, W)

        # ── Spatial gating ──
        gate_input = torch.cat([cam_attended, sat_attended], dim=1)
        g = torch.sigmoid(self.gate(gate_input))  # (B, C, H, W)
        fused = g * cam_attended + (1 - g) * sat_attended

        # ── FFN ──
        fused_flat = fused.flatten(2).permute(0, 2, 1)
        fused_flat = fused_flat + self.ffn(fused_flat)
        fused_flat = self.norm_ffn(fused_flat)
        fused = fused_flat.permute(0, 2, 1).reshape(B, C, H, W)

        # ── Residual to camera ──
        fused = fused + cam_bev

        # ── Monitoring ──
        with torch.no_grad():
            info = {
                'gate_cam_mean': g.mean().item(),
                'gate_sat_mean': (1 - g).mean().item(),
            }

        return fused, info

"""
Satellite image encoder using ResNet + FPN.
Returns tokens (B, N, C) + grid_size, same interface as SatMAEEncoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES, build_backbone


@BACKBONES.register_module()
class SatelliteEncoder(nn.Module):
    """Encode satellite images into patch tokens via ResNet + FPN.

    Returns (tokens, grid_size) — same interface as SatMAEEncoder,
    so it can be used as a drop-in replacement.

    Args:
        backbone_cfg (dict): Config for ResNet backbone.
        neck_in_channels (list): Input channels from each backbone stage.
        out_channels (int): Output token channels (should match bev_embed_dims).
        token_grid_size (int): Output spatial grid size. Tokens = grid_size^2.
        frozen (bool): Whether to freeze all parameters.
    """

    def __init__(self,
                 backbone_cfg=None,
                 neck_in_channels=None,
                 out_channels=256,
                 token_grid_size=14,
                 frozen=False):
        super().__init__()
        self.out_channels = out_channels
        self.token_grid_size = token_grid_size

        if backbone_cfg is None:
            backbone_cfg = dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                norm_cfg=dict(type='BN2d'),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            )

        if neck_in_channels is None:
            depth = backbone_cfg.get('depth', 50)
            if depth in (18, 34):
                neck_in_channels = [64, 128, 256, 512]
            else:
                neck_in_channels = [256, 512, 1024, 2048]

        self.backbone = build_backbone(backbone_cfg)

        # FPN-like neck: lateral convs + sum aggregation
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in neck_in_channels
        ])
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if frozen:
            self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def init_weights(self):
        self.backbone.init_weights()
        for m in self.lateral_convs:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        for m in self.output_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, sat_img):
        """
        Args:
            sat_img (Tensor): (B, 3, H, W) satellite image.
        Returns:
            tokens (Tensor): (B, grid_size^2, out_channels)
            grid_size (int): spatial grid dimension
        """
        feats = self.backbone(sat_img)

        # FPN-like aggregation: upsample and sum
        target_size = feats[0].shape[2:]
        out = self.lateral_convs[0](feats[0])
        for i in range(1, len(feats)):
            lateral = self.lateral_convs[i](feats[i])
            lateral = F.interpolate(lateral, size=target_size,
                                    mode='bilinear', align_corners=False)
            out = out + lateral

        out = self.output_conv(out)

        # Resize to square token grid
        gs = self.token_grid_size
        out = F.interpolate(out, size=(gs, gs),
                            mode='bilinear', align_corners=False)

        # (B, C, gs, gs) → (B, gs*gs, C)
        tokens = out.flatten(2).permute(0, 2, 1)

        return tokens, gs

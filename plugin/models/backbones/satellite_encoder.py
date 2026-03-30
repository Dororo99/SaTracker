"""Satellite image encoder for BEV-aligned feature extraction.

Encodes AID4AD aerial imagery into BEV features matching camera BEV shape.
Uses any mmdet backbone (ResNet, etc.) with FPN-like multi-scale aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES, build_backbone


@BACKBONES.register_module()
class SatelliteEncoder(nn.Module):
    """Encode satellite images into BEV-aligned feature maps.

    Args:
        backbone_cfg (dict): Config for backbone (defaults to ResNet50).
        neck_in_channels (list): Input channels from each backbone stage.
        out_channels (int): Output feature channels (should match bev_embed_dims).
        bev_size (tuple): Target BEV feature size (H, W).
        frozen (bool): Freeze all parameters after init.
    """

    def __init__(self,
                 backbone_cfg=None,
                 neck_in_channels=None,
                 out_channels=256,
                 bev_size=(50, 100),
                 frozen=False):
        super().__init__()
        self.bev_size = bev_size
        self.out_channels = out_channels

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
                init_cfg=dict(type='Pretrained',
                              checkpoint='torchvision://resnet50'),
            )

        if neck_in_channels is None:
            depth = backbone_cfg.get('depth', 50)
            if depth in (18, 34):
                neck_in_channels = [64, 128, 256, 512]
            else:
                neck_in_channels = [256, 512, 1024, 2048]

        self.backbone = build_backbone(backbone_cfg)

        # FPN-like neck: merge multi-scale features
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
        self._frozen = True
        self.eval()

    def train(self, mode=True):
        """Override train to keep BN in eval mode when frozen."""
        if getattr(self, '_frozen', False):
            # Keep everything in eval mode (BN uses running stats)
            return super().train(False)
        return super().train(mode)

    def forward(self, sat_img):
        """
        Args:
            sat_img (Tensor): (B, 3, H, W) normalized satellite image.
        Returns:
            Tensor: (B, C, bev_h, bev_w) satellite BEV features.
        """
        feats = self.backbone(sat_img)

        # FPN-like aggregation: upsample all to finest resolution then sum
        target_size = feats[0].shape[2:]
        out = self.lateral_convs[0](feats[0])
        for i in range(1, len(feats)):
            lateral = self.lateral_convs[i](feats[i])
            lateral = F.interpolate(lateral, size=target_size,
                                    mode='bilinear', align_corners=False)
            out = out + lateral

        out = self.output_conv(out)

        # Resize to target BEV size if needed
        if out.shape[2:] != (self.bev_size[0], self.bev_size[1]):
            out = F.interpolate(out, size=self.bev_size,
                                mode='bilinear', align_corners=False)
        return out

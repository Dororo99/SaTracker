"""Lightweight satellite encoder based on ResNet-18.

Satellite images are already top-down BEV-aligned, so no 3D projection needed.
Simply extract features and resize to target BEV resolution.
"""
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from torchvision.models import resnet18


@BACKBONES.register_module()
class SimpleSatEncoder(BaseModule):
    """ResNet-18 based satellite encoder.

    Input:  (B, 3, H_sat, W_sat) - normalized satellite image
    Output: (B, out_channels, bev_h, bev_w) - BEV-aligned features
    """

    def __init__(self, out_channels=256, bev_size=(50, 100),
                 pretrained=True, frozen=False):
        super().__init__()
        self.bev_size = bev_size
        self._frozen = False

        # ResNet-18 backbone (remove avgpool + fc)
        backbone = resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64ch
        self.layer2 = backbone.layer2  # 128ch
        self.layer3 = backbone.layer3  # 256ch
        self.layer4 = backbone.layer4  # 512ch

        # Project to target channels
        self.proj = nn.Sequential(
            nn.Conv2d(512, out_channels, 1, bias=False),
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
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    def forward(self, sat_img):
        x = self.layer0(sat_img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        if x.shape[2:] != tuple(self.bev_size):
            x = F.interpolate(x, size=self.bev_size,
                              mode='bilinear', align_corners=False)
        return x

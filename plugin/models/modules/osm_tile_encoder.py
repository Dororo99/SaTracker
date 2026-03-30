"""SD Raster Encoder and BEV Fusion for Stage 1.

Encodes SD raster images into BEV-sized feature maps,
then fuses with BEV features via spatially-blurred gated addition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OSMTileEncoder(nn.Module):
    """Lightweight CNN to encode SD raster image to BEV-sized feature map.

    Input:  [B, 1, 100, 200]  (SD raster in ego BEV frame)
    Output: [B, embed_dims, 50, 100]  (matches BEV grid: bev_h=50, bev_w=100)
    """

    def __init__(self, in_channels=1, embed_dims=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),  # (100,200) → (50,100)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dims, 3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tile):
        """tile: [B, 1, 100, 200] → [B, embed_dims, 50, 100]"""
        return self.encoder(tile)


class TileBEVFusion(nn.Module):
    """Spatially-blurred gated addition: SD features are spread via large-kernel
    depthwise conv then added to BEV features with a sigmoid gate.

    Handles SD map positional noise (~2m) by blurring features over a
    local neighborhood before fusion. Conveys "roughly this shape exists
    around here" rather than precise spatial alignment.
    """

    def __init__(self, embed_dims=256, blur_kernel=11):
        super().__init__()
        self.embed_dims = embed_dims
        pad = blur_kernel // 2

        # Learnable spatial blur: depthwise conv spreads features over local area
        # kernel=11 at 0.6m/px ≈ 6.6m coverage, handles ~2-3m SD noise
        self.spatial_blur = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, blur_kernel, padding=pad,
                      groups=embed_dims, bias=False),
            nn.BatchNorm2d(embed_dims),
        )
        # Channel mixing after blur
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )
        # Sigmoid gate: init sigmoid(-3) ≈ 0.05
        self.gamma_raw = nn.Parameter(torch.tensor(-3.0))

    def init_weights(self):
        nn.init.constant_(self.gamma_raw, -3.0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, bev_feat, tile_feat):
        """
        bev_feat:  [B, C, H, W]  (e.g. [B, 256, 50, 100])
        tile_feat: [B, C, H, W]  (same spatial size)
        Returns:   [B, C, H, W]  (tile-enhanced BEV features)
        """
        # Spatial blur: spread tile features to handle misalignment
        blurred = self.spatial_blur(tile_feat)
        blurred = self.proj(blurred)

        # Gated addition
        gamma = torch.sigmoid(self.gamma_raw)
        return bev_feat + gamma * blurred

"""
SD Prior Encoder: per-point SD token generation for decoder cross-attention.

Architecture (per element):
    raw polyline [Pi, 2] → resample to [num_pts, 2] → [0,1] normalize
    → sinusoidal PE (32-dim) + Conv1d neighbor context (224-dim) = geo [256]
    → SemanticEncoder (256-dim) broadcast to per-point
    → concat [geo(256) | sem(256)] = [512] per point

Output: sd_features [B, max_sd_tokens, 512], sd_coords [B, max_sd_tokens, 2]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROSSING_CLASS = 11  # highway_class 0~10 = vehicle ways, 11 = crossing
COORD_PE_DIM = 32    # sinusoidal_coord_encoding output dim (num_freqs=8)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def resample_polyline_torch(coords, num_pts=5):
    """Resample variable-length polyline to fixed num_pts via linear interpolation.

    Pure torch implementation — no shapely dependency.

    Args:
        coords: Tensor[Pi, 2], ego BEV meters
        num_pts: number of output points
    Returns:
        Tensor[num_pts, 2]
    """
    if coords.shape[0] < 2:
        return coords[0:1].expand(num_pts, -1).clone()

    diffs = coords[1:] - coords[:-1]                           # [Pi-1, 2]
    seg_lens = torch.norm(diffs, dim=1)                         # [Pi-1]
    cum_lens = torch.cat([torch.zeros(1, device=coords.device),
                          seg_lens.cumsum(0)])                  # [Pi]
    total_len = cum_lens[-1]

    if total_len < 1e-6:
        return coords[0:1].expand(num_pts, -1).clone()

    targets = torch.linspace(0, total_len, num_pts, device=coords.device)
    idx = torch.searchsorted(cum_lens, targets).clamp(1, len(coords) - 1)
    t = (targets - cum_lens[idx - 1]) / (seg_lens[idx - 1] + 1e-8)
    points = coords[idx - 1] + t.unsqueeze(-1) * diffs[idx - 1]
    return points  # [num_pts, 2]


def sinusoidal_coord_encoding(coords, num_freqs=8):
    """Deterministic sinusoidal positional encoding for 2D coordinates.

    No learnable parameters.

    Args:
        coords: [..., 2], [0,1] normalized coordinates
        num_freqs: number of frequency bands (default 8 → output 32-dim)
    Returns:
        [..., 4 * num_freqs] = [..., 32]
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=coords.device,
                                  dtype=coords.dtype)) * math.pi
    x = coords.unsqueeze(-1) * freqs          # [..., 2, num_freqs]
    enc = torch.cat([x.sin(), x.cos()], dim=-1)  # [..., 2, 2*num_freqs]
    return enc.flatten(-2)                        # [..., 4*num_freqs]


# ---------------------------------------------------------------------------
# Geometry Encoder
# ---------------------------------------------------------------------------
class GeometryEncoder(nn.Module):
    """Encode polylines into per-point features: sinusoidal coord PE + Conv1d neighbor context.

    Output per point: [sin_PE(32) | conv1d(geo_dim - 32)]
    No pooling — each point becomes an independent SD token.
    """

    def __init__(self, geo_dim=256, num_pts=5):
        super().__init__()
        self.num_pts = num_pts
        self.geo_dim = geo_dim
        conv_out_dim = geo_dim - COORD_PE_DIM  # 256 - 32 = 224

        # Conv1d: neighbor context (kernel_size=3, each point sees left+right neighbors)
        self.conv1 = nn.Conv1d(2, conv_out_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_out_dim // 2, conv_out_dim, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(conv_out_dim)

    def forward(self, geometries, roi_size):
        """
        Args:
            geometries: List[ndarray or Tensor [Pi, 2]], length K, ego BEV meters
            roi_size: (60, 30)
        Returns:
            geo_feats: [K, num_pts, geo_dim]  — [sin_PE(32) | conv1d(224)]
            coords:    [K, num_pts, 2]        — [0,1] normalized coords (for key_pos)
        """
        K = len(geometries)
        device = self.conv1.weight.device

        if K == 0:
            return (torch.zeros(0, self.num_pts, self.geo_dim, device=device),
                    torch.zeros(0, self.num_pts, 2, device=device))

        # 1. Resample: variable [Pi, 2] → fixed [num_pts, 2]
        resampled = torch.stack([
            resample_polyline_torch(
                torch.as_tensor(g, dtype=torch.float32, device=device),
                self.num_pts)
            for g in geometries
        ])  # [K, num_pts, 2]

        # 2. Normalize to [0, 1]
        roi_half = torch.tensor([roi_size[0] / 2, roi_size[1] / 2],
                                device=device, dtype=torch.float32)
        roi_full = torch.tensor([roi_size[0], roi_size[1]],
                                device=device, dtype=torch.float32)
        coords = (resampled + roi_half) / roi_full  # [K, num_pts, 2]

        # 3. Sinusoidal coord encoding (no learnable params)
        coord_pe = sinusoidal_coord_encoding(coords)  # [K, num_pts, 32]

        # 4. Conv1d: neighbor context → [K, num_pts, 224]
        x = coords.permute(0, 2, 1)              # [K, 2, num_pts]
        x = F.relu(self.conv1(x))                 # [K, 112, num_pts]
        x = F.relu(self.conv2(x))                 # [K, 224, num_pts]
        x = x.permute(0, 2, 1)                    # [K, num_pts, 224]
        x = self.conv_norm(x)

        # 5. Concat: sin_PE + conv1d
        geo_feats = torch.cat([coord_pe, x], dim=-1)  # [K, num_pts, 256]

        return geo_feats, coords


# ---------------------------------------------------------------------------
# Semantic Encoder
# ---------------------------------------------------------------------------
class SemanticEncoder(nn.Module):
    """Encode way_tags into [N, sem_dim] semantic features.

    Handles missing values via learnable mask tokens:
    - lanes=-1 → lanes_mask_token
    - width=-1.0 → width_mask_token
    - crossing uses highway_class=CROSSING_CLASS(11) with lanes/width masked
    """

    def __init__(self, embed_dims=256):
        super().__init__()
        self.embed_dims = embed_dims

        # Categorical → Embedding
        self.highway_embed = nn.Embedding(12, embed_dims)  # 0~10 vehicle ways, 11 crossing
        self.city_embed = nn.Embedding(4, embed_dims)      # 0~3

        # Continuous with -1 masking → MLP + learnable mask token
        self.lanes_proj = nn.Sequential(
            nn.Linear(1, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        self.lanes_mask_token = nn.Parameter(torch.randn(embed_dims))

        self.width_proj = nn.Sequential(
            nn.Linear(1, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, embed_dims),
        )
        self.width_mask_token = nn.Parameter(torch.randn(embed_dims))

        self.layer_norm = nn.LayerNorm(embed_dims)

    def forward(self, highway_class, lanes, width, city):
        """
        Args:
            highway_class: Tensor[N] int64
            lanes: Tensor[N] int64 (-1 = missing)
            width: Tensor[N] float32 (-1.0 = missing)
            city: Tensor[N] int64
        Returns:
            sem_feats: [N, embed_dims]
        """
        hw_feat = self.highway_embed(highway_class)  # [N, D]
        city_feat = self.city_embed(city)             # [N, D]

        # Lanes: -1 → mask token, else → MLP
        lanes_valid = (lanes != -1)
        lanes_input = lanes.float().unsqueeze(-1)
        lanes_proj = self.lanes_proj(lanes_input)
        lanes_feat = torch.where(
            lanes_valid.unsqueeze(-1),
            lanes_proj,
            self.lanes_mask_token.unsqueeze(0).expand(len(lanes), -1)
        )

        # Width: -1.0 → mask token, else → MLP
        width_valid = (width != -1.0)
        width_input = width.unsqueeze(-1)
        width_proj = self.width_proj(width_input)
        width_feat = torch.where(
            width_valid.unsqueeze(-1),
            width_proj,
            self.width_mask_token.unsqueeze(0).expand(len(width), -1)
        )

        # Sum fusion (4 features, all [N, D])
        sem_feats = hw_feat + city_feat + lanes_feat + width_feat
        sem_feats = self.layer_norm(sem_feats)
        return sem_feats


# ---------------------------------------------------------------------------
# SD Prior Encoder (main module)
# ---------------------------------------------------------------------------
class SDPriorEncoder(nn.Module):
    """Encode SD cache v3 into per-point SD tokens for decoder cross-attention.

    Each SD element (way or crossing) produces num_pts tokens.
    Total tokens = num_elements × num_pts, padded to max_sd_tokens.

    Args:
        embed_dims: total token dimension (512 = geo_dim + sem_dim)
        num_pts: points per element after resampling
        max_sd_tokens: max total tokens (max_elements × num_pts)
    """

    def __init__(self, embed_dims=512, num_pts=5, max_sd_tokens=125):
        super().__init__()
        geo_dim = embed_dims // 2   # 256
        sem_dim = embed_dims // 2   # 256
        self.embed_dims = embed_dims
        self.num_pts = num_pts
        self.max_sd_tokens = max_sd_tokens

        self.geo_encoder = GeometryEncoder(geo_dim, num_pts)
        self.sem_encoder = SemanticEncoder(sem_dim)

    def _build_crossing_pseudo_tags(self, M, city_values, device):
        """Build pseudo-semantic tags for crossings."""
        highway_class = torch.full((M,), CROSSING_CLASS, dtype=torch.long, device=device)
        lanes = torch.full((M,), -1, dtype=torch.long, device=device)
        width = torch.full((M,), -1.0, dtype=torch.float, device=device)
        city = city_values
        return highway_class, lanes, width, city

    def _to_tensor(self, arr, dtype, device):
        """Convert numpy array to tensor on device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(dtype=dtype, device=device)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    def forward(self, sd_cache_list, roi_size):
        """
        Args:
            sd_cache_list: List[dict], length B (batch). Each dict from sd_cache_v3.
            roi_size: (60, 30)
        Returns:
            sd_features:     [B, max_sd_tokens, embed_dims]  — per-point SD tokens
            sd_padding_mask: [B, max_sd_tokens]               — True = padding
            sd_coords:       [B, max_sd_tokens, 2]            — [0,1] normalized (for key_pos)
        """
        batch_feats = []
        batch_masks = []
        batch_coords = []
        device = self.geo_encoder.conv1.weight.device
        D = self.embed_dims

        for cache in sd_cache_list:
            if cache is None:
                # No SD data for this sample — keep 1st token valid (zero feature)
                # to prevent softmax NaN when all tokens are masked
                _mask = torch.ones(self.max_sd_tokens, dtype=torch.bool, device=device)
                _mask[0] = False  # dummy valid token (zero feature, no real effect)
                batch_feats.append(torch.zeros(self.max_sd_tokens, D, device=device))
                batch_masks.append(_mask)
                batch_coords.append(torch.zeros(self.max_sd_tokens, 2, device=device))
                continue

            way_geos = cache['way_geometry']
            cross_geos = cache['crossing_geometry']
            N = len(way_geos)
            M = len(cross_geos)
            E = N + M

            if E > 0:
                # Geometry: resample + Conv1d → per-point features
                all_geos = way_geos + cross_geos
                geo_feats, coords = self.geo_encoder(all_geos, roi_size)
                # geo_feats: [E, num_pts, geo_dim(256)], coords: [E, num_pts, 2]

                # Semantic: per-element → broadcast to per-point
                way_hw = self._to_tensor(cache['way_tags']['highway_class'], torch.long, device)
                way_lanes = self._to_tensor(cache['way_tags']['lanes'], torch.long, device)
                way_width = self._to_tensor(cache['way_tags']['width'], torch.float, device)
                way_city = self._to_tensor(cache['way_tags']['city'], torch.long, device)

                if M > 0:
                    cross_city = self._to_tensor(cache['crossing_city'], torch.long, device)
                    cross_hw, cross_lanes, cross_width, cross_city = \
                        self._build_crossing_pseudo_tags(M, cross_city, device)
                    all_hw = torch.cat([way_hw, cross_hw])
                    all_lanes = torch.cat([way_lanes, cross_lanes])
                    all_width = torch.cat([way_width, cross_width])
                    all_city = torch.cat([way_city, cross_city])
                else:
                    all_hw, all_lanes, all_width, all_city = \
                        way_hw, way_lanes, way_width, way_city

                sem_feats = self.sem_encoder(all_hw, all_lanes, all_width, all_city)  # [E, 256]

                # Broadcast sem to per-point
                sem_expanded = sem_feats.unsqueeze(1).expand(-1, self.num_pts, -1)  # [E, num_pts, 256]

                # Per-point fusion: concat [geo(256) | sem(256)] = [512]
                fused = torch.cat([geo_feats, sem_expanded], dim=-1)  # [E, num_pts, 512]

                # Flatten
                all_tokens = fused.reshape(-1, D)      # [E*num_pts, 512]
                all_coords = coords.reshape(-1, 2)     # [E*num_pts, 2]
            else:
                all_tokens = torch.zeros(0, D, device=device)
                all_coords = torch.zeros(0, 2, device=device)

            # Pad to max_sd_tokens
            T = all_tokens.shape[0]
            if T >= self.max_sd_tokens:
                all_tokens = all_tokens[:self.max_sd_tokens]
                all_coords = all_coords[:self.max_sd_tokens]
                mask = torch.zeros(self.max_sd_tokens, dtype=torch.bool, device=device)
            else:
                pad_t = torch.zeros(self.max_sd_tokens - T, D, device=device)
                all_tokens = torch.cat([all_tokens, pad_t])
                pad_c = torch.zeros(self.max_sd_tokens - T, 2, device=device)
                all_coords = torch.cat([all_coords, pad_c])
                mask = torch.zeros(self.max_sd_tokens, dtype=torch.bool, device=device)
                mask[T:] = True
                # Prevent all-True mask (causes softmax NaN in MHA)
                if T == 0:
                    mask[0] = False  # dummy valid token (zero feature)

            batch_feats.append(all_tokens)
            batch_masks.append(mask)
            batch_coords.append(all_coords)

        return (torch.stack(batch_feats),
                torch.stack(batch_masks),
                torch.stack(batch_coords))

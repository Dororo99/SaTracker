"""SatMapTracker: MapTracker with independent satellite BEV branch.

Clean implementation without SD map dependencies.
Uses AID4AD satellite imagery for BEV feature enhancement.

Training strategy:
  Stage 1 (BEV Pretrain):
    - Camera BEV → cam_seg_head → seg_loss_cam
    - Satellite BEV → sat_seg_head → seg_loss_sat
    - Independent training, no fusion.

  Stage 2 (Tracking Warmup):
    - Camera BEV only → vector head warmup (standard MapTracker)
    - Satellite encoder frozen/idle

  Stage 3 (Joint Finetuning):
    - Camera BEV + Satellite BEV → Fusion → fused_bev
    - Differentiated LRs: encoders small, fusion module full
"""
import numpy as np
import torch
import torch.nn as nn

from mmdet3d.models.builder import build_backbone, build_head, build_neck

from .base_mapper import MAPPERS
from .MapTracker import MapTracker

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def _bev_to_image(feat, reduce='norm'):
    """Convert BEV feature tensor to visualizable numpy image."""
    with torch.no_grad():
        if reduce == 'squeeze':
            img = feat.squeeze(0)
        else:
            img = feat.norm(dim=0)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img.cpu().numpy()


class _FusionNeckWrapper(nn.Module):
    """Wraps the original neck to inject satellite fusion.

    When sat_feats is set, fuses satellite BEV features into the output
    of the wrapped neck. Satellite images are ego-aligned to the current
    frame, so fusion is only applied on the current (last) frame to avoid
    spatial misalignment with previous frames.

    Usage:
        wrapper.sat_feats = sat_features
        wrapper.fuse_on_call = num_prev_frames  # fuse only on Nth call
        wrapper._call_count = 0
        # ... run forward (multiple neck calls) ...
        wrapper.sat_feats = None
    """

    def __init__(self, neck, fusion_module):
        super().__init__()
        self.neck = neck
        self.fusion_module = fusion_module
        self.sat_feats = None
        self.fusion_log = {}
        self.fuse_on_call = -1  # -1 = all calls, N = only Nth call
        self._call_count = 0

    def forward(self, x):
        out = self.neck(x)
        if self.sat_feats is not None and self.fusion_module is not None:
            should_fuse = (self.fuse_on_call < 0 or
                           self._call_count == self.fuse_on_call)
            if should_fuse:
                out, info = self.fusion_module(out, self.sat_feats)
                if info:
                    self.fusion_log.update(info)
        self._call_count += 1
        return out

    def init_weights(self):
        if hasattr(self.neck, 'init_weights'):
            self.neck.init_weights()


@MAPPERS.register_module()
class SatMapTracker(MapTracker):

    def __init__(self,
                 sat_encoder_cfg=None,
                 sat_seg_cfg=None,
                 sat_fusion_cfg=None,
                 use_sat_fusion=False,
                 freeze_sat_encoder=False,
                 vis_interval=500,
                 **kwargs):
        # Explicitly disable SD map features inherited from MapTracker
        kwargs.setdefault('use_sd_prior', False)
        kwargs.setdefault('use_osm_tile', False)
        kwargs.setdefault('sd_augment', False)

        super().__init__(**kwargs)

        # Satellite branch
        self.sat_encoder = None
        if sat_encoder_cfg is not None:
            self.sat_encoder = build_backbone(sat_encoder_cfg)

        # Satellite-only seg head (Stage 1: independent training)
        self.sat_seg_decoder = None
        if sat_seg_cfg is not None:
            self.sat_seg_decoder = build_head(sat_seg_cfg)

        # Fusion module (Stage 3)
        self.use_sat_fusion = use_sat_fusion
        if sat_fusion_cfg is not None and use_sat_fusion:
            fusion_module = build_neck(sat_fusion_cfg)
            self.neck = _FusionNeckWrapper(self.neck, fusion_module)

        # Optionally freeze satellite encoder (keeps BN in eval mode)
        if freeze_sat_encoder and self.sat_encoder is not None:
            self.sat_encoder.freeze()

        self.vis_interval = vis_interval
        self._vis_iter = 0

    def _encode_satellite(self, sat_img):
        """Encode satellite image to BEV features.

        The satellite encoder outputs features in image convention (y- at H=0),
        but BEVFormer uses inverted y-axis (y+ at H=0). Flip H to align with
        BEVFormer's BEV convention so that fusion and seg GT are consistent.
        """
        if self.sat_encoder is None or sat_img is None:
            return None
        feats = self.sat_encoder(sat_img)
        return torch.flip(feats, [2,])

    def _log_bev_metrics(self, log_vars, cam_bev, sat_feats):
        """Log BEV feature statistics for monitoring."""
        with torch.no_grad():
            log_vars['cam_bev_norm'] = cam_bev.norm(dim=1).mean().item()
            log_vars['sat_bev_norm'] = sat_feats.norm(dim=1).mean().item()

            # Cosine similarity between cam and sat features
            c_flat = cam_bev.flatten(2)
            s_flat = sat_feats.flatten(2)
            cos_sim = nn.functional.cosine_similarity(c_flat, s_flat, dim=1).mean()
            log_vars['cam_sat_cosine_sim'] = cos_sim.item()

            # Spatial analysis: far (top half = far from ego) vs near
            h = cam_bev.shape[2]
            log_vars['cam_bev_far_norm'] = cam_bev[:, :, :h//2, :].norm(dim=1).mean().item()
            log_vars['cam_bev_near_norm'] = cam_bev[:, :, h//2:, :].norm(dim=1).mean().item()
            log_vars['sat_bev_far_norm'] = sat_feats[:, :, :h//2, :].norm(dim=1).mean().item()
            log_vars['sat_bev_near_norm'] = sat_feats[:, :, h//2:, :].norm(dim=1).mean().item()

    def _log_wandb_images(self, cam_bev, sat_feats, sat_img, semantic_mask,
                          cam_seg_preds=None, sat_seg_preds=None):
        """Log BEV feature visualizations and seg predictions to wandb."""
        if not HAS_WANDB or wandb.run is None:
            return

        self._vis_iter += 1
        if self._vis_iter % self.vis_interval != 0:
            return

        log_dict = {}

        # BEV feature norm maps
        log_dict['bev/cam_bev'] = wandb.Image(
            _bev_to_image(cam_bev[0]), caption='Camera BEV feature norm')
        log_dict['bev/sat_bev'] = wandb.Image(
            _bev_to_image(sat_feats[0]), caption='Satellite BEV feature norm')

        # Feature difference map
        with torch.no_grad():
            diff = ((cam_bev[0] - sat_feats[0]) ** 2).mean(dim=0)
            diff = diff - diff.min()
            if diff.max() > 0:
                diff = diff / diff.max()
            log_dict['bev/cam_sat_diff'] = wandb.Image(
                diff.cpu().numpy(), caption='Camera vs Satellite L2 diff')

        # Satellite input image (denormalize)
        if sat_img is not None:
            with torch.no_grad():
                sat_vis = sat_img[0].cpu().float()
                sat_vis = sat_vis * torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) + \
                          torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                sat_vis = torch.flip(sat_vis, [1,])
                sat_vis = sat_vis.clamp(0, 255).byte().permute(1, 2, 0).numpy()
                log_dict['input/sat_image'] = wandb.Image(
                    sat_vis, caption='Satellite Input (AID4AD)')

        # Seg GT
        if semantic_mask is not None:
            with torch.no_grad():
                seg_vis = torch.flip(semantic_mask[0], [1,]).cpu().float()
                seg_rgb = seg_vis.permute(1, 2, 0).numpy()
                log_dict['input/seg_gt'] = wandb.Image(
                    seg_rgb, caption='Seg GT (R=ped, G=div, B=bound)')

        # Camera seg prediction (already in BEVFormer inverted convention — no flip needed)
        if cam_seg_preds is not None:
            with torch.no_grad():
                pred_vis = cam_seg_preds[0].sigmoid().cpu().float()
                pred_rgb = pred_vis.permute(1, 2, 0).numpy()
                log_dict['pred/seg_cam'] = wandb.Image(
                    pred_rgb, caption='Seg Pred (Camera)')

        # Satellite seg prediction (already in BEVFormer inverted convention — no flip needed)
        if sat_seg_preds is not None:
            with torch.no_grad():
                pred_vis = sat_seg_preds[0].sigmoid().cpu().float()
                pred_rgb = pred_vis.permute(1, 2, 0).numpy()
                log_dict['pred/seg_sat'] = wandb.Image(
                    pred_rgb, caption='Seg Pred (Satellite)')

                # Overlay satellite seg pred on satellite image
                if sat_img is not None:
                    sat_bg = sat_img[0].cpu().float()
                    sat_bg = sat_bg * torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) + \
                             torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                    sat_bg = torch.flip(sat_bg, [1,])
                    sat_bg = sat_bg.clamp(0, 255).byte().permute(1, 2, 0).numpy().astype(np.float32)
                    pred_mask = (pred_rgb * 255).astype(np.float32)
                    # Resize pred_mask to match sat_bg if sizes differ
                    if pred_mask.shape[:2] != sat_bg.shape[:2]:
                        import cv2
                        pred_mask = cv2.resize(pred_mask, (sat_bg.shape[1], sat_bg.shape[0]),
                                               interpolation=cv2.INTER_LINEAR)
                    fg = pred_mask.sum(axis=2) > 30
                    overlay = sat_bg.copy()
                    overlay[fg] = sat_bg[fg] * 0.4 + pred_mask[fg] * 0.6
                    log_dict['pred/overlay_sat_seg_on_sat'] = wandb.Image(
                        overlay.clip(0, 255).astype(np.uint8),
                        caption='Sat Seg Pred overlaid on Satellite')

        wandb.log(log_dict, commit=False)

    def forward_train(self, img, vectors, semantic_mask, points=None,
                      img_metas=None, all_prev_data=None,
                      all_local2global_info=None, sat_img=None, **kwargs):
        has_sat = (sat_img is not None and self.sat_encoder is not None)

        # ── Stage 1: Independent satellite pretraining ──
        if has_sat and self.skip_vector_head and not self.use_sat_fusion:
            # Run base MapTracker forward (camera BEV + camera seg)
            loss, log_vars, num_sample = super().forward_train(
                img, vectors, semantic_mask, points=points,
                img_metas=img_metas, all_prev_data=all_prev_data,
                all_local2global_info=all_local2global_info, **kwargs)

            # Train satellite encoder independently with its own seg head
            if self.sat_seg_decoder is not None:
                sat_feats = self._encode_satellite(sat_img)
                # Flip GT to match BEVFormer convention (same as camera seg in MapTracker L573)
                sat_gt = torch.flip(semantic_mask, [2,])

                sat_seg_preds, _, sat_seg_loss, sat_seg_dice = \
                    self.sat_seg_decoder(sat_feats, sat_gt, None, return_loss=True)

                loss = loss + sat_seg_loss + sat_seg_dice
                log_vars['sat_seg'] = sat_seg_loss.item()
                log_vars['sat_seg_dice'] = sat_seg_dice.item()
                log_vars['total'] = loss.item()

                # Log BEV feature metrics
                cam_bev = self._last_bev_feats
                self._log_bev_metrics(log_vars, cam_bev, sat_feats)

                # Log camera seg preds (from parent forward, stored in _last_bev_feats)
                with torch.no_grad():
                    cam_seg_preds, _ = self.seg_decoder(
                        bev_features=cam_bev, return_loss=False)

                # Wandb image logging
                self._log_wandb_images(
                    cam_bev, sat_feats, sat_img, semantic_mask,
                    cam_seg_preds=cam_seg_preds, sat_seg_preds=sat_seg_preds)

            return loss, log_vars, num_sample

        # ── Stage 3: Fused forward via _FusionNeckWrapper ──
        if has_sat and self.use_sat_fusion:
            sat_feats = self._encode_satellite(sat_img)

            assert isinstance(self.neck, _FusionNeckWrapper), \
                'use_sat_fusion=True requires sat_fusion_cfg to be set'

            # Only fuse on the CURRENT (last) frame — satellite is ego-aligned
            # to current pose, previous frames have different ego poses.
            num_prev = len(all_prev_data) if all_prev_data is not None else 0
            self.neck.sat_feats = sat_feats
            self.neck.fusion_log = {}
            self.neck.fuse_on_call = num_prev  # Nth call = current frame
            self.neck._call_count = 0

            loss, log_vars, num_sample = super().forward_train(
                img, vectors, semantic_mask, points=points,
                img_metas=img_metas, all_prev_data=all_prev_data,
                all_local2global_info=all_local2global_info, **kwargs)

            # Collect fusion metrics
            if self.neck.fusion_log:
                for k, v in self.neck.fusion_log.items():
                    log_vars[f'fusion/{k}'] = v
            with torch.no_grad():
                log_vars['sat_bev_norm'] = sat_feats.norm(dim=1).mean().item()

            log_vars['total'] = loss.item()

            self.neck.sat_feats = None
            self.neck.fusion_log = {}

            return loss, log_vars, num_sample

        # ── Stage 2 or no satellite: standard MapTracker ──
        return super().forward_train(
            img, vectors, semantic_mask, points=points,
            img_metas=img_metas, all_prev_data=all_prev_data,
            all_local2global_info=all_local2global_info, **kwargs)

    def forward_test(self, img, points=None, img_metas=None, sat_img=None, **kwargs):
        """Test-time forward with optional satellite fusion."""
        if (sat_img is not None and self.sat_encoder is not None
                and self.use_sat_fusion
                and isinstance(self.neck, _FusionNeckWrapper)):
            sat_feats = self._encode_satellite(sat_img)
            self.neck.sat_feats = sat_feats
            self.neck.fuse_on_call = 0  # Test: single neck call per frame
            self.neck._call_count = 0
            result = super().forward_test(img, points=points,
                                          img_metas=img_metas, **kwargs)
            self.neck.sat_feats = None
            return result

        return super().forward_test(img, points=points,
                                    img_metas=img_metas, **kwargs)

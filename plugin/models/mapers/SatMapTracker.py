"""SatMapTracker: Camera-only temporal context + current-frame satellite fusion.

Temporal history carries pure cam_bev features; satellite fusion happens only
at the current frame through a dedicated fused decoder. This isolates the
camera encoder from fusion-layer drift and keeps the fused branch as a
"sparse delta" learned on top of cam.

Training flow (history_mode='cam'):
  For each frame t in temporal loop:
    backbone(cam_t, history=[cam_{<t}]) → cam_bev_t
    parent's seg_decoder(cam_bev_t) → seg_loss  (cam branch, dense over frames)
    history.append(cam_bev_t)
  Extra (current frame t=T only):
    sat_encoder(sat_img_T) → sat_bev_T
    ConvFuser(cam_bev_T, sat_bev_T) → fused_bev_T → seg_decoder_fused (main eval)
    sat_bev_T → seg_decoder_sat (auxiliary sat supervision)

Decoder assignment:
  self.seg_decoder        (parent) : cam branch  — dense supervision from MapTracker loop
  self.seg_decoder_fused  (new)    : fused branch — current frame only, main eval metric
  self.seg_decoder_sat    (kept)   : sat branch  — current frame only, auxiliary

Encoder: SatelliteEncoder (ResNet50 + FPN)
Fusion:  Configurable (SatelliteConvFuser / SatCamConvFusion / etc.)
"""
import torch
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import copy
from mmdet3d.models.builder import build_backbone, build_head, build_neck

from .base_mapper import MAPPERS
from .MapTracker import MapTracker


@MAPPERS.register_module()
class SatMapTracker(MapTracker):

    def __init__(self,
                 sat_encoder_cfg=None,
                 sat_fusion_cfg=None,
                 freeze_sat_encoder=False,
                 history_mode='cam',
                 seg_cfg=None,
                 **kwargs):
        kwargs.setdefault('use_sd_prior', False)
        kwargs.setdefault('use_osm_tile', False)
        kwargs.setdefault('sd_augment', False)

        super().__init__(seg_cfg=seg_cfg, **kwargs)

        # Satellite encoder (ResNet50 + FPN)
        self.sat_encoder = None
        if sat_encoder_cfg is not None:
            self.sat_encoder = build_backbone(sat_encoder_cfg)

        # ConvFuser (concat + conv) — applied at current frame only
        self.sat_fusion = None
        if sat_fusion_cfg is not None:
            self.sat_fusion = build_neck(sat_fusion_cfg)

        if freeze_sat_encoder and self.sat_encoder is not None:
            self.sat_encoder.freeze()

        # Only 'cam' mode is supported: temporal history carries pure cam_bev,
        # fusion happens at the current frame through a dedicated fused decoder.
        assert history_mode == 'cam', \
            f"history_mode must be 'cam' (got '{history_mode}'). " \
            f"'fused' mode is deprecated — it caused temporal-drift of the fusion " \
            f"layer to bleed into the camera encoder through TemporalSelfAttention."
        self.history_mode = history_mode

        # Decoder assignment:
        #   self.seg_decoder         (from parent) : cam branch — dense supervision
        #   self.seg_decoder_fused   (new)         : fused branch — current frame only, eval main
        #   self.seg_decoder_sat     (kept)        : sat branch  — current frame only
        if seg_cfg is not None:
            self.seg_decoder_fused = build_head(copy.deepcopy(seg_cfg))
            self.seg_decoder_sat = build_head(copy.deepcopy(seg_cfg))
        else:
            self.seg_decoder_fused = None
            self.seg_decoder_sat = None

    def _encode_satellite(self, sat_img):
        """Encode satellite image to BEV features.

        H-flip aligns with BEVFormer's inverted y-axis convention.
        """
        if self.sat_encoder is None or sat_img is None:
            return None
        feats = self.sat_encoder(sat_img)
        return torch.flip(feats, [2,])

    # NOTE: _post_backbone_hook is NOT overridden.
    # In cam mode, the temporal loop must pass through pure camera BEV features
    # (parent's default hook is an identity passthrough). Fusion happens only
    # at the current frame inside forward_train/forward_test, using
    # self._last_bev_feats (stored by parent after neck).

    def forward_train(self, img, vectors, semantic_mask, points=None,
                      img_metas=None, all_prev_data=None,
                      all_local2global_info=None, sat_img=None, **kwargs):
        """Cam-mode training.

        Step 1: parent.forward_train runs the full temporal loop with pure
                cam_bev features (hook is passthrough). Parent applies
                seg_decoder(cam_bev_t) at every frame — this IS the cam
                branch's dense supervision, no extra code needed.
        Step 2: At the current frame only, we add two more supervision
                branches using separate decoders:
                - Fused: ConvFuser(cam_bev_T, sat_bev_T) → seg_decoder_fused
                - Sat:                          sat_bev_T → seg_decoder_sat

        After parent returns, self._last_bev_feats holds the current frame's
        post-neck cam BEV (stored by parent at MapTracker.forward_train:743).
        """
        # Step 1: parent does the cam-only temporal loop + dense cam seg_loss.
        loss, log_vars, num_sample = super().forward_train(
            img, vectors, semantic_mask, points=points,
            img_metas=img_metas, all_prev_data=all_prev_data,
            all_local2global_info=all_local2global_info, **kwargs)

        # Step 2: extra supervision at current frame only.
        if (self.sat_encoder is None or self.sat_fusion is None
                or sat_img is None):
            return loss, log_vars, num_sample

        sat_bev = self._encode_satellite(sat_img)
        if sat_bev is None:
            return loss, log_vars, num_sample

        if sat_bev.shape[0] != img.shape[0]:
            sat_bev = sat_bev.expand(img.shape[0], -1, -1, -1)

        cam_bev = self._last_bev_feats  # current frame, post-neck
        gt_semantic = torch.flip(semantic_mask, [2,])

        # Fused branch — main eval metric
        fused_bev, _ = self.sat_fusion(cam_bev, sat_bev)
        _, _, seg_loss_fused, seg_dice_fused = self.seg_decoder_fused(
            fused_bev, gt_semantic, None, return_loss=True)
        loss = loss + seg_loss_fused + seg_dice_fused
        log_vars['seg_fused'] = seg_loss_fused.item()
        log_vars['seg_dice_fused'] = seg_dice_fused.item()

        # Sat branch — auxiliary
        _, _, seg_loss_sat, seg_dice_sat = self.seg_decoder_sat(
            sat_bev, gt_semantic, None, return_loss=True)
        loss = loss + seg_loss_sat + seg_dice_sat
        log_vars['seg_sat'] = seg_loss_sat.item()
        log_vars['seg_dice_sat'] = seg_dice_sat.item()

        log_vars['total'] = loss.item()
        return loss, log_vars, num_sample

    def _seg_preds_to_mask(self, seg_preds, thr=0.4):
        """Convert seg_decoder output to uint8 mask."""
        pred = seg_preds[0]
        tmp_scores, tmp_labels = pred.max(0)
        tmp_scores = tmp_scores.sigmoid()
        mask = torch.zeros(tmp_labels.shape, dtype=torch.uint8, device=pred.device)
        pos = tmp_scores >= thr
        mask[pos] = tmp_labels[pos].type(torch.uint8) + 1
        return mask.cpu().numpy()

    def _bev_to_image(self, bev_feat):
        """Convert BEV feature [C,H,W] to normalized grayscale image for wandb."""
        with torch.no_grad():
            img = bev_feat.norm(dim=0)
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()
            return img.cpu().numpy()

    def _seg_to_rgb(self, seg_preds):
        """Convert seg_decoder output [3,H,W] to RGB image for wandb."""
        with torch.no_grad():
            pred_vis = seg_preds[0].sigmoid().cpu().float()
            return pred_vis.permute(1, 2, 0).numpy()

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, sat_img=None, **kwargs):
        """Cam-mode test.

        Parent.forward_test runs the backbone on pure cam features (hook is
        passthrough) and writes results_list[0]['semantic_mask'] from
        self.seg_decoder(cam_bev). Because cam mode is in effect, that
        semantic_mask IS the cam-branch prediction.

        We then:
          1. Move parent's semantic_mask → semantic_mask_cam
          2. Compute the fused prediction with seg_decoder_fused and
             overwrite semantic_mask (main eval metric)
          3. Compute the sat-only prediction with seg_decoder_sat → semantic_mask_sat
        """
        results_list = super().forward_test(
            img, points=points, img_metas=img_metas, **kwargs)

        if (self.sat_encoder is None or self.sat_fusion is None
                or sat_img is None):
            return results_list

        sat_bev = self._encode_satellite(sat_img)
        if sat_bev is None:
            return results_list

        cam_bev = self._last_bev_feats  # parent stored post-neck cam BEV
        if sat_bev.shape[0] != cam_bev.shape[0]:
            sat_bev = sat_bev.expand(cam_bev.shape[0], -1, -1, -1)

        # 1. Relabel parent's cam prediction
        if 'semantic_mask' in results_list[0]:
            results_list[0]['semantic_mask_cam'] = results_list[0]['semantic_mask']

        # 2. Fused prediction (main eval)
        fused_bev, _ = self.sat_fusion(cam_bev, sat_bev)
        seg_preds_fused, _ = self.seg_decoder_fused(
            bev_features=fused_bev, return_loss=False)
        results_list[0]['semantic_mask'] = self._seg_preds_to_mask(seg_preds_fused)

        # 3. Sat prediction
        seg_preds_sat, _ = self.seg_decoder_sat(
            bev_features=sat_bev, return_loss=False)
        results_list[0]['semantic_mask_sat'] = self._seg_preds_to_mask(seg_preds_sat)

        # Wandb visualization (every 50 samples)
        if HAS_WANDB and wandb.run is not None:
            self._test_vis_count = getattr(self, '_test_vis_count', 0) + 1
            if self._test_vis_count % 50 == 1:
                # Recompute raw cam seg_preds for heatmap visualization
                seg_preds_cam, _ = self.seg_decoder(
                    bev_features=cam_bev, return_loss=False)
                log_dict = {
                    'test/bev_cam':   wandb.Image(self._bev_to_image(cam_bev[0]),   caption='Camera BEV'),
                    'test/bev_sat':   wandb.Image(self._bev_to_image(sat_bev[0]),   caption='Satellite BEV'),
                    'test/bev_fused': wandb.Image(self._bev_to_image(fused_bev[0]), caption='Fused BEV'),
                    'test/seg_cam':   wandb.Image(self._seg_to_rgb(seg_preds_cam),   caption='Seg Cam'),
                    'test/seg_sat':   wandb.Image(self._seg_to_rgb(seg_preds_sat),   caption='Seg Sat'),
                    'test/seg_fused': wandb.Image(self._seg_to_rgb(seg_preds_fused), caption='Seg Fused'),
                }
                wandb.log(log_dict, commit=False)

        return results_list

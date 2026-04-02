"""SatMapTracker v6/v7: ConvFuser + Triple Supervision.

Per-frame satellite fusion: each temporal frame uses its own satellite image
(spatially aligned to that frame's ego pose) for fusion.

Training flow (history_mode='fused'):
  For each frame t in temporal loop:
    backbone(cam_t, history) → cam_bev_t
    hook: ConvFuser(cam_bev_t, sat_bev_t) → fused_bev_t  (per-frame aligned sat)
    seg_decoder(fused_bev_t) → seg_loss
    history.append(fused_bev_t)
  Extra (current frame only):
    cam_bev → seg_decoder, sat_bev → seg_decoder

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

from mmdet3d.models.builder import build_backbone, build_neck

from .base_mapper import MAPPERS
from .MapTracker import MapTracker


@MAPPERS.register_module()
class SatMapTracker(MapTracker):

    def __init__(self,
                 sat_encoder_cfg=None,
                 sat_fusion_cfg=None,
                 freeze_sat_encoder=False,
                 history_mode='fused',
                 **kwargs):
        kwargs.setdefault('use_sd_prior', False)
        kwargs.setdefault('use_osm_tile', False)
        kwargs.setdefault('sd_augment', False)

        super().__init__(**kwargs)

        # Satellite encoder (ResNet50 + FPN)
        self.sat_encoder = None
        if sat_encoder_cfg is not None:
            self.sat_encoder = build_backbone(sat_encoder_cfg)

        # ConvFuser (concat + conv)
        self.sat_fusion = None
        if sat_fusion_cfg is not None:
            self.sat_fusion = build_neck(sat_fusion_cfg)

        if freeze_sat_encoder and self.sat_encoder is not None:
            self.sat_encoder.freeze()

        assert history_mode in ('fused', 'cam'), \
            f"history_mode must be 'fused' or 'cam', got '{history_mode}'"
        self.history_mode = history_mode

        # Internal state (per-forward, cleaned up in finally)
        self._sat_feats_list = []   # per-frame sat features [prev0, prev1, ..., current]
        self._hook_frame_idx = 0    # counter for hook calls
        self._cached_cam_bev = None

    def _encode_satellite(self, sat_img):
        """Encode satellite image to BEV features.

        H-flip aligns with BEVFormer's inverted y-axis convention.
        """
        if self.sat_encoder is None or sat_img is None:
            return None
        feats = self.sat_encoder(sat_img)
        return torch.flip(feats, [2,])

    def _post_backbone_hook(self, bev_feats):
        """Cache cam_bev. When history_mode='fused', fuse with the
        per-frame aligned satellite features."""
        self._cached_cam_bev = bev_feats

        if (self.history_mode == 'fused'
                and self._hook_frame_idx < len(self._sat_feats_list)
                and self.sat_fusion is not None):
            sat = self._sat_feats_list[self._hook_frame_idx]
            self._hook_frame_idx += 1
            if sat is not None:
                if sat.shape[0] != bev_feats.shape[0]:
                    sat = sat.expand(bev_feats.shape[0], -1, -1, -1)
                fused, _ = self.sat_fusion(bev_feats, sat)
                return fused

        self._hook_frame_idx += 1
        return bev_feats

    def forward_train(self, img, vectors, semantic_mask, points=None,
                      img_metas=None, all_prev_data=None,
                      all_local2global_info=None, sat_img=None, **kwargs):
        # Encode per-frame satellite images
        self._sat_feats_list = []
        self._hook_frame_idx = 0
        self._cached_cam_bev = None

        if self.sat_encoder is not None:
            # Prev frames
            if all_prev_data is not None:
                for prev_data in all_prev_data:
                    prev_sat = prev_data.get('sat_img', None)
                    if prev_sat is not None:
                        if hasattr(prev_sat, 'data'):
                            prev_sat = prev_sat.data
                        prev_sat = prev_sat.to(img.device)
                    self._sat_feats_list.append(
                        self._encode_satellite(prev_sat))
            # Current frame
            self._sat_feats_list.append(self._encode_satellite(sat_img))

        try:
            loss, log_vars, num_sample = super().forward_train(
                img, vectors, semantic_mask, points=points,
                img_metas=img_metas, all_prev_data=all_prev_data,
                all_local2global_info=all_local2global_info, **kwargs)

            # --- Extra supervision branches (current frame only) ---
            current_sat_feats = self._sat_feats_list[-1] if self._sat_feats_list else None
            if current_sat_feats is not None and self.sat_fusion is not None:
                gt_semantic = torch.flip(semantic_mask, [2,])
                sat_bev = current_sat_feats
                if sat_bev.shape[0] != img.shape[0]:
                    sat_bev = sat_bev.expand(img.shape[0], -1, -1, -1)

                if self.history_mode == 'fused':
                    cam_bev = self._cached_cam_bev
                    if cam_bev is not None:
                        _, _, seg_loss_cam, seg_dice_cam = self.seg_decoder(
                            cam_bev, gt_semantic, None, return_loss=True)
                        loss = loss + seg_loss_cam + seg_dice_cam
                        log_vars['seg_cam'] = seg_loss_cam.item()
                        log_vars['seg_dice_cam'] = seg_dice_cam.item()
                else:
                    cam_bev = self._last_bev_feats
                    fused_bev, _ = self.sat_fusion(cam_bev, sat_bev)
                    _, _, seg_loss_fused, seg_dice_fused = self.seg_decoder(
                        fused_bev, gt_semantic, None, return_loss=True)
                    loss = loss + seg_loss_fused + seg_dice_fused
                    log_vars['seg_fused'] = seg_loss_fused.item()
                    log_vars['seg_dice_fused'] = seg_dice_fused.item()

                # sat_bev supervision (always)
                _, _, seg_loss_sat, seg_dice_sat = self.seg_decoder(
                    sat_bev, gt_semantic, None, return_loss=True)
                loss = loss + seg_loss_sat + seg_dice_sat
                log_vars['seg_sat'] = seg_loss_sat.item()
                log_vars['seg_dice_sat'] = seg_dice_sat.item()

                log_vars['total'] = loss.item()

            return loss, log_vars, num_sample
        finally:
            self._sat_feats_list = []
            self._hook_frame_idx = 0
            self._cached_cam_bev = None

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
        # Test: single frame, single sat_img
        self._sat_feats_list = []
        self._hook_frame_idx = 0
        self._cached_cam_bev = None

        if self.sat_encoder is not None and sat_img is not None:
            self._sat_feats_list.append(self._encode_satellite(sat_img))

        try:
            # Parent forward_test: uses fused_bev (history_mode='fused') or cam_bev
            results_list = super().forward_test(
                img, points=points, img_metas=img_metas, **kwargs)

            sat_bev = self._sat_feats_list[-1] if self._sat_feats_list else None
            cam_bev = self._cached_cam_bev

            # --- Generate all three seg predictions ---
            if sat_bev is not None and cam_bev is not None and self.sat_fusion is not None:
                if sat_bev.shape[0] != cam_bev.shape[0]:
                    sat_bev = sat_bev.expand(cam_bev.shape[0], -1, -1, -1)

                # Fused
                fused_bev, _ = self.sat_fusion(cam_bev, sat_bev)
                seg_preds_fused, _ = self.seg_decoder(bev_features=fused_bev, return_loss=False)

                # Cam only
                seg_preds_cam, _ = self.seg_decoder(bev_features=cam_bev, return_loss=False)

                # Sat only
                seg_preds_sat, _ = self.seg_decoder(bev_features=sat_bev, return_loss=False)

                # Main eval uses fused
                results_list[0]['semantic_mask'] = self._seg_preds_to_mask(seg_preds_fused)
                results_list[0]['semantic_mask_cam'] = self._seg_preds_to_mask(seg_preds_cam)
                results_list[0]['semantic_mask_sat'] = self._seg_preds_to_mask(seg_preds_sat)

                # Wandb logging (every 50 samples)
                if HAS_WANDB and wandb.run is not None:
                    self._test_vis_count = getattr(self, '_test_vis_count', 0) + 1
                    if self._test_vis_count % 50 == 1:
                        log_dict = {}
                        # BEV features
                        log_dict['test/bev_cam'] = wandb.Image(
                            self._bev_to_image(cam_bev[0]), caption='Camera BEV')
                        log_dict['test/bev_sat'] = wandb.Image(
                            self._bev_to_image(sat_bev[0]), caption='Satellite BEV')
                        log_dict['test/bev_fused'] = wandb.Image(
                            self._bev_to_image(fused_bev[0]), caption='Fused BEV')
                        # Seg predictions
                        log_dict['test/seg_cam'] = wandb.Image(
                            self._seg_to_rgb(seg_preds_cam), caption='Seg Cam')
                        log_dict['test/seg_sat'] = wandb.Image(
                            self._seg_to_rgb(seg_preds_sat), caption='Seg Sat')
                        log_dict['test/seg_fused'] = wandb.Image(
                            self._seg_to_rgb(seg_preds_fused), caption='Seg Fused')
                        # GT (from parent's result)
                        gt_mask = results_list[0].get('semantic_mask', None)
                        if gt_mask is not None:
                            log_dict['test/seg_gt'] = wandb.Image(
                                gt_mask.astype(np.float32) / max(gt_mask.max(), 1),
                                caption='GT Mask')
                        wandb.log(log_dict, commit=False)

            return results_list
        finally:
            self._sat_feats_list = []
            self._hook_frame_idx = 0
            self._cached_cam_bev = None

"""
SatMapTracker: MapTracker with Satellite Cross-Attention in BEVFormer encoder.

Extends MapTracker to:
  1. Load satellite images via SatMAE encoder (frozen ViT-L)
  2. Pass satellite tokens to SatBEVFormerBackbone for cross-attention
  3. Everything else (TemporalNet, seg head, vector head) unchanged
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import build_backbone
from .base_mapper import MAPPERS
from .MapTracker import MapTracker
from ..backbones.conv_fusion import SatConvFusion


@MAPPERS.register_module()
class SatMapTracker(MapTracker):

    def __init__(self,
                 sat_encoder_cfg=None,
                 conv_fusion_cfg=None,
                 sat_prior_cfg=None,
                 use_sat_backbone_fusion=True,
                 use_sat_prior=False,
                 use_sat_class_range_gate=False,
                 prior_gate_cfg=None,
                 use_sat_prior_warp=False,
                 sat_prior_warp_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_sat_backbone_fusion = bool(use_sat_backbone_fusion)
        self.use_sat_prior = bool(use_sat_prior)
        self.use_sat_class_range_gate = bool(use_sat_class_range_gate)
        self.use_sat_prior_warp = bool(use_sat_prior_warp)

        if sat_encoder_cfg is not None:
            self.sat_encoder = build_backbone(sat_encoder_cfg)
        else:
            self.sat_encoder = None

        if conv_fusion_cfg is not None:
            self.conv_fusion = SatConvFusion(**conv_fusion_cfg)
        else:
            self.conv_fusion = None

        if sat_prior_cfg is not None:
            self.sat_prior = build_backbone(sat_prior_cfg)
        else:
            self.sat_prior = None

        gate_cfg = prior_gate_cfg or {}
        n_cls = int(getattr(self.seg_decoder, 'num_classes', 3))
        class_logit_init = gate_cfg.get('class_logit_init', None)
        if class_logit_init is None:
            class_logit_init = [-2.0] + [0.0] * max(0, n_cls - 1)
        class_logit_init = [float(v) for v in class_logit_init]
        if len(class_logit_init) < n_cls:
            class_logit_init = class_logit_init + [0.0] * (n_cls - len(class_logit_init))
        elif len(class_logit_init) > n_cls:
            class_logit_init = class_logit_init[:n_cls]
        self.sat_prior_class_gate = nn.Parameter(torch.tensor(class_logit_init))
        self.sat_prior_range_center = float(gate_cfg.get('range_center', 12.0))
        self.sat_prior_range_scale = float(gate_cfg.get('range_scale', 4.0))

        warp_cfg = sat_prior_warp_cfg or {}
        self.sat_prior_warp_offset_scale = float(warp_cfg.get('offset_scale', 0.05))
        self.sat_prior_warp_reg_weight = float(warp_cfg.get('offset_reg_weight', 0.01))
        if self.use_sat_prior_warp:
            hidden = int(warp_cfg.get('hidden_channels', 64))
            self.sat_prior_warp_head = nn.Sequential(
                nn.Conv2d(n_cls * 2, hidden, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
            )
        else:
            self.sat_prior_warp_head = None

    def _encode_satellite(self, sat_img):
        if self.sat_encoder is None or sat_img is None:
            return None, None
        sat_tokens, grid_size = self.sat_encoder(sat_img)
        return sat_tokens, grid_size

    def _apply_conv_fusion(self, bev_feats, sat_tokens, sat_grid_size):
        if self.conv_fusion is not None and sat_tokens is not None:
            return self.conv_fusion(bev_feats, sat_tokens, sat_grid_size)
        return bev_feats

    def _sat_tokens_for_backbone(self, sat_tokens, sat_grid_size):
        if not self.use_sat_backbone_fusion:
            return None, None
        return sat_tokens, sat_grid_size

    def _build_class_range_alpha(self, seg_preds):
        if not self.use_sat_class_range_gate:
            return None, None

        _, num_classes, h, w = seg_preds.shape
        device = seg_preds.device
        dtype = seg_preds.dtype

        class_gate = torch.sigmoid(self.sat_prior_class_gate[:num_classes]).to(
            device=device, dtype=dtype).view(1, num_classes, 1, 1)

        roi_x = float(self.roi_size[0])
        roi_y = float(self.roi_size[1])
        xs = torch.linspace(-roi_x / 2.0, roi_x / 2.0, w, device=device, dtype=dtype)
        ys = torch.linspace(roi_y / 2.0, -roi_y / 2.0, h, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs)
        dist = torch.sqrt(xx * xx + yy * yy).view(1, 1, h, w)

        scale = max(self.sat_prior_range_scale, 1e-6)
        range_alpha = torch.sigmoid((dist - self.sat_prior_range_center) / scale)
        alpha = class_gate * range_alpha

        stats = {
            'sat_prior_range_alpha_mean': range_alpha.mean().item(),
        }
        for idx in range(num_classes):
            stats[f'sat_prior_class_gate_c{idx}'] = class_gate[0, idx, 0, 0].item()
        return alpha, stats

    def _warp_sat_prior(self, seg_preds, delta_logits, u_sat):
        if not self.use_sat_prior_warp:
            return delta_logits, u_sat, None, None
        if self.sat_prior_warp_head is None:
            raise RuntimeError(
                'use_sat_prior_warp=True but sat_prior_warp_head is not initialized. '
                'Check sat_prior_warp_cfg.')

        bs, _, h, w = delta_logits.shape
        warp_in = torch.cat([seg_preds, delta_logits], dim=1)
        offsets = torch.tanh(self.sat_prior_warp_head(warp_in)) * self.sat_prior_warp_offset_scale

        theta = torch.eye(2, 3, device=delta_logits.device, dtype=delta_logits.dtype).unsqueeze(0).repeat(bs, 1, 1)
        base_grid = F.affine_grid(theta, size=delta_logits.shape, align_corners=False)
        grid = torch.clamp(base_grid + offsets.permute(0, 2, 3, 1), min=-1.0, max=1.0)

        if not getattr(self, '_sat_prior_warp_identity_checked', False):
            with torch.no_grad():
                zero_grid = base_grid
                probe = F.grid_sample(
                    delta_logits.detach(), zero_grid,
                    mode='bilinear', padding_mode='border', align_corners=False)
                diff = (probe - delta_logits.detach()).abs().mean().item()
                if diff > 1e-4:
                    raise RuntimeError(
                        f'Sat prior warp identity check failed (mean abs diff={diff:.6f}). '
                        'Grid construction is inconsistent with grid_sample align_corners=False.')
            self._sat_prior_warp_identity_checked = True

        delta_warp = F.grid_sample(
            delta_logits, grid, mode='bilinear', padding_mode='border', align_corners=False)
        u_warp = F.grid_sample(
            u_sat, grid, mode='bilinear', padding_mode='border', align_corners=False)

        reg = self.sat_prior_warp_reg_weight * (offsets ** 2).mean()
        stats = {
            'sat_prior_warp_offset_abs_mean': offsets.abs().mean().item(),
        }
        return delta_warp, u_warp, reg, stats

    def _apply_sat_prior_correction(self, seg_preds, sat_tokens, sat_grid_size):
        """Residual logit correction: S_final = S_cam + U_sat * DeltaS_sat."""
        if not self.use_sat_prior:
            return seg_preds, None, None
        if self.sat_prior is None:
            raise RuntimeError(
                'use_sat_prior=True but sat_prior is not initialized. '
                'Check model.sat_prior_cfg.')
        if sat_tokens is None:
            raise RuntimeError(
                'use_sat_prior=True but satellite tokens are None. '
                'Check sat_img pipeline and sat_encoder_cfg.')

        delta_logits, u_sat = self.sat_prior(
            sat_tokens=sat_tokens,
            sat_grid_size=sat_grid_size,
            target_hw=seg_preds.shape[-2:])
        delta_logits, u_sat, warp_reg, warp_stats = self._warp_sat_prior(
            seg_preds, delta_logits, u_sat)
        alpha, alpha_stats = self._build_class_range_alpha(seg_preds)
        if alpha is None:
            seg_preds = seg_preds + u_sat * delta_logits
        else:
            seg_preds = seg_preds + alpha * u_sat * delta_logits
        stats = dict(
            sat_prior_u_mean=u_sat.mean().item(),
            sat_prior_delta_abs_mean=delta_logits.abs().mean().item(),
        )
        if alpha is not None:
            stats['sat_prior_alpha_mean'] = alpha.mean().item()
            stats.update(alpha_stats)
        if warp_stats is not None:
            stats.update(warp_stats)
        return seg_preds, stats, warp_reg

    def forward_train(self, img, vectors, semantic_mask, points=None,
                      img_metas=None, all_prev_data=None,
                      all_local2global_info=None, sat_img=None,
                      skeleton_mask=None, **kwargs):

        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        bs = img.shape[0]

        _use_memory = self.use_memory and self.num_iter > self.mem_warmup_iters

        if all_prev_data is not None:
            num_prev_frames = len(all_prev_data)
            all_gts_prev, all_img_prev, all_img_metas_prev, all_semantic_mask_prev = [], [], [], []
            all_sat_img_prev = []
            all_skeleton_mask_prev = []
            for prev_data in all_prev_data:
                gts_prev, img_prev, img_metas_prev, valid_idx_prev, _ = self.batch_data(
                    prev_data['vectors'], prev_data['img'], prev_data['img_metas'], img.device)
                all_gts_prev.append(gts_prev)
                all_img_prev.append(img_prev)
                all_img_metas_prev.append(img_metas_prev)
                all_semantic_mask_prev.append(prev_data['semantic_mask'])
                all_sat_img_prev.append(prev_data.get('sat_img', None))
                all_skeleton_mask_prev.append(prev_data.get('skeleton_mask', None))
        else:
            num_prev_frames = 0

        assert points is None

        if self.skip_vector_head:
            backprop_backbone_ids = [0, num_prev_frames]
        else:
            backprop_backbone_ids = [num_prev_frames]

        track_query_info = None
        all_loss_dict_prev = []
        all_seg_aux_prev = []
        all_trans_loss = []
        all_outputs_prev = []
        prior_stats_all = []
        self.tracked_query_length = {}

        if _use_memory:
            self.memory_bank.set_bank_size(self.mem_len)
            self.memory_bank.init_memory(bs=bs)

        history_bev_feats = []
        history_img_metas = []
        gt_semantic = torch.flip(semantic_mask, [2,])
        gt_skeleton = torch.flip(skeleton_mask, [2,]) if skeleton_mask is not None else None

        # Iterate through prev frames
        for t in range(num_prev_frames):
            img_backbone_gradient = (t in backprop_backbone_ids)

            all_history_curr2prev, all_history_prev2curr, all_history_coord = \
                self.process_history_info(all_img_metas_prev[t], history_img_metas)

            # Encode satellite for this frame
            sat_img_t = all_sat_img_prev[t] if all_prev_data is not None else None
            sat_tokens_t, sat_grid_t = self._encode_satellite(sat_img_t)
            sat_tokens_backbone_t, sat_grid_backbone_t = self._sat_tokens_for_backbone(
                sat_tokens_t, sat_grid_t)

            _bev_feats, mlvl_feats = self.backbone(
                all_img_prev[t], all_img_metas_prev[t], t,
                history_bev_feats, history_img_metas, all_history_coord,
                points=None, img_backbone_gradient=img_backbone_gradient,
                sat_tokens=sat_tokens_backbone_t, sat_grid_size=sat_grid_backbone_t)

            bev_feats = self.neck(_bev_feats)
            if img_backbone_gradient:
                bev_feats = self._apply_conv_fusion(bev_feats, sat_tokens_t, sat_grid_t)
            else:
                with torch.no_grad():
                    bev_feats = self._apply_conv_fusion(bev_feats, sat_tokens_t, sat_grid_t)

            if _use_memory:
                self.memory_bank.curr_t = t

            if self.skip_vector_head or t == 0:
                self.temporal_propagate(bev_feats, all_img_metas_prev[t],
                    all_history_curr2prev, all_history_prev2curr,
                    _use_memory, track_query_info, timestep=t, get_trans_loss=False)
            else:
                trans_loss_dict = self.temporal_propagate(bev_feats, all_img_metas_prev[t],
                    all_history_curr2prev, all_history_prev2curr,
                    _use_memory, track_query_info, timestep=t, get_trans_loss=True)

            img_metas_prev = all_img_metas_prev[t]
            img_metas_next = all_img_metas_prev[t+1] if t < num_prev_frames-1 else img_metas
            gts_prev = all_gts_prev[t]
            gts_next = all_gts_prev[t+1] if t != num_prev_frames-1 else gts
            gts_semantic_prev = torch.flip(all_semantic_mask_prev[t], [2,])
            gts_semantic_curr = torch.flip(all_semantic_mask_prev[t+1], [2,]) if t != num_prev_frames-1 else gt_semantic
            gts_skeleton_prev = torch.flip(all_skeleton_mask_prev[t], [2,]) if all_skeleton_mask_prev[t] is not None else None

            local2global_prev = all_local2global_info[t]
            local2global_next = all_local2global_info[t+1]

            seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
                bev_feats, gts_semantic_prev, all_history_coord, skel_gts=gts_skeleton_prev, return_loss=True)
            aux_losses_prev = getattr(self.seg_decoder, '_last_aux_losses', {})
            seg_lovasz_loss = aux_losses_prev.get('seg_lovasz', seg_preds.new_tensor(0.0))
            seg_cldice_loss = aux_losses_prev.get('seg_cldice', seg_preds.new_tensor(0.0))
            seg_abl_loss = aux_losses_prev.get('seg_abl', seg_preds.new_tensor(0.0))
            seg_preds, prior_stats_t, prior_warp_reg_t = self._apply_sat_prior_correction(
                seg_preds, sat_tokens_t, sat_grid_t)
            if prior_stats_t is not None:
                prior_stats_all.append(prior_stats_t)
                losses_prev = self.seg_decoder.compute_losses(
                    seg_preds, gts_semantic_prev, skel_gts=gts_skeleton_prev)
                seg_loss = losses_prev['seg'] + losses_prev['lovasz'] + losses_prev['cldice'] + losses_prev['abl']
                seg_dice_loss = losses_prev['dice']
                seg_skel_loss = losses_prev['skel']
                seg_lovasz_loss = losses_prev['lovasz']
                seg_cldice_loss = losses_prev['cldice']
                seg_abl_loss = losses_prev['abl']
            if prior_warp_reg_t is None:
                prior_warp_reg_t = seg_preds.new_tensor(0.0)

            history_bev_feats.append(bev_feats)
            history_img_metas.append(all_img_metas_prev[t])
            if len(history_bev_feats) > self.history_steps:
                history_bev_feats.pop(0)
                history_img_metas.pop(0)

            if not self.skip_vector_head:
                gt_cur2prev, gt_prev2cur = self.get_two_frame_matching(
                    local2global_prev, local2global_next, gts_prev, gts_next)
                if t == 0:
                    memory_bank = None
                else:
                    memory_bank = self.memory_bank if _use_memory else None
                loss_dict_prev, outputs_prev, prev_inds_list, prev_gt_inds_list, \
                    prev_matched_reg_cost, prev_gt_list = self.head(
                        bev_features=bev_feats, img_metas=img_metas_prev,
                        gts=gts_prev, track_query_info=track_query_info,
                        memory_bank=memory_bank, return_loss=True, return_matching=True)
                all_outputs_prev.append(outputs_prev)
                if t > 0:
                    all_trans_loss.append(trans_loss_dict)

                pos_th = 0.4
                track_query_info = self.prepare_track_queries_and_targets(
                    gts_next, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost,
                    prev_gt_list, outputs_prev, gt_cur2prev, gt_prev2cur,
                    img_metas_prev, _use_memory, pos_th=pos_th, timestep=t)
            else:
                loss_dict_prev = {}

            loss_dict_prev['seg'] = seg_loss
            loss_dict_prev['seg_dice'] = seg_dice_loss
            loss_dict_prev['seg_skel'] = seg_skel_loss
            loss_dict_prev['seg_warp_reg'] = prior_warp_reg_t
            all_loss_dict_prev.append(loss_dict_prev)
            all_seg_aux_prev.append(dict(
                seg_lovasz=seg_lovasz_loss,
                seg_cldice=seg_cldice_loss,
                seg_abl=seg_abl_loss,
            ))

        if _use_memory:
            self.memory_bank.curr_t = num_prev_frames

        # Current frame
        img_backbone_gradient = num_prev_frames in backprop_backbone_ids

        all_history_curr2prev, all_history_prev2curr, all_history_coord = \
            self.process_history_info(img_metas, history_img_metas)

        # Encode satellite for current frame
        sat_tokens, sat_grid = self._encode_satellite(sat_img)
        sat_tokens_backbone, sat_grid_backbone = self._sat_tokens_for_backbone(
            sat_tokens, sat_grid)

        _bev_feats, mlvl_feats = self.backbone(
            img, img_metas, num_prev_frames, history_bev_feats, history_img_metas,
            all_history_coord, points=None, img_backbone_gradient=img_backbone_gradient,
            sat_tokens=sat_tokens_backbone, sat_grid_size=sat_grid_backbone)

        bev_feats = self.neck(_bev_feats)
        if self.conv_fusion is not None:
            self._vis_bev_pre_fusion = bev_feats.detach()
        bev_feats = self._apply_conv_fusion(bev_feats, sat_tokens, sat_grid)
        if self.conv_fusion is not None:
            self._vis_bev_post_fusion = bev_feats.detach()

        if self.skip_vector_head or num_prev_frames == 0:
            assert track_query_info is None
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, _use_memory, track_query_info,
                timestep=num_prev_frames, get_trans_loss=False)
        else:
            trans_loss_dict = self.temporal_propagate(bev_feats, img_metas,
                all_history_curr2prev, all_history_prev2curr, _use_memory,
                track_query_info, timestep=num_prev_frames, get_trans_loss=True)
            all_trans_loss.append(trans_loss_dict)

        seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
            bev_feats, gt_semantic, all_history_coord, skel_gts=gt_skeleton, return_loss=True)
        aux_losses = getattr(self.seg_decoder, '_last_aux_losses', {})
        seg_lovasz_loss = aux_losses.get('seg_lovasz', seg_preds.new_tensor(0.0))
        seg_cldice_loss = aux_losses.get('seg_cldice', seg_preds.new_tensor(0.0))
        seg_abl_loss = aux_losses.get('seg_abl', seg_preds.new_tensor(0.0))
        seg_preds, prior_stats, prior_warp_reg = self._apply_sat_prior_correction(
            seg_preds, sat_tokens, sat_grid)
        if prior_stats is not None:
            prior_stats_all.append(prior_stats)
            losses_curr = self.seg_decoder.compute_losses(
                seg_preds, gt_semantic, skel_gts=gt_skeleton)
            seg_loss = losses_curr['seg'] + losses_curr['lovasz'] + losses_curr['cldice'] + losses_curr['abl']
            seg_dice_loss = losses_curr['dice']
            seg_skel_loss = losses_curr['skel']
            seg_lovasz_loss = losses_curr['lovasz']
            seg_cldice_loss = losses_curr['cldice']
            seg_abl_loss = losses_curr['abl']
        if prior_warp_reg is None:
            prior_warp_reg = seg_preds.new_tensor(0.0)

        if not self.skip_vector_head:
            memory_bank = self.memory_bank if _use_memory else None
            preds_list, loss_dict, det_match_idxs, det_match_gt_idxs, gt_list = self.head(
                bev_features=bev_feats, img_metas=img_metas, gts=gts,
                track_query_info=track_query_info, memory_bank=memory_bank,
                return_loss=True)
        else:
            loss_dict = {}

        loss_dict['seg'] = seg_loss
        loss_dict['seg_dice'] = seg_dice_loss
        loss_dict['seg_skel'] = seg_skel_loss
        loss_dict['seg_warp_reg'] = prior_warp_reg
        seg_aux_curr = dict(
            seg_lovasz=seg_lovasz_loss,
            seg_cldice=seg_cldice_loss,
            seg_abl=seg_abl_loss,
        )

        # Aggregate losses
        loss = 0
        losses_t = []
        for loss_dict_t in (all_loss_dict_prev + [loss_dict]):
            loss_t = 0
            for name, var in loss_dict_t.items():
                loss_t = loss_t + var
            losses_t.append(loss_t)
            loss += loss_t

        for trans_loss_dict_t in all_trans_loss:
            trans_loss_t = trans_loss_dict_t['f_trans'] + trans_loss_dict_t['b_trans']
            loss += trans_loss_t

        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({k: v.item() for k, v in seg_aux_curr.items()})
        for t, loss_dict_t in enumerate(all_loss_dict_prev):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in loss_dict_t.items()}
            log_vars.update(log_vars_t)
        for t, aux_dict_t in enumerate(all_seg_aux_prev):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in aux_dict_t.items()}
            log_vars.update(log_vars_t)
        for t, loss_t in enumerate(losses_t):
            log_vars.update({'total_t{}'.format(t): loss_t.item()})
        for t, trans_loss_dict_t in enumerate(all_trans_loss):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in trans_loss_dict_t.items()}
            log_vars.update(log_vars_t)
        log_vars.update({'total': loss.item()})

        # Log satellite gate values
        if hasattr(self.backbone, 'transformer'):
            encoder = self.backbone.transformer.encoder
            for lid, layer in enumerate(encoder.layers):
                if hasattr(layer, 'sat_gate'):
                    log_vars[f'sat_gate_L{lid}'] = torch.sigmoid(layer.sat_gate).item()

        if self.conv_fusion is not None:
            log_vars['conv_fusion_gate'] = torch.sigmoid(self.conv_fusion.gate).item()
        if self.use_sat_prior and self.sat_prior is not None:
            log_vars['sat_prior_delta_scale'] = self.sat_prior.delta_scale.item()
        if prior_stats_all:
            keys = prior_stats_all[0].keys()
            for k in keys:
                log_vars[k] = sum(s[k] for s in prior_stats_all) / len(prior_stats_all)

        # Store attention maps from encoder layers for visualization
        if hasattr(self.backbone, 'transformer'):
            encoder = self.backbone.transformer.encoder
            for lid, layer in enumerate(encoder.layers):
                if hasattr(layer, '_sat_attn_weights'):
                    self._vis_sat_attn_map = layer._sat_attn_weights.detach()
                    del layer._sat_attn_weights
                    break  # take first layer only

        # Store BEV seg preds and GT for visualization hooks
        self._vis_seg_preds = seg_preds.detach()
        self._vis_gt_semantic = gt_semantic.detach()
        if gt_skeleton is not None:
            self._vis_gt_skeleton = gt_skeleton.detach()

        num_sample = img.size(0)
        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, seq_info=None,
                     sat_img=None, **kwargs):
        assert img.shape[0] == 1

        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        scene_name, local_idx, seq_length = seq_info[0]
        first_frame = (local_idx == 0)
        img_metas[0]['local_idx'] = local_idx

        if first_frame:
            if self.use_memory:
                self.memory_bank.set_bank_size(self.test_time_history_steps)
                self.memory_bank.init_memory(bs=1)
            self.history_bev_feats_all = []
            self.history_img_metas_all = []

        if self.use_memory:
            self.memory_bank.curr_t = local_idx

        selected_mem_ids = self.select_memory_entries(self.history_img_metas_all, img_metas)
        history_img_metas = [self.history_img_metas_all[idx] for idx in selected_mem_ids]
        history_bev_feats = [self.history_bev_feats_all[idx] for idx in selected_mem_ids]

        all_history_curr2prev, all_history_prev2curr, all_history_coord = \
            self.process_history_info(img_metas, history_img_metas)

        # Encode satellite
        sat_tokens, sat_grid = self._encode_satellite(sat_img)
        sat_tokens_backbone, sat_grid_backbone = self._sat_tokens_for_backbone(
            sat_tokens, sat_grid)

        _bev_feats, mlvl_feats = self.backbone(
            img, img_metas, local_idx, history_bev_feats, history_img_metas,
            all_history_coord, points=points,
            sat_tokens=sat_tokens_backbone, sat_grid_size=sat_grid_backbone)

        bev_feats = self.neck(_bev_feats)
        bev_feats = self._apply_conv_fusion(bev_feats, sat_tokens, sat_grid)

        if self.skip_vector_head or first_frame:
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, self.use_memory, track_query_info=None)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
            seg_preds, _, _ = self._apply_sat_prior_correction(seg_preds, sat_tokens, sat_grid)
            if not self.skip_vector_head:
                preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
            track_dict = None
        else:
            track_query_info = self.head.get_track_info(scene_name, local_idx)
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, self.use_memory, track_query_info)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
            seg_preds, _, _ = self._apply_sat_prior_correction(seg_preds, sat_tokens, sat_grid)
            memory_bank = self.memory_bank if self.use_memory else None
            preds_list = self.head(bev_feats, img_metas=img_metas,
                track_query_info=track_query_info, memory_bank=memory_bank,
                return_loss=False)
            track_dict = self._process_track_query_info(track_query_info)

        if not self.skip_vector_head:
            preds_dict = preds_list[-1]
        else:
            preds_dict = None

        self.history_bev_feats_all.append(bev_feats)
        self.history_img_metas_all.append(img_metas)
        if len(self.history_bev_feats_all) > self.test_time_history_steps:
            self.history_bev_feats_all.pop(0)
            self.history_img_metas_all.pop(0)

        if not self.skip_vector_head:
            memory_bank = self.memory_bank if self.use_memory else None
            thr_det = 0.4 if first_frame else 0.6
            pos_results = self.head.prepare_temporal_propagation(
                preds_dict, scene_name, local_idx, memory_bank,
                thr_track=0.5, thr_det=thr_det)

        if not self.skip_vector_head:
            results_list = self.head.post_process(preds_dict, tokens, track_dict)
            results_list[0]['pos_results'] = pos_results
            results_list[0]['meta'] = img_metas[0]
        else:
            results_list = [{'vectors': [], 'scores': [], 'labels': [],
                             'props': [], 'token': token} for token in tokens]

        for b_i in range(len(results_list)):
            tmp_scores, tmp_labels = seg_preds[b_i].max(0)
            tmp_scores = tmp_scores.sigmoid()
            preds_i = torch.zeros(tmp_labels.shape, dtype=torch.uint8).to(tmp_scores.device)
            pos_ids = self._get_seg_positive_ids(tmp_scores, tmp_labels)
            preds_i[pos_ids] = tmp_labels[pos_ids].type(torch.uint8) + 1
            preds_i = preds_i.cpu().numpy()
            results_list[b_i]['semantic_mask'] = preds_i
            if 'token' not in results_list[b_i]:
                results_list[b_i]['token'] = tokens[b_i]

        return results_list

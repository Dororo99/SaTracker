import torch
from torch import nn as nn
from torch.nn import functional as F
import mmcv

from mmdet.models.builder import LOSSES
from mmdet.models.losses import FocalLoss, weight_reduce_loss

from einops import rearrange


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological erosion."""
    erode_h = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_w = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(erode_h, erode_w)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological dilation."""
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological opening."""
    return _soft_dilate(_soft_erode(mask))


def _soft_skeletonize(mask: torch.Tensor, num_iters: int) -> torch.Tensor:
    """Differentiable soft skeletonization."""
    mask = mask.clamp(0.0, 1.0)
    skel = F.relu(mask - _soft_open(mask))
    for _ in range(num_iters):
        mask = _soft_erode(mask)
        delta = F.relu(mask - _soft_open(mask))
        skel = torch.maximum(skel, delta)
    return skel


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class MaskFocalLoss(FocalLoss):
    def __init__(self,**kwargs):
        super(MaskFocalLoss, self).__init__(**kwargs)
    
    def forward(self, 
                pred, 
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if not self.use_sigmoid:
            raise NotImplementedError
        
        num_classes = pred.size(1)
        loss = 0
        for index in range(num_classes):
            loss += self.loss_weight * py_sigmoid_focal_loss(
                pred[:,index],
                target[:,index],
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        loss /= num_classes
        return loss * self.loss_weight


@LOSSES.register_module()
class MaskDiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, loss_weight):
        super(MaskDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        bs, num_classes = pred.shape[:2]
        pred = rearrange(pred, 'b n h w -> b n (h w)')
        target = rearrange(target, 'b n h w -> b n (h w)')
        pred = pred.sigmoid()
        intersection = torch.sum(pred * target, dim=2)  # (N, C)
        union = torch.sum(pred.pow(2), dim=2) + torch.sum(target, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1
        
        loss = self.loss_weight * dice_loss
        return loss


@LOSSES.register_module()
class MaskSkeletonLoss(nn.Module):
    """Skeleton consistency loss for thin structures.

    This loss is intended to be used as an additive regularizer on top of
    focal/dice losses, typically for line-like classes (e.g., divider/boundary).
    """

    def __init__(self,
                 loss_weight=0.1,
                 num_dilation=1,
                 class_indices=None,
                 class_weights=None,
                 ignore_empty_targets=True,
                 eps=1e-5):
        super(MaskSkeletonLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_dilation = int(num_dilation)
        self.class_indices = class_indices
        self.class_weights = class_weights
        self.ignore_empty_targets = ignore_empty_targets
        self.eps = eps
        if self.num_dilation < 0:
            raise ValueError('num_dilation must be >= 0')
        if self.class_indices is not None and len(self.class_indices) == 0:
            raise ValueError('class_indices must be None or non-empty')
        if self.class_weights is not None and len(self.class_weights) == 0:
            raise ValueError('class_weights must be None or non-empty')
        if (self.class_indices is not None and self.class_weights is not None
                and len(self.class_indices) != len(self.class_weights)):
            raise ValueError('class_weights length must match class_indices length')

    def _select_classes(self, pred, target):
        if self.class_indices is None:
            return pred, target
        idx = torch.as_tensor(self.class_indices, device=pred.device, dtype=torch.long)
        return pred.index_select(1, idx), target.index_select(1, idx)

    def _class_weights_tensor(self, num_classes, device, dtype):
        if self.class_weights is None:
            return None
        class_w = torch.as_tensor(self.class_weights, device=device, dtype=dtype)
        if class_w.numel() != num_classes:
            raise ValueError(
                f'class_weights length ({class_w.numel()}) must match selected classes ({num_classes})')
        return class_w.view(1, num_classes)

    def forward(self, pred, target):
        # pred: [B, C, H, W] logits, target: [B, C, H, W] binary
        pred_prob = pred.sigmoid()
        target = target.type_as(pred_prob)
        pred_prob, target = self._select_classes(pred_prob, target)

        pred_skel = _soft_skeletonize(pred_prob, self.num_dilation)
        target_skel = _soft_skeletonize(target, self.num_dilation)

        pred_flat = rearrange(pred_skel, 'b c h w -> b c (h w)')
        target_flat = rearrange(target_skel, 'b c h w -> b c (h w)')

        intersection = (pred_flat * target_flat).sum(dim=-1)
        denom = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        loss_per_bc = 1 - (2 * intersection + self.eps) / (denom + self.eps)
        class_w = self._class_weights_tensor(
            num_classes=loss_per_bc.shape[1],
            device=loss_per_bc.device,
            dtype=loss_per_bc.dtype)

        if self.ignore_empty_targets:
            valid = (target_flat.sum(dim=-1) > 0).to(loss_per_bc.dtype)
            if class_w is not None:
                valid = valid * class_w
            valid_sum = valid.sum()
            if valid_sum <= 0:
                return pred.new_zeros(())
            loss_val = (loss_per_bc * valid).sum() / valid_sum
        else:
            if class_w is None:
                loss_val = loss_per_bc.mean()
            else:
                weighted = loss_per_bc * class_w
                loss_val = weighted.sum() / (loss_per_bc.shape[0] * class_w.sum())

        return self.loss_weight * loss_val

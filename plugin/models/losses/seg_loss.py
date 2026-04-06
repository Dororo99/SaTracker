import torch
from torch import nn as nn
from torch.nn import functional as F
import mmcv

from mmdet.models.builder import LOSSES
from mmdet.models.losses import FocalLoss, weight_reduce_loss

from einops import rearrange


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
    def __init__(self, class_weights=None, **kwargs):
        super(MaskFocalLoss, self).__init__(**kwargs)
        self.class_weights = [float(w) for w in class_weights] if class_weights is not None else None
    
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
            cls_loss = py_sigmoid_focal_loss(
                pred[:,index],
                target[:,index],
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
            if self.class_weights is not None and index < len(self.class_weights):
                cls_loss = cls_loss * self.class_weights[index]
            loss += cls_loss

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
class SkelRecallLoss(nn.Module):
    """Skeleton Recall Loss for thin-structure segmentation.

    Computes (1 - recall) on skeleton pixels for specified classes.
    Classes with all-zero skeleton contribute zero loss.

    Args:
        loss_weight (float): Weight for this loss term. Default: 1.0.
        skel_classes (list[int]): Class indices to compute loss on.
            Default: [1, 2] (divider, boundary).
    """

    def __init__(self, loss_weight=1.0, skel_classes=None):
        super(SkelRecallLoss, self).__init__()
        self.loss_weight = loss_weight
        self.skel_classes = skel_classes if skel_classes is not None else [1, 2]
        self.eps = 1e-5

    def forward(self, pred, skel_gt):
        """
        Args:
            pred: (B, C, H, W) raw logits from seg head.
            skel_gt: (B, C, H, W) skeleton ground truth (0 or 1).
        Returns:
            loss: scalar tensor.
        """
        pred_sigmoid = pred.sigmoid()
        loss = pred.new_tensor(0.0)
        num_valid = 0

        for cls_idx in self.skel_classes:
            skel_cls = skel_gt[:, cls_idx].float()  # (B, H, W)
            pred_cls = pred_sigmoid[:, cls_idx]      # (B, H, W)
            skel_sum = skel_cls.sum()
            if skel_sum > 0:
                recall = (pred_cls * skel_cls).sum() / (skel_sum + self.eps)
                loss = loss + (1.0 - recall)
                num_valid += 1

        if num_valid > 0:
            loss = loss / num_valid

        return self.loss_weight * loss


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _flatten_binary_scores(scores, labels):
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels


def _lovasz_hinge_flat(logits, labels):
    if labels.numel() == 0:
        return logits.new_tensor(0.0)
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def _lovasz_hinge(logits, labels, per_image=True):
    if per_image:
        losses = []
        for logit_i, label_i in zip(logits, labels):
            logit_i, label_i = _flatten_binary_scores(logit_i, label_i)
            losses.append(_lovasz_hinge_flat(logit_i, label_i))
        if len(losses) == 0:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()
    logits, labels = _flatten_binary_scores(logits, labels)
    return _lovasz_hinge_flat(logits, labels)


@LOSSES.register_module()
class MaskLovaszLoss(nn.Module):
    """Lovasz hinge loss for multi-label segmentation logits."""

    def __init__(self, loss_weight=1.0, classes=None, per_image=True):
        super(MaskLovaszLoss, self).__init__()
        self.loss_weight = loss_weight
        self.classes = classes
        self.per_image = per_image

    def forward(self, pred, target):
        num_classes = pred.shape[1]
        class_ids = self.classes if self.classes is not None else list(range(num_classes))
        losses = []
        for cls_idx in class_ids:
            losses.append(_lovasz_hinge(
                pred[:, cls_idx], target[:, cls_idx], per_image=self.per_image))
        if len(losses) == 0:
            return pred.new_tensor(0.0)
        return self.loss_weight * torch.stack(losses).mean()


def _soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img, iters):
    img = img.clamp(min=0.0, max=1.0)
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


@LOSSES.register_module()
class SoftCLDiceLoss(nn.Module):
    """Soft clDice loss for connectivity preservation on thin structures."""

    def __init__(self, loss_weight=1.0, classes=None, iterations=10, eps=1e-6):
        super(SoftCLDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.classes = classes
        self.iterations = int(iterations)
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.sigmoid()
        num_classes = pred.shape[1]
        class_ids = self.classes if self.classes is not None else list(range(num_classes))

        losses = []
        for cls_idx in class_ids:
            p = pred[:, cls_idx:cls_idx+1]
            t = target[:, cls_idx:cls_idx+1].float()
            skel_p = _soft_skel(p, self.iterations)
            skel_t = _soft_skel(t, self.iterations)

            tprec = (skel_p * t).sum(dim=(1, 2, 3)) / (skel_p.sum(dim=(1, 2, 3)) + self.eps)
            tsens = (skel_t * p).sum(dim=(1, 2, 3)) / (skel_t.sum(dim=(1, 2, 3)) + self.eps)
            cl_dice = 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
            losses.append(cl_dice.mean())

        if len(losses) == 0:
            return pred.new_tensor(0.0)
        return self.loss_weight * torch.stack(losses).mean()


@LOSSES.register_module()
class ActiveBoundaryLoss(nn.Module):
    """Boundary alignment loss using Sobel edge magnitude consistency."""

    def __init__(self, loss_weight=1.0, classes=None, eps=1e-6):
        super(ActiveBoundaryLoss, self).__init__()
        self.loss_weight = loss_weight
        self.classes = classes
        self.eps = eps

        sobel_x = torch.tensor([[1., 0., -1.],
                                [2., 0., -2.],
                                [1., 0., -1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _edge_mag(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def forward(self, pred, target):
        pred = pred.sigmoid()
        num_classes = pred.shape[1]
        class_ids = self.classes if self.classes is not None else list(range(num_classes))
        losses = []
        for cls_idx in class_ids:
            p = pred[:, cls_idx:cls_idx+1]
            t = target[:, cls_idx:cls_idx+1].float()
            p_edge = self._edge_mag(p)
            t_edge = self._edge_mag(t)
            losses.append(F.l1_loss(p_edge, t_edge))

        if len(losses) == 0:
            return pred.new_tensor(0.0)
        return self.loss_weight * torch.stack(losses).mean()

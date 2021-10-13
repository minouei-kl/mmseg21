import math
import numpy as np

from typing import Optional

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import dice_loss
from .cross_entropy_loss import  (cross_entropy, binary_cross_entropy, 
                                  mask_cross_entropy, _expand_onehot_labels)
from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

@weighted_loss
def focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        ignore_index=255,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target, weight = _expand_onehot_labels(target, None, output.shape,
                                              ignore_index)
    target = target.type(output.type())
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    
    # logpt = binary_cross_entropy(output, target, reduction="none", 
    #                                            ignore_index=ignore_index)
    # print(output.shape, target.shape)
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss

@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 **kwards):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        # target = F.one_hot(
        #     torch.clamp(target.long(), 0, num_classes - 1),
        #     num_classes=num_classes)

        loss = self.loss_weight * focal_loss_with_logits(
            pred,
            target,
            ignore_index=self.ignore_index)
        
        return loss
    
@LOSSES.register_module()
class FocalDiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 focal_weight = 0.75,
                 **kwards):
        super(FocalDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent=exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.focal_weight = focal_weight

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        

        loss_focal = self.focal_weight * self.loss_weight * focal_loss_with_logits(
            pred,
            target,
            ignore_index=self.ignore_index)
        
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss_dice = (1 - self.focal_weight) * self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        
        return loss_focal + loss_dice
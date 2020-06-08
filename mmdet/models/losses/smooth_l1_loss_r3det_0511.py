import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss
from math import pi
import random
import string
from mmdet.core.bbox.transforms_rbbox import distance2obb,RotBox2Polys_torch, poly2bbox_torch, mask2poly, Tuplelist2Polylist
import matplotlib.pyplot as plt 
def randomString(stringLength=8):

    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(stringLength))


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.zeros_like(pred)
    diff[:,:4] = torch.log((target[:,:4] / pred[:,:4]).clamp(min=1e-1).clamp(max=5))
    diff[:,4] = (pred[:,4]-target[:,4])/ (pi/2)

    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if random.random() < 0.001:
        print("Sample :",(target[:5]),(pred[:5]),'\n\n----------------------',flush=True)

    return loss.sum(1)


@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta, #)
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


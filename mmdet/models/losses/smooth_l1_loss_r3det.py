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
    
    diff = torch.abs(pred - target)
    #diff[:,4] = torch.remainder(diff[:,4],pi*2)/ (torch.ones_like(diff[:,4])*pi*2)
    #diff[:,:4] = torch.log(diff[:,:4]) 
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if random.random() < 0.0001:
        print("Sample :",(target[:5]),(pred[:5]),'\n\n----------------------',flush=True)

    '''
    if random.random() < 0.001:
        print('Random sample bbox pred\n',pred[:2],'\n\n GT bbox target',target[:2],'\n\n----------------------',flush=True)
        gt_poly = RotBox2Polys_torch(target.detach().cpu())
        pred_poly = RotBox2Polys_torch(pred.detach().cpu())
        query = pred_poly[0].clone().cpu().numpy()
        ox = gt_poly[0].clone().cpu().numpy()
        #figtext 
        plt.scatter(query[::2],query[1::2],c='blue')
        plt.scatter(ox[::2],ox[1::2],c='red')
        str_ = randomString(6) 
        plt.savefig('./tests/0428_r3det_smoothl1_%s.png'%(str_))
        plt.clf()
        print("added %s"%(str_),flush=True) 
#            print('\t\t',cls_score[i].shape,cls_score[i],flush=True)
#            print('\t\t',gt_cls[i].shape, gt_cls[i],flush=True)
    '''
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


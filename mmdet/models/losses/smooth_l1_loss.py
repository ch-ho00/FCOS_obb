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

def dist2obb(pos_points,dist):
    '''
    Input:
        pos_point = Tensor (N,2) containing coordinate of predicted distances
        dist = Tensor (N,5) containing (top,right, bottom, left, angle)
    Output:
        decoded_obb = Tensor (N,5) in (x,y,w,h,angle)
    '''
    t,r,b,l,th = dist[:,0], dist[:,1], dist[:,2], dist[:,3],dist[:,4]
    x,y = pos_points[:,0], pos_points[:,1]
    th = 5/4*pi + th*pi/2
    angle = th
    angle = angle % (pi/2)
    angle_2 = angle + pi/2

    angle = torch.where(angle<pi/4, angle_2, angle)

    cos_t = torch.cos(angle);  sin_t =torch.sin(angle)
    t_cos = t* cos_t ; t_sin = t*sin_t
    r_cos = r* cos_t ; r_sin = r*sin_t
    b_cos = b* cos_t ; b_sin = b*sin_t
    l_cos = l* cos_t ; l_sin = l*sin_t
    
    #coords = [x-b_cos-l_sin, y-b_sin+l_cos, x+r_sin-b_cos, y-r_cos-b_sin, x+t_cos+ r_sin, y+t_sin-r_cos, x-l_sin + t_cos, y + l_cos + t_sin]
    x_new = x + (t_cos+ r_sin -b_cos -l_sin)/2
    y_new = y + (t_sin- r_cos -b_sin +l_cos)/2
    h = t+b
    w = r+l
    return torch.cat([x_new.unsqueeze(1),y_new.unsqueeze(1),h.unsqueeze(1),w.unsqueeze(1),th.unsqueeze(1)],-1)
    

#@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0,eps=1e-6):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.zeros_like(pred)
    diff[:,0] = (target[:,0] - pred[:,0])/ target[:,2]
    diff[:,1] = (target[:,1] - pred[:,1])/ target[:,3]
    diff[:,2:4] = torch.log((target[:,2:4]) / (pred[:,2:4]))
    diff[:,4] = (target[:,4]-pred[:,4])/ (pi/2)

    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    reg_weight = torch.tensor((10.0, 10.0, 5., 5., 1.)).cuda()
    loss = loss * reg_weight


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
                pos_points,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred = dist2obb(pos_points,pred).clamp(min=1e-6)
        target = dist2obb(pos_points,target).clamp(min=1e-6)

        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
#            weight,
            beta=self.beta)
#            reduction=reduction,
#            avg_factor=avg_factor,
#            **kwargs)
        return loss_bbox


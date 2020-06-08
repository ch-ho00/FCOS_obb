import torch
import torch.nn as nn
from mmdet.core import bbox_overlaps
from mmdet.core.bbox.transforms_rbbox import RotBox2Polys_torch, poly2bbox_torch, mask2poly, Tuplelist2Polylist

from ..registry import LOSSES
from .utils import weighted_loss


#@weighted_loss
def smooth_l1_loss(pred_box, gt_box, beta=1.0):
    assert beta > 0
    gt_poly = RotBox2Polys_torch(gt_box)
    pred_poly = RotBox2Polys_torch(pred_box)

    # polygon to hbb
    h_bboxes= poly2bbox_torch(gt_poly)
    h_query_bboxes = poly2bbox_torch(pred_poly)
    # hious
    ious = bbox_overlaps(h_query_bboxes, h_bboxes)
    import pdb
    # pdb.set_trace()
    inds = (ious>0).int().nonzero()
    iou =  torch.tensor(0).float()
    #print(inds.shape, ious.shape,flush=True)
    loss = torch.zeros(5).cuda()
    for index in range(inds.shape[0]):
        box_index = inds[index][1]
        query_box_index = inds[index][0]

        box = gt_box[box_index]
        query_box = pred_box[query_box_index]
        #union_poly = torch.cat([box,query_bo],0)

        diff = torch.abs(box - query_box)
        l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        l1_loss = l1_loss / ((l1_loss**2).sum().sqrt())

        loss += l1_loss * (-ious[query_box_index][box_index].log())
    return loss


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
#            weight,
            beta=self.beta)
#            reduction=reduction,
#            avg_factor=avg_factor,
#            **kwargs)
        return loss_bbox

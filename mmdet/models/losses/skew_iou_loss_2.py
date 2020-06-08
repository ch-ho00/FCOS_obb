import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mmdet.core import bbox_overlaps
from ..registry import LOSSES
import numpy as np
import DOTA_devkit.polyiou as polyiou
from mmdet.core.bbox.transforms_rbbox import RotBox2Polys_torch, poly2bbox_torch, mask2poly, Tuplelist2Polylist
from .utils import weighted_loss
from shapely.geometry import Polygon,MultiPoint
#from poly_nms_gpu.poly_overlaps import poly_overlaps

import random
import string

def randomString(stringLength=8):

    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(stringLength))


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

@weighted_loss
def skew_iou_loss(pred_box,gt_box,eps=1e-6):
    # first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
    # from 5 pt representation to 8pt polygon
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
    for index in range(inds.shape[0]):
        box_index = inds[index][1]
        query_box_index = inds[index][0]

        box = gt_poly[box_index]
        query_box = pred_poly[query_box_index]
        #union_poly = torch.cat([box,query_box],0)
        
        
        overlap = polygon_iou(query_box.detach().cpu().numpy(),box.detach().cpu().numpy())
        #overlap = polyiou.iou_poly(polyiou.VectorDouble(query_box.cpu().numpy().astype(np.float),polyiou.VectorDouble(box.cpu().numpy().astype(np.float))))
        '''
        query = query_box.clone().detach().cpu().numpy()
        ox = box.clone().detach().cpu().numpy()
        plt.scatter(query[::2],query[1::2])
        plt.scatter(ox[::2],ox[1::2])
        plt.title(str(overlap))
        plt.savefig('./tests/NEW_skew_iou_%s'%(randomString(6)))
        plt.clf()
        '''
        if overlap < eps:
            overlap = eps
        iou += -torch.tensor(overlap).log()
        
#    ious  = ious.clamp(min=eps)
#    loss = -ious.log()
#    loss = -np.log(iou)  
    loss = torch.tensor(iou).cuda()
    return loss



#def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
#    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
#    loss = -ious.log()
#    return loss

@LOSSES.register_module
class skew_IoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(skew_IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
#        if weight is not None and not torch.any(weight > 0):
#            print(pred.shape, weight.shape,"????????????", flush=True)
#            return (weight * pred).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * skew_iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

if __name__ == "__main__":
    gt_bbox = np.array([[1,2,3,5,0],[6,6,4,4,np.pi/2]])
    pred = np.array([[2,1,5,3,0],[6,6,4,4,0]])
    print(skew_iou_loss(gt_bbox,pred))

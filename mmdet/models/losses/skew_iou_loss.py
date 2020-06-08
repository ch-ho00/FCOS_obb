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
import cv2
import random
import string
import numpy as np

from math import pi, cos, sin

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )




def dist2rect_list(dist_obb,pos_point):
    t,r,b,l,th = dist_obb.detach().cpu().numpy().astype(np.float)
    th = th % (np.pi/2)
    if th < np.pi/4:
        th += np.pi/2
    x,y = pos_point; x = x.detach().cpu().numpy() ; y=y.detach().cpu().numpy()
    cos_t = cos(th);  sin_t =sin(th)
    t_cos = t* cos_t ; t_sin = t*sin_t
    r_cos = r* cos_t ; r_sin = r*sin_t
    b_cos = b* cos_t ; b_sin = b*sin_t
    l_cos = l* cos_t ; l_sin = l*sin_t
    return [x-l_sin + t_cos, y + l_cos + t_sin, x+t_cos+ r_sin, y+t_sin-r_cos, x+r_sin-b_cos, y-r_cos-b_sin, x-b_cos-l_sin, y-b_sin+l_cos]



def dist2rect(dist_obb,pos_point):
    t,r,b,l,th = dist_obb
    x,y = pos_point
    th = 5/4*pi + th*pi/2
    th = th % (pi/2) 
    if th < pi/4:
        th += pi/2

    cos_t = cos(th);  sin_t =sin(th)
    t_cos = t* cos_t ; t_sin = t*sin_t
    r_cos = r* cos_t ; r_sin = r*sin_t
    b_cos = b* cos_t ; b_sin = b*sin_t
    l_cos = l* cos_t ; l_sin = l*sin_t
    
    return ( 
        Vector(x-b_cos-l_sin, y-b_sin+l_cos),
        Vector(x+r_sin-b_cos, y-r_cos-b_sin),
        Vector(x+t_cos+ r_sin, y+t_sin-r_cos),
        Vector(x-l_sin + t_cos, y + l_cos + t_sin)
    )


def rectangle_vertices(cx, cy, w, h, r):
    angle = r
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector( dxcos -  dysin,  dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def intersection_area(r1, r2,pos_point):

#    rect1 = rectangle_vertices(*r1)
#    rect2 = rectangle_vertices(*r2)
    area1 = (r1[0]+r1[2]) * (r1[3]+r1[1])
    area2 = (r2[0]+r2[2]) * (r2[3]+r2[1])

    rect1 = dist2rect(r1,pos_point)
    rect2 = dist2rect(r2,pos_point)

#    print('\t',r1,"->",[(r.x,r.y) for r in rect1],"\n\t",r2,'->',[(r.x,r.y) for r in rect2],flush=True)
    intersection = rect1

    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return torch.tensor(0.0).cuda()
    int_ =0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))
    return int_/(area1+area2-int_)

def randomString(stringLength=8):

    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(stringLength))


#@weighted_loss
def skew_iou_loss(pred,target, pos_points,eps=1e-6):

    # polygon to hbb    
    #pred_box[:,4] = pred_box[:,4].detach()
    #pred_box[:,4] = np.pi *3 /2

    for i in range(pred.shape[0]):
        overlap = intersection_area(pred[i],target[i],pos_points[i]).clamp(min=eps)
        if i ==0:
            iou = -overlap.log()
        else:
            iou += -overlap.log()

        if random.random() < 0.0001:
            print("Sample :",(target[:5]),(pred[:5]),overlap,'\n\n----------------------',flush=True)
    return iou


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
                pos_point,
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
            pred=pred,
            target=target,
            pos_points=pos_point)
#            eps=self.eps,
#            weight=weight,
#            reduction=reduction,
#            avg_factor=avg_factor,
#            **kwargs)
        return loss

if __name__ == "__main__":
    for i in range(11):
        gt_bbox = torch.tensor([[6.,6.,4.,4.,i/10]])
        pred = torch.tensor([[6.,6.,4.,4.,0.]])
        print((-skew_iou_loss(gt_bbox,pred,torch.tensor([[0,0]]))).exp())

import torch
from bbox import bbox_overlaps_cython
# from bbox_v2 import bbox_overlaps_cython_v2
import numpy as np
import DOTA_devkit.polyiou as polyiou
from mmdet.core.bbox.transforms_rbbox import RotBox2Polys_torch,RotBox2Polys,poly2bbox_torch, poly2bbox, mask2poly, Tuplelist2Polylist
import matplotlib.pyplot as plt
import string
import random
def bbox_overlaps_cy(boxes, query_boxes):
    box_device = boxes.device
    boxes_np = boxes.cpu().numpy().astype(np.float)
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
    ious = bbox_overlaps_cython(boxes_np, query_boxes_np)
    return torch.from_numpy(ious).to(box_device)

# def bbox_overlaps_cy2(boxes, query_boxes):
#     box_device = boxes.device
#     boxes_np = boxes.cpu().numpy().astype(np.float)
#     query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
#     ious = bbox_overlaps_cython_v2(boxes_np, query_boxes_np)
#     return torch.from_numpy(ious).to(box_device)

def bbox_overlaps_np(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    box_device = boxes.device
    boxes = boxes.cpu().numpy().astype(np.float)
    query_boxes = query_boxes.cpu().numpy().astype(np.float)

    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return torch.from_numpy(overlaps).to(box_device)

# def bbox_overlaps_torch_v2(anchors, gt_boxes):
#     """
#     anchors: (N, 4) ndarray of float
#     gt_boxes: (K, 4) ndarray of float
#
#     overlaps: (N, K) ndarray of overlap between boxes and query_boxes
#     """
#     N = anchors.size(0)
#     K = gt_boxes.size(0)
#
#     gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
#                 (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)
#
#     anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
#                 (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)
#
#     boxes = anchors.view(N, 1, 4).expand(N, K, 4)
#     query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)
#
#     iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
#         torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
#     iw[iw < 0] = 0
#
#     ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
#         torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
#     ih[ih < 0] = 0
#
#     ua = anchors_area + gt_boxes_area - (iw * ih)
#     overlaps = iw * ih / ua
#
#     return overlaps

def bbox_overlaps_np_v2(bboxes1, bboxes2):
    """
    :param bboxes1: (N, 4) ndarray of float
    :param bboxes2: (K, 4) ndarray of float
    :return: overlaps (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = bboxes1.shape[0]
    K = bboxes2.shape[0]

    area2 = ((bboxes2[:, 2] - bboxes2[:, 0] + 1) *
             (bboxes2[:, 3] - bboxes2[:, 1] + 1))[np.newaxis, :]

    area1 = ((bboxes1[:, 2] - bboxes1[:, 0] + 1) *
             (bboxes1[:, 3] - bboxes1[:, 1]))[:, np.newaxis]

    bboxes2 = bboxes2[np.newaxis, :, :]

    bboxes1 = bboxes1[:, np.newaxis, :]

    iw = np.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2]) - \
         np.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0]) + 1

    iw[iw < 0] = 0

    ih = np.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3]) - \
         np.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1]) + 1

    ih[ih < 0] = 0

    ua = area1 + area2 - (iw * ih)

    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']
    # import pdb
    # pdb.set_trace()
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def bbox_overlaps_np_v3(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """

    :param bboxes1: (ndarray): shape (m, 4)
    :param bboxes2: (ndarray): shape (n, 4)
    :param mode: (str) : "iou" or "iof"
    :param is_aligned: (ndarray): shape (m, n) if is_aligned == False else shape (m, 1)
    :return:
    """

    assert mode in ['iou', 'iof']

    box_device = bboxes1.device
    bboxes1 = bboxes1.cpu().numpy().astype(np.float)
    bboxes2 = bboxes2.cpu().numpy().astype(np.float)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return np.random.rand(rows, 1).astype(bboxes1.dtype) if is_aligned \
            else np.random.rand(rows, cols).astype(bboxes1.dtype)

    if is_aligned:
        lt = np.maximum(bboxes1[:, :2], bboxes2[:, :2])
        rb = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])

        wh = np.clip(rb - lt + 1, a_min=0, a_max=None)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = np.clip(rb - lt + 1, a_min=0, a_max=None) # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])
    ious = torch.from_numpy(ious).to(box_device)
    return ious








def bbox_overlaps_fp16(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """
    The fp16 version exist some bugs
    Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    bboxes1_fp16 = bboxes1.half()/100.
    bboxes2_fp16 = bboxes2.half()/100.

    rows = bboxes1_fp16.size(0)
    cols = bboxes2_fp16.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1_fp16.new(rows, 1) if is_aligned else bboxes1_fp16.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1_fp16[:, :2], bboxes2_fp16[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1_fp16[:, 2:], bboxes2_fp16[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1_fp16[:, 2] - bboxes1_fp16[:, 0] + 1) * (
            bboxes1_fp16[:, 3] - bboxes1_fp16[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2_fp16[:, 2] - bboxes2_fp16[:, 0] + 1) * (
                bboxes2_fp16[:, 3] - bboxes2_fp16[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1_fp16[:, None, :2], bboxes2_fp16[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1_fp16[:, None, 2:], bboxes2_fp16[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1_fp16[:, 2] - bboxes1_fp16[:, 0] + 1) * (
            bboxes1_fp16[:, 3] - bboxes1_fp16[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2_fp16[:, 2] - bboxes2_fp16[:, 0] + 1) * (
                bboxes2_fp16[:, 3] - bboxes2_fp16[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious.float()

def mask_overlaps():
    pass

# def bbox_overlaps_cy(boxes, query_boxes):
#     box_device = boxes.device
#     boxes_np = boxes.cpu().numpy().astype(np.float)
#     query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
#     ious = bbox_overlaps_cython(boxes_np, query_boxes_np)
#     return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
    # import pdb
    # pdb.set_trace()
    box_device = query_boxes.device
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)

    # polys_np = RotBox2Polys(boxes_np)
    # TODO: change it to only use pos gt_masks
    # polys_np = mask2poly(gt_masks)
    # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

    polys_np = RotBox2Polys(rbboxes).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np)

    h_bboxes_np = poly2bbox(polys_np)
    h_query_bboxes_np = poly2bbox(query_polys_np)

    # hious
    ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_hybrid(boxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, use the gpu_overlaps to calculate the obb overlaps
    # box_device = boxes.device
    pass

def rbbox_overlaps_cy(boxes_np, query_boxes_np):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps

    polys_np = RotBox2Polys(boxes_np).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np).astype(np.float)

    h_bboxes_np = poly2bbox(polys_np).astype(np.float)
    h_query_bboxes_np = poly2bbox(query_polys_np).astype(np.float)

    # hious
    ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return ious


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




def poly2rect(r):
    return (
        Vector(r[0], r[1]),
        Vector(r[2], r[3]),
        Vector(r[4], r[5]),
        Vector(r[6], r[7])
    )

def intersection_area(r1, r2,area1,area2):

#    rect1 = rectangle_vertices(*r1)
#    rect2 = rectangle_vertices(*r2)

    rect1 = poly2rect(r1)
    rect2 = poly2rect(r2)

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

def rbbox_overlaps_torch(boxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps

    polys = RotBox2Polys_torch(boxes)
    query_polys = RotBox2Polys_torch(query_boxes)

    h_bboxes = poly2bbox_torch(polys)
    h_query_bboxes = poly2bbox_torch(query_polys)

    # hious
    ious = bbox_overlaps(h_bboxes, h_query_bboxes)
    import pdb
    # pdb.set_trace()
    inds = (ious > 0).int().nonzero()
    
    for index in range(inds.shape[0]):
        box_index = inds[index][0]
        query_box_index = inds[index][1]

        box = polys[box_index]
        query_box = query_polys[query_box_index]
        
        area1 = boxes[box_index][2] * boxes[box_index][3]
        area2 = query_boxes[query_box_index][2] * query_boxes[query_box_index][3]
        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = intersection_area(box,query_box, area1,area2)
        ious[box_index][query_box_index] = overlap
        '''
        plt.scatter(box[::2].detach().cpu().numpy(),box[1::2].detach().cpu().numpy(),c='red')
        plt.scatter(query_box[::2].detach().cpu().numpy(),query_box[1::2].detach().cpu().numpy(),c='blue')
        plt.title(str(overlap))
        plt.savefig('./r3det_test/%s'%(randomString(8)))
        plt.clf()
        '''
    return ious


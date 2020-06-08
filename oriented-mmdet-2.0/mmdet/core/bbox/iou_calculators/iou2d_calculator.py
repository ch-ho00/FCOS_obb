import torch
from .builder import IOU_CALCULATORS
import DOTA_devkit.polyiou as polyiou
import numpy as np
import matplotlib.pyplot as plt 
import string
import random
@IOU_CALCULATORS.register_module()
class BboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str



@IOU_CALCULATORS.register_module()
class RBboxOverlaps2D(object):
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <x_ct, y_ct, w, h, angle>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <x_ct, y_ct, w, h , angle>
                format, shape (m, 5) in <x_ct, y_ct, w, h , angle, score> format, or be
                empty. If is_aligned is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or iof (intersection
                over foreground).

        Returns:
            ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps(bboxes1, bboxes2)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def RotBox2Polys(dboxes):
    """

    :param dboxes:
    :return:
    """
    cs = torch.cos(dboxes[:, 4])
    ss = torch.sin(dboxes[:, 4])
    w = dboxes[:, 2] 
    h = dboxes[:, 3] 

    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    polys = torch.cat((x1.unsqueeze(1),
                       y1.unsqueeze(1),
                       x2.unsqueeze(1),
                       y2.unsqueeze(1),
                       x3.unsqueeze(1),
                       y3.unsqueeze(1),
                       x4.unsqueeze(1),
                       y4.unsqueeze(1)), 1)

    return polys

def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = polys.reshape(n, 4, 2)[:, :, 0]
    ys = polys.reshape(n, 4, 2)[:, :, 1]

    xmin, _  = torch.min(xs, axis=1)
    ymin, _  = torch.min(ys, axis=1)
    xmax, _  = torch.max(xs, axis=1)
    ymax, _  = torch.max(ys, axis=1)

    xmin = xmin.unsqueeze(1)
    ymin = ymin.unsqueeze(1)
    xmax = xmax.unsqueeze(1)
    ymax = ymax.unsqueeze(1)

    return torch.cat((xmin, ymin, xmax, ymax), 1)

def randomString(stringLength=8):

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))



def rbbox_overlaps(boxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps

    polys = RotBox2Polys(boxes) #.cpu().numpy() #.astype(np.float)
    query_polys = RotBox2Polys(query_boxes) #.cpu().numpy() #.astype(np.float)

    h_bboxes = poly2bbox(polys) #.astype(np.float)
    h_query_bboxes = poly2bbox(query_polys) #.astype(np.float)

    # hious
    ious = bbox_overlaps(h_bboxes, h_query_bboxes).cpu().numpy()
    import pdb
    # pdb.set_trace()
    #print(query_boxes.shape, boxes.shape,ious.shape,np.max(ious), type(ious),'\n\n',query_boxes[11000:11010],'\n\n',h_query_bboxes[11000:11010],'\n__________________\n', flush=True)
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys[box_index].cpu().numpy().astype(np.float)
        query_box = query_polys[query_box_index].cpu().numpy().astype(np.float)

        h_box = h_bboxes[box_index].cpu().numpy()
        h_query = h_query_bboxes[query_box_index].cpu().numpy()
        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        '''
        if overlap > 0.1 and query_boxes[query_box_index][4] != 0:
            plt.plot(box[::2], box[1::2],color='red')
            plt.plot(query_box[::2], query_box[1::2],color='blue')
            plt.plot(h_box[::2], h_box[1::2])
            plt.plot(h_query[::2], h_query[1::2])
            plt.title(str([int(a) for a in query_boxes[query_box_index][:4]])+" "+str(query_boxes[query_box_index][4])+" "+str(overlap))
            plt.savefig('./check/'+randomString(4)+'.png')
            plt.clf()
        ''' 
        ious[box_index][query_box_index] = overlap
    return torch.Tensor(ious).to(boxes.device)

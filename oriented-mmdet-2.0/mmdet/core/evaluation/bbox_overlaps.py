import numpy as np
import torch
from mmdet.ops.rotated.rotate_nms import rotate_iou as get_iou_matrix

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']
    annotation = bboxes2.shape[-1]
    assert annotation == 4 or annotation ==5 
    if isinstance(bboxes1, torch.Tensor):
        bboxes1_np = bboxes1.cpu().detach().numpy()
        bboxes2_np = bboxes2.cpu().detach().numpy()
    else:
        bboxes1_np = bboxes1
        bboxes2_np = bboxes2
    bboxes1_np = bboxes1_np.astype(np.float32)
    bboxes2_np = bboxes2_np.astype(np.float32)
    rows = bboxes1_np.shape[0]
    cols = bboxes2_np.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    if annotation == 4:
        exchange = False
        if bboxes1_np.shape[0] > bboxes2_np.shape[0]:
            bboxes1_np, bboxes2_np = bboxes2_np, bboxes1_np
            ious = np.zeros((cols, rows), dtype=np.float32)
            exchange = True
        area1 = (bboxes1_np[:, 2] - bboxes1_np[:, 0]) * (bboxes1_np[:, 3] - bboxes1_np[:, 1])
        area2 = (bboxes2_np[:, 2] - bboxes2_np[:, 0]) * (bboxes2_np[:, 3] - bboxes2_np[:, 1])
        for i in range(bboxes1_np.shape[0]):
            x_start = np.maximum(bboxes1_np[i, 0], bboxes2_np[:, 0])
            y_start = np.maximum(bboxes1_np[i, 1], bboxes2_np[:, 1])
            x_end = np.minimum(bboxes1_np[i, 2], bboxes2_np[:, 2])
            y_end = np.minimum(bboxes1_np[i, 3], bboxes2_np[:, 3])
            overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
                y_end - y_start + 1, 0)
            if mode == 'iou':
                union = area1[i] + area2 - overlap
            else:
                union = area1[i] if not exchange else area2
            ious[i, :] = overlap / union
        if exchange:
            ious = ious.T
    else:
    #    bboxes1_np[:,4] = bboxes1_np[:,4] * np.pi / 180
    #    bboxes2_np[:,4] = bboxes2_np[:,4] * np.pi / 180
        ious = get_iou_matrix(torch.from_numpy(bboxes1_np), torch.from_numpy(bboxes2_np)).numpy()
    return torch.Tensor(ious)


if __name__ == "__main__":
    bbox1 = np.array([[5,5,10,10, -np.pi/4],[5,5,10,10,np.pi/4]])
    bbox2 = np.array([[1,1,10,10,-np.pi/2],[1,1,10,10, np.pi/2]] )
    print(bbox_overlaps(bbox1, bbox2))    


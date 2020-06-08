import torch

from mmdet.ops.nms import batched_nms
from mmdet.ops.nms import nms_wrapper
from mmdet.core import RotBox2Polys

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # TODO: Check if -1 is necessary
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    annotation = multi_bboxes.shape[-1] 
    annotation = 4 if annotation % 4 == 0 else 5
    if annotation == 4:
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > score_thr
        bboxes = bboxes[valid_mask]
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = scores[valid_mask]
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
            return bboxes, labels

        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
        return dets, labels[keep]
    else:
        bboxes, labels = [], []
        nms_cfg_ = nms_cfg.copy()
        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_type)
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # get bboxes and scores of this class
            if multi_bboxes.shape[1] == 5:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 5:(i + 1) * 5]
            _scores = multi_scores[cls_inds, i]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                            i - 1,
                                            dtype=torch.long)
            cls_dets = torch.from_numpy(RotBox2Polys(cls_dets.detach().cpu().numpy())).to(_bboxes.device)
            bboxes.append(cls_dets)
            labels.append(cls_labels)
        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds]
                labels = labels[inds]
        else:
            bboxes = multi_bboxes.new_zeros((0, 9))
            labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        return bboxes, labels
        

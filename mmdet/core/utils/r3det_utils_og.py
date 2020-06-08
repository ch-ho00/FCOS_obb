import torch
import numpy as np 
import cv2 
from mmdet.core.bbox import rbbox_overlaps_cy

import random
def coordinate_present_convert(coords, mode=1):
    """
    :param coords: shape [-1, 5]
    :param mode: -1 convert coords range to [-90, 90), 1 convert coords range to [-90, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [-180, 0)
    if mode == -1:
        w, h = coords[:, 2], coords[:, 3]

        remain_mask = np.greater(w, h)
        convert_mask = np.logical_not(remain_mask).astype(np.int32)
        remain_mask = remain_mask.astype(np.int32)

        remain_coords = coords * np.reshape(remain_mask, [-1, 1])

        coords[:, [2, 3]] = coords[:, [3, 2]]
        coords[:, 4] += 90

        convert_coords = coords * np.reshape(convert_mask, [-1, 1])

        coords_new = remain_coords + convert_coords

        coords_new[:, 4] -= 90

    # angle range from [-180, 0) to [-90, 0)
    elif mode == 1:
        coords[:, 4] += 90

        # theta = coords[:, 4]
        # remain_mask = np.logical_and(np.greater_equal(theta, -90), np.less(theta, 0))
        # convert_mask = np.logical_not(remain_mask)
        #
        # remain_coords = coords * np.reshape(remain_mask, [-1, 1])
        #
        # coords[:, [2, 3]] = coords[:, [3, 2]]
        # coords[:, 4] -= 90
        #
        # convert_coords = coords * np.reshape(convert_mask, [-1, 1])
        #
        # coords_new = remain_coords + convert_coords

        xlt, ylt = -1 * coords[:, 2] / 2.0, coords[:, 3] / 2.0
        xld, yld = -1 * coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrd, yrd = coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrt, yrt = coords[:, 2] / 2.0, coords[:, 3] / 2.0

        theta = -coords[:, -1] / 180 * np.pi

        xlt_ = np.cos(theta) * xlt + np.sin(theta) * ylt + coords[:, 0]
        ylt_ = -np.sin(theta) * xlt + np.cos(theta) * ylt + coords[:, 1]

        xrt_ = np.cos(theta) * xrt + np.sin(theta) * yrt + coords[:, 0]
        yrt_ = -np.sin(theta) * xrt + np.cos(theta) * yrt + coords[:, 1]

        xld_ = np.cos(theta) * xld + np.sin(theta) * yld + coords[:, 0]
        yld_ = -np.sin(theta) * xld + np.cos(theta) * yld + coords[:, 1]

        xrd_ = np.cos(theta) * xrd + np.sin(theta) * yrd + coords[:, 0]
        yrd_ = -np.sin(theta) * xrd + np.cos(theta) * yrd + coords[:, 1]

        convert_box = np.transpose(np.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))

        coords_new = backward_convert(convert_box, False)


    else:
        raise Exception('mode error!')

    return torch.tensor(coords_new).float()


def backward_convert(coordinate, with_label=False):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)



def rbbox_transform(ex_rois, gt_rois, scale_factors=None):

    targets_dx = (gt_rois[:, 0] - ex_rois[:, 0]) / ex_rois[:, 2]
    targets_dy = (gt_rois[:, 1] - ex_rois[:, 1]) / ex_rois[:, 3]
    targets_dw = np.log(gt_rois[:, 2] / ex_rois[:, 2])
    targets_dh = np.log(gt_rois[:, 3] / ex_rois[:, 3])
#    print("rbbox_trans:",gt_rois[:5],ex_rois[:5],'---------------\n',flush=True)
    targets_dtheta = (gt_rois[:, 4] - ex_rois[:, 4]) 

    if scale_factors:
        targets_dx *= scale_factors[0]
        targets_dy *= scale_factors[1]
        targets_dw *= scale_factors[2]
        targets_dh *= scale_factors[3]
        targets_dtheta *= scale_factors[4]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dtheta)).transpose()

    return targets

def anchor_target_layer(gt_boxes_h, gt_boxes_r, anchors,gt_labels, iou_p_th, iou_n_th, gpu_id=0):
    '''
    gt_boxes_r = (N,5) 
    '''
    anchor_states = np.zeros((anchors.shape[0],))
    labels = np.zeros((anchors.shape[0],))
    if gt_boxes_r.shape[0]:
        # [N, M]
        overlaps = rbbox_overlaps_cy(np.ascontiguousarray(anchors, dtype=np.float32),
                                    np.ascontiguousarray(gt_boxes_r, dtype=np.float32))

        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # compute box regression targets
        target_boxes = gt_boxes_r[argmax_overlaps_inds]
        target_labels = gt_labels[argmax_overlaps_inds]

        delta_theta = np.abs(target_boxes[:, -1] - anchors[:, -1])
        theta_indices = delta_theta < 15 *np.pi / 180

        positive_indices = (max_overlaps >= iou_p_th) & theta_indices
        ignore_indices = (max_overlaps > iou_n_th) & (max_overlaps < iou_p_th)

        anchor_states[ignore_indices] = -1
        anchor_states[positive_indices] = 1
        # compute target class labels
        labels[positive_indices] = target_labels[positive_indices].astype(int) - 1
    else:
        # no annotations? then everything is background
        target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))
    '''
    if cfgs.ANGLE_RANGE == 180:
        anchors = coordinate_present_convert(anchors, mode=-1)
        target_boxes = coordinate_present_convert(target_boxes, mode=-1)
    '''
    target_delta = rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)
    return torch.tensor(labels).long().cuda(), torch.tensor(target_delta).float().cuda(), \
           torch.tensor(anchor_states).float().cuda(), torch.tensor(target_boxes).float().cuda()


def refinebox_target_layer(gt_boxes_r, anchors,gt_labels, pos_threshold, neg_threshold, gpu_id=0):

    anchor_states = np.zeros((anchors.shape[0],))
    labels = np.zeros((anchors.shape[0],))
    if gt_boxes_r.shape[0]:
        # [N, M]

        overlaps = rbbox_overlaps_cy(np.ascontiguousarray(anchors, dtype=np.float32),
                                    np.ascontiguousarray(gt_boxes_r, dtype=np.float32))

        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # compute box regression targets
        target_boxes = gt_boxes_r[argmax_overlaps_inds]
        target_labels = gt_labels[argmax_overlaps_inds]

        positive_indices = max_overlaps >= pos_threshold
        ignore_indices = (max_overlaps > neg_threshold) & ~positive_indices
        anchor_states[ignore_indices] = -1
        anchor_states[positive_indices] = 1

        # compute target class labels
        labels[positive_indices] = target_labels[positive_indices].astype(int) - 1
    else:
        # no annotations? then everything is background
        target_boxes = np.zeros((anchors.shape[0], gt_boxes_r.shape[1]))

    target_delta = rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)

    return torch.tensor(labels).long().cuda(), torch.tensor(target_delta).float().cuda(), \
           torch.tensor(anchor_states).float().cuda(), torch.tensor(target_boxes).float().cuda()


def filter_detections(boxes, scores, vis_score=None, filter_score=None, nms=None,nms_iou_th=None,is_training=None):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = ((scores > vis_score).int().nonzero()).reshape(-1)
        # indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
    else:
        indices = ((scores > filter_score).int().nonzero()).reshape(-1)
        # indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

    if nms:
        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]
        # filtered_boxes = tf.gather(boxes, indices)
        # filtered_scores = tf.gather(scores, indices)

        # perform NMS
        print('nms on',filtered_boxes.shape, filtered_scores.shape,flush=True)
        nms_indices = nms_rotate(boxes=filtered_boxes,
                                            scores=filtered_scores,
                                            iou_threshold=nms_iou_th,
                                            max_output_size=100)

        # filter indices based on NMS
        indices = indices[nms_indices.tolist()]
        # indices = tf.gather(indices, nms_indices)

    # add indices to list of all indices
    return indices

def nms_rotate(boxes, scores, iou_threshold, max_output_size):

    keep = []
#    order = scores.argsort()[::-1]
    order = torch.argsort(scores,dim=-1,descending=True)
    num = boxes.shape[0]
    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        print(_i,flush=True)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0
            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area = cv2.contourArea(order_pts)
                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)
            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                inter = 0.9999
            if inter >= iou_threshold:
                suppressed[j] = 1
    return torch.tensor(keep).int()

# lib/box_utils/bbox_transform
def rbbox_transform_inv(boxes, deltas, scale_factors=None):

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dtheta = deltas[:, 4]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]
        dtheta /= scale_factors[4]

    pred_ctr_x = dx * boxes[:, 2] + boxes[:, 0]
    pred_ctr_y = dy * boxes[:, 3] + boxes[:, 1]
    pred_w = dw.exp() * boxes[:, 2]
    pred_h = dh.exp() * boxes[:, 3]

    pred_theta = dtheta  + boxes[:, 4]

    return torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_theta]).t()



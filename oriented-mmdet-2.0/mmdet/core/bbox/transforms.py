import mmcv
import numpy as np
import torch
import pycocotools.mask as mask_util


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        annotation = bboxes.shape[-1]
        if annotation % 4 == 0: 
            flipped = bboxes.clone()
            flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
            flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
            return flipped
        elif annotation % 5 == 0: 
            w = img_shape[1]
            bboxes[..., 0::5] = w - bboxes[..., 0::5]
            bboxes[..., 4::5] = -bboxes[..., 4::5]
            return bboxes

    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if bboxes.shape[-1] % 5 == 0: 
        bboxes[..., 4::5] = bboxes[..., 4::5] / scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    if bboxes.shape[-1] % 5  == 0: 
        new_bboxes[..., 4::5] = new_bboxes[..., 4::5] * scale_factor
    return new_bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    annotation = bbox_list[0].size(-1) - 1 ##

    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :annotation]], dim=-1) ## 
        else:
            rois = bboxes.new_zeros((0, annotation + 1)) ## 
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    annotation = bboxes.size(-1)
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, annotation + 1), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

'''bbox and mask 转成result mask要画图'''
def bbox_mask2result(bboxes, masks, labels, num_classes, img_meta):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        masks (Tensor): shape (n, 2, 36)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    ori_shape = img_meta['ori_shape']
    img_h, img_w, _ = ori_shape

    mask_results = [[] for _ in range(num_classes - 1)]

    for i in range(masks.shape[0]):
        im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask = [masks[i].transpose(1,0).unsqueeze(1).int().data.cpu().numpy()]
        im_mask = cv2.drawContours(im_mask, mask, -1,1,-1)
        rle = mask_util.encode(
            np.array(im_mask[:, :, np.newaxis], order='F'))[0]

        label = labels[i]

        mask_results[label].append(rle)


    if bboxes.shape[0] == 0:
        bbox_results = [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
        return bbox_results, mask_results
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        bbox_results = [bboxes[labels == i, :] for i in range(num_classes - 1)]
        return bbox_results, mask_results

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)


def RotBox2Polys(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    if dboxes.shape[-1] == 6:
        scores = dboxes[:,-1][:, np.newaxis]
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
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

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]
    
    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4, scores), axis=1)
    return polys


def distance2rbbox(points, dist, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom, angle). 
            Remark. Angles should be in radian value
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes (x_ct, y_ct, w, h, angle).
    """
    t,r,b,l,th = dist[:,0], dist[:,1], dist[:,2], dist[:,3],dist[:,4]
    x,y = points[:,0], points[:,1]

    cos_t = torch.cos(angle);  sin_t =torch.sin(angle)
    t_cos = t * cos_t 
    t_sin = t * sin_t
    r_cos = r * cos_t 
    r_sin = r * sin_t
    b_cos = b * cos_t
    b_sin = b * sin_t
    l_cos = l * cos_t 
    l_sin = l * sin_t

    #coords = [x-b_cos-l_sin, y-b_sin+l_cos, x+r_sin-b_cos, y-r_cos-b_sin, x+t_cos+ r_sin, y+t_sin-r_cos, x-l_sin + t_cos, y + l_cos + t_sin]
    x_new = x + (t_cos + r_sin - b_cos - l_sin)/2
    y_new = y + (t_sin - r_cos - b_sin + l_cos)/2
    h = t+b
    w = r+l
    return torch.cat([x_new.unsqueeze(1),y_new.unsqueeze(1),
        h.unsqueeze(1),w.unsqueeze(1),th.unsqueeze(1)],-1)

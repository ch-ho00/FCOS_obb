from .coco import CocoDataset
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import Polygon as plg
from tqdm import tqdm

from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
import os.path as osp
import warnings

import mmcv
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import torch
from mmdet.core import poly2bbox,RotBox2Polys,gt_mask_bp_obbs,distance2obb ,bbox_overlaps
from .extra_aug import ExtraAugmentation
from .registry import DATASETS
from .transforms import (BboxTransform, ImageTransform, MaskTransform,
                         Numpy2Tensor, SegMapTransform, SegmapTransform)
from .utils import random_scale, to_tensor
from IPython import embed
import time
#remove later 
import DOTA_devkit.polyiou as polyiou

INF = 1e8


def get_angle(v1, v2=[0,0,100,0]):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    included_angle = angle2 - angle1
    if included_angle < 0:
        included_angle += 360
    return included_angle


@DATASETS.register_module
class DOTADataset(CocoDataset):

    CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        #print(ann_info,"!!!!!!!!!!!!!!", flush=True)
        return self._parse_ann_info(ann_info) # ,self.with_mask

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
#        valid_inds = valid_inds[:1000]
        return valid_inds
    def convert(self,mask):
        return  gt_mask_bp_obbs(mask)


    def convert_v2(self,bbox, with_module=True):
        """
        :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
                shape [num_boxes, 8]
        :return: Rotated Rectangle in format [cx, cy, w, h, theta]
                shape [num_rot_recs, 5]
        """
        og_bb = np.int0(bbox)
        og_bb = og_bb.reshape([4,2])
        bbox = np.array(bbox,dtype=np.float32)
        bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
        rect1 = cv2.minAreaRect(og_bb)

        # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
        # print('bbox: ', bbox)
        angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
        # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
        # center = [[0],[0]] ## shape [2, 1]
        # print('angle: ', angle)
        center = np.zeros((bbox.shape[0], 2, 1))
        for i in range(4):
            center[:, 0, 0] += bbox[:, 0,i]
            center[:, 1, 0] += bbox[:, 1,i]

        center = np.array(center,dtype=np.float32)/4.0

        # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

        normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


        xmin = np.min(normalized[:, 0, :], axis=1)
        # print('diff: ', (xmin - normalized[:, 0, 3]))
        # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
        xmax = np.max(normalized[:, 0, :], axis=1)
        # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
        # print('diff2: ', xmax - normalized[:, 0, 1])
        ymin = np.min(normalized[:, 1, :], axis=1)
        # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
        # print('diff3: ', ymin - normalized[:, 1, 3])
        ymax = np.max(normalized[:, 1, :], axis=1)
        # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
        # print('diff4: ', ymax - normalized[:, 1, 1])

        w = xmax - xmin + 1
        h = ymax - ymin + 1

        w = w[:, np.newaxis]
        h = h[:, np.newaxis]
        # TODO: check it
        theta = rect1[2] *np.pi / 180
        theta = np.array([theta])
        if np.abs(angle) > np.pi/2-0.1:
            tmp = w
            w = h
            h = tmp
            if angle > 0:
                angle = angle - np.pi/2
            else:
                angle = angle + np.pi/2
        if with_module:
            angle = theta[:, np.newaxis] 
        else:
            angle = theta[:, np.newaxis]
        dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w,h, angle), axis=1)
#        print('Before\n',bbox,'\n\n',og_bb,'\n\nafter',dboxes,og,flush=True)
        return dboxes[0]
                  

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_obbs = [] 
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.


        self.debug = False


        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        if self.debug:
            count = 0
            total = 0
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            [coordinates] = ann['segmentation']
            #filter bbox < 10
            if self.debug:
                total+=1
            if ann['area'] <= 15  or self.coco.annToMask(ann).sum() < 15:
                # print('filter, area:{},w:{},h:{}'.format(ann['area'],w,h))
                if self.debug:
                    count+=1
                continue
            [bbox] = self.convert(self.coco.annToMask(ann))
            xc,yc,w,h,th = bbox
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_obbs.append(coordinates)
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
   
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)          

        gt_hbbs = poly2bbox(np.array(gt_obbs))
        gt_hbb2obb = np.zeros_like(gt_hbbs)
        gt_hbb2obb[:,0] = (gt_hbbs[:,0] + gt_hbbs[:,2])/2
        gt_hbb2obb[:,1] = (gt_hbbs[:,1] + gt_hbbs[:,3])/2
        gt_hbb2obb[:,2] = gt_hbbs[:,2] - gt_hbbs[:,0]
        gt_hbb2obb[:,3] = gt_hbbs[:,3] - gt_hbbs[:,1]
        gt_hbb2obb = torch.cat([torch.tensor(gt_hbb2obb).float() , (torch.ones(gt_hbbs.shape[0],1)*np.pi*3/2).float()],1)
        if self.debug:
            print('filter:',count/total)
        if gt_bboxes:
            gt_bboxes = np.array(gt_hbb2obb, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_obbs = np.array(gt_obbs, dtype=np.float32)
           
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_obbs = np.array((0,8),dtype=np.float32)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore,obbs= gt_obbs, hbbs=gt_hbbs
        )
        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)

        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # apply transforms
        flip = False #True if np.random.rand() < self.flip_ratio else False

        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip, keep_ratio=self.resize_keep_ratio)


        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix,
                         img_info['filename'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        ann['hbbs'] = self.bbox_transform(ann['hbbs'], img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
#        if self.with_mask:
#            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        '''
        plt.imshow(np.transpose(img,(1,2,0)))
        for i in range(ann['hbbs'].shape[0]):
            plt.scatter(ann['hbbs'][i,::2], ann['hbbs'][i,1::2])
        plt.savefig('./tests/'+img_info['filename'][:-4]+"_hbb.png")
        plt.close()
        plt.imshow(np.transpose(img,(1,2,0)))
        for i in range(ann['hbbs'].shape[0]):
            plt.scatter(ann['obbs'][i,::2], ann['obbs'][i,1::2])
        plt.savefig('./tests/'+img_info['filename'][:-4]+"_obb.png")
        plt.close()
        plt.imshow(np.transpose(img,(1,2,0)))
        pts = RotBox2Polys(ann['bboxes'])
        for i in range(ann['hbbs'].shape[0]):
            plt.scatter(pts[i,::2], pts[i,1::2])
        plt.title(str(ann['bboxes'][0,4]*180/np.pi))
        plt.savefig('./tests/'+img_info['filename'][:-4]+"_converted_obb.png")
        plt.close()
        '''
        #--------------------offline ray label generation-----------------------------

        self.center_sample = True
        self.use_mask_center = True
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        featmap_sizes = self.get_featmap_size(pad_shape)
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        self.num_points_per_level = [i.size()[0] for i in all_level_points]

        expanded_regress_ranges = [
            all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                all_level_points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)
        gt_masks = gt_masks[:len(gt_bboxes)]

        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_labels = torch.Tensor(gt_labels)

#        print( gt_bboxes.shape,gt_masks.shape,gt_labels.shape,ann['hbbs'].shape,"before polar target single!!!",flush=True)
        _labels, _bbox_targets  = self.polar_target_single(
            gt_bboxes,gt_masks,gt_labels,concat_points, concat_regress_ranges,ann['hbbs']) # _mask_targets
       # print(gt_bboxes,"\n\n" ,_bbox_targets.shape,"\n\n",flush=True)
        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)
#        data['l1_targets'] = DC(l1_targets)
#        data['_gt_masks'] = DC(_mask_targets)
        #--------------------offline ray label generation-----------------------------


        return data

    def get_featmap_size(self, shape):
        h,w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([int(h / i), int(w / i)])
        return featmap_sizes

    def get_points(self, featmap_sizes):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i]))
        return mlvl_points

    def get_points_single(self, featmap_size, stride):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride)
        y_range = torch.arange(
            0, h * stride, stride)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points.float()

    def polar_target_single(self, gt_bboxes, gt_masks, gt_labels, points, regress_ranges,gt_hbbs):
        '''
        gt_bboxes = (n,5)
        gt_masks = (n,1024,1024)
        points = (M, 2)
        gt_hbbs = (n,4)
        '''
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        gt_hbbs = torch.tensor(gt_hbbs).float()
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 5))

        areas = (gt_hbbs[:, 2] - gt_hbbs[:, 0] + 1) * (
            gt_hbbs[:, 3] - gt_hbbs[:, 1] + 1)

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_hbbs = gt_hbbs[None].expand(num_points,num_gts,4)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        #xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_hbbs[..., 0]
        right = gt_hbbs[..., 2] - xs
        top = ys - gt_hbbs[..., 1]
        bottom = gt_hbbs[..., 3] - ys
        bbox_targets_og = torch.stack((left, top, right, bottom), -1)   #feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]



        #mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []
        #第一步 先算重心  return [num_gt, 2]


        for mask in gt_masks:
            cnt, contour = self.get_single_centerpoint(mask)
            contour = contour[0]
            contour = torch.Tensor(contour).float()
            y, x = cnt
            mask_centers.append([x,y])
            mask_contours.append(contour)
        mask_centers = torch.Tensor(mask_centers).float()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        #加入center sample
        if self.center_sample:
            strides = [8, 16, 32, 64, 128]
            if self.use_mask_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(gt_hbbs,
                                                             mask_centers,
                                                             strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
            else:
                inside_gt_bbox_mask = self.get_sample_region(gt_hbbs,
                                                             strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
        else:
            inside_gt_bbox_mask = bbox_targets_og.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets_og.max(-1)[0]

        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
         
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0         #[num_gt] 介于0-80
#        bboxes = gt_bboxes[min_area_inds]
#        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        pos_inds = labels.nonzero().reshape(-1)

        bbox_targets = torch.zeros(num_points, 5).float()
#        l1_targets = torch.zeros(num_points, 5).float()
        nonzero = 0 
        inmask = 0
        pos_mask_ids = min_area_inds[pos_inds]
        for p,id in zip(pos_inds, pos_mask_ids):
            x, y = points[p]
            pos_mask_contour = mask_contours[id]
            mask = pos_mask_contour[:,0,:]
 
            x_ct , y_ct, _ , _ , theta = gt_bboxes[0,id] 
            dist,coords = self.get_5_coordinates(x,y,pos_mask_contour,theta,str(p),str(id))
            inmask += 1
            '''
            polys = RotBox2Polys(gt_bboxes[0,id].unsqueeze(0))
            pts = list([i.squeeze(0).squeeze(0) for i in  coords.values()])
            pts = torch.stack(pts)
            
            if len(coords) == 3:
                
                plt.scatter(mask[:,0].cpu().numpy(), mask[:,1].cpu().numpy())
                plt.scatter(polys[0,::2],polys[0,1::2])
                plt.scatter(pts[:,0], pts[:,1])
        #     plt.scatter(pts_og[:,0],pts_og[:,1])
                plt.scatter(x,y)
                plt.scatter(x_ct,y_ct)
                plt.figtext(.2,.2,str(coords))
                plt.title(str(int(theta/np.pi*180)%90))
                plt.savefig("./tests/3pts_%s_%s.png"%(str(p),str(id)))
                plt.close()
            '''
            if len(coords) ==4:
                nonzero += 1
                bbox_targets[p] = dist
            else:
                labels[p] = 0 
                continue            
            '''
            polys = RotBox2Polys(gt_bboxes[0,id].unsqueeze(0))
            pts = list([i.squeeze(0).squeeze(0) for i in  coords.values()])
            pts = torch.stack(pts)
            

            plt.scatter(mask[:,0].cpu().numpy(), mask[:,1].cpu().numpy())
            plt.scatter(polys[0,::2],polys[0,1::2])
            plt.scatter(pts[:,0], pts[:,1])
       #     plt.scatter(pts_og[:,0],pts_og[:,1])
            plt.scatter(x,y)
            plt.scatter(x_ct,y_ct)
            plt.figtext(.2,.2,str(coords))
            plt.title(str(int(theta/np.pi*180)%90))
            plt.savefig("./tests/polar_target_%s_%s.png"%(str(p),str(id)))
            plt.close()
            re_obb = distance2obb(points[p].unsqueeze(0),dist.unsqueeze(0))
            re_polys = RotBox2Polys(re_obb)           
            iou = self.skew_iou(re_obb,gt_bboxes[0,id].unsqueeze(0))

#            print("x=",x," | y=",y," \n Ground Truth OBB =",gt_bboxes[0,id],"\n Reconstructed OBB",re_obb,"\n------",flush=True)


            plt.scatter(polys[0,::2],polys[0,1::2])
            plt.scatter(re_polys[0,::2],re_polys[0,1::2])
            plt.scatter(x,y)
            plt.scatter(re_obb[0,0],re_obb[0,1])
            plt.title(str(iou))
            plt.savefig('./tests/praful_distance2obb_%s_%s.png'%(str(p),str(id)))
            plt.close()
            '''
        #print(nonzero,inmask,num_points,labels.nonzero().shape ,bbox_targets.nonzero().shape,torch.max(bbox_targets),flush=True)
        #print(bbox_targets,flush=True)
        return labels, bbox_targets #, mask_targets
  
    def skew_iou(self,pred_box,gt_box,eps=1e-6):
        # first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
        # from 5 pt representation to 8pt polygon
        gt_box = gt_box.detach().cpu().numpy()
        pred_box = pred_box.detach().cpu().numpy()
        gt_poly = RotBox2Polys(gt_box).astype(np.float)
        pred_poly = RotBox2Polys(pred_box).astype(np.float)
    #    print("skew_iou_convert\n\n",gt_poly,"\n\nbefore",gt_box,"____________\n",flush=True)

        # polygon to hbb
        h_bboxes_np = torch.Tensor(poly2bbox(gt_poly))
        h_query_bboxes_np = torch.Tensor(poly2bbox(pred_poly))
        # hious
        ious = bbox_overlaps(h_query_bboxes_np, h_bboxes_np)
    #    print("\n\n\n",ious,"*****************\n\n\n",flush=True)
        import pdb
        # pdb.set_trace()
        inds = np.where(ious > 0)
        iou =  0
        overlap = eps
        for index in range(len(inds[0])):
            box_index = inds[1][index]
            query_box_index = inds[0][index]

            box = gt_poly[box_index]
            query_box = pred_poly[query_box_index]
            overlap = polyiou.iou_poly( polyiou.VectorDouble(query_box),polyiou.VectorDouble(box))
            '''        
          #  print("start--------\n",box,"\n",query_box,"\n",overlap,-np.log(overlap),"\n-------------end",flush=True)
            plt.scatter(box[::2],box[1::2])
            plt.scatter(query_box[::2],query_box[1::2])
            plt.title(str(overlap))
            plt.savefig('./tests/'+"skew_iou_"+str(index)+".png")
            plt.close()
            #ious[box_index][query_box_index] = overlap
            '''
            if overlap < eps:
                overlap = eps
        return overlap
    


    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_x = gt[..., 0] 
        center_y = gt[..., 1] 
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)

            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_mask_sample_region(self, gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        #no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level,n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def get_single_centerpoint(self, mask):
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]

        # max_points = 360
        # if len(contour[0]) > max_points:
        #     compress_rate = len(contour[0]) // max_points
        #     contour[0] = contour[0][::compress_rate, ...]
        return center, contour

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
        angle, idx = torch.sort(angle)
        dist = dist[idx]

        #生成36个角度
        new_coordinate = {}
        for i in range(0, 360, 10):
            if i in angle:
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                new_coordinate[i] = d


        distances = torch.zeros(36)

        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[a//10] = 1e-6
            else:
                distances[a//10] = new_coordinate[a]
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist

        return distances, new_coordinate

    def get_5_coordinates(self, c_x, c_y, pos_mask_contour,theta,p=None,id=None):
        ct = pos_mask_contour[:, 0, :]
        x = ct[:, 0] - c_x
        y = ct[:, 1] - c_y
        # angle = np.arctan2(x, y)*180/np.pi
        angle = torch.atan2(x, y) * 180 / np.pi
        angle[angle < 0] += 360
        angle = angle.int()
            
        '''
        plt.scatter(ct[:,0].cpu().numpy(),ct[:,1].cpu().numpy(), c=angle.cpu().numpy(),s=50, cmap='gray')
        plt.scatter(c_x,c_y)
        plt.savefig('./tests/angles_shape_%s_%s.png'%(p,id))
        plt.close()
        ''' 
        # dist = np.sqrt(x ** 2 + y ** 2)
        dist = torch.sqrt(x ** 2 + y ** 2)
#        angle, idx = torch.sort(angle)
#        dist = dist[idx]
        # Radian to Degree
        theta_deg = (theta) * 180 / np.pi
        theta_deg = theta_deg.int() %90 
        if theta_deg != 0:
            theta_deg = 90 - theta_deg        
        
        new_coordinate = {}
        pts = {}
        
        for i in range(theta_deg, 360, 90):
            if i in angle:
                pts[i] = ct[(angle==i).nonzero()[-1]]
                d = dist[angle==i].max()
                new_coordinate[i] = d
            elif i + 1 in angle:
                d = dist[angle == i+1].max()
                new_coordinate[i] = d
                pts[i] = ct[(angle==i+1).nonzero()[-1]]
            elif i - 1 in angle:
                d = dist[angle == i-1].max()
                pts[i] = ct[(angle==i-1).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 2 in angle:
                d = dist[angle == i+2].max()
                pts[i] = ct[(angle==i+2).nonzero()[-1]]
                new_coordinate[i] = d
            elif i - 2 in angle:
                d = dist[angle == i-2].max()
                pts[i] = ct[(angle==i-2).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 3 in angle:
                d = dist[angle == i+3].max()
                pts[i] =ct[(angle==i+3).nonzero()[-1]]
                new_coordinate[i] = d
            elif i - 3 in angle:
                d = dist[angle == i-3].max()
                pts[i] = ct[(angle==i-3).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 4 in angle:
                d = dist[angle == i+4].max()
                pts[i] = ct[(angle==i+4).nonzero()[-1]]
                new_coordinate[i] = d
            elif i - 4 in angle:
                d = dist[angle == i-4].max()
                pts[i] = ct[(angle==i-4).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 5 in angle:
                pts[i] = ct[(angle==i+5).nonzero()[-1]]
                d = dist[angle == i+5].max()
                new_coordinate[i] = d
            elif i - 5 in angle:
                d = dist[angle == i-5].max()
                pts[i] = ct[(angle==i-5).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 6 in angle:
                d = dist[angle == i+6].max()
                pts[i] = ct[(angle==i+6).nonzero()[-1]]
                new_coordinate[i] = d
            elif i - 6 in angle:
                d = dist[angle == i-6].max()
                pts[i] = ct[(angle==i-6).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 7 in angle:
                pts[i] = ct[(angle==i+7).nonzero()[-1]]
                d = dist[angle == i+7].max()
                new_coordinate[i] = d
            elif i - 7 in angle:
                d = dist[angle == i-7].max()
                pts[i] = ct[(angle==i-7).nonzero()[-1]]
                new_coordinate[i] = d
            elif i + 8 in angle:
                d = dist[angle == i+8].max()
                pts[i] = ct[(angle==i+8).nonzero()[-1]]
                new_coordinate[i] = d
            elif i - 8 in angle:
                d = dist[angle == i-8].max()
                pts[i] = ct[(angle==i-8).nonzero()[-1]]
                new_coordinate[i] = d

        distances = torch.zeros(5)

        for j,a in enumerate(range(theta_deg, 360, 90)):
            if not a in new_coordinate.keys():
                new_coordinate[a] = torch.tensor(1e-6)
                distances[j] = 1e-6
            else:
                distances[j] = new_coordinate[a]
        distances[4] = theta #- np.pi * 5/4
        '''
        reordered = 0
        try:
            if pts[range(theta_deg,360,90)[-1]][0][1] > pts[range(theta_deg,360,90)[0]][0][1]:
                reordered =1 
                tmp = distances[3]
                distances[1:4] = distances[0:3]
                distances[0] = tmp
        except:
            pass
        '''
        # for idx in range(36):
        #     dist = new_coordinate[idx * 10]
        #     distances[idx] = dist        print(pos_mask_contour.shape, "???",flush=True)


        return distances, pts  #,pts_og


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data



class DOTADataset_v3(CocoDataset):

    CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter')

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            # This config verified
            # if ann['area'] <= 0 or w < 10 or h < 10:
            #     continue
            # if ann['area'] <= 50 or max(w, h) < 10:
            #     continue
            # TODO: make the threshold a paramater in config
            if ann['area'] <= 80 or max(w, h) < 12:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import matplotlib.pyplot as plt
from mmdet.core import gt_mask_bp_obbs,polygonToRotRectangle_batch,dbbox2mask, RotBox2Polys,distance2bbox, force_fp32, multi_apply, multiclass_nms_rbbox,distance2obb, merge_rotate_aug_bboxes #, multiclass_nms, multiclass_nms_with_mask

from mmdet.ops import ModulatedDeformConvPack

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob, build_norm_layer
from IPython import embed
import cv2
import numpy as np
import math
import time
from math import pi,cos,sin
INF = 1e8


@HEADS.register_module
class PolarMask_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_dcn=False,
                 mask_nms=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
 #                loss_mask=dict(type='MaskIOULoss'),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(PolarMask_Head, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
#        self.loss_mask = build_loss(loss_mask)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        # xez add for polarmask
        self.use_dcn = use_dcn
        self.mask_nms = mask_nms

        # debug vis img
        self.vis_num = 1000
        self.count = 0

        # test
        self.angles = torch.range(0, 350, 10).cuda() / 180 * math.pi

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
#        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if not self.use_dcn:
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
#                self.mask_convs.append(
#                    ConvModule(
#                        chn,
#                        self.feat_channels,
#                        3,
#                        stride=1,
#                        padding=1,
#                        conv_cfg=self.conv_cfg,
#                        norm_cfg=self.norm_cfg,
#                        bias=self.norm_cfg is None))
            else:
                self.cls_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.cls_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.reg_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.reg_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))

#                self.mask_convs.append(
#                    ModulatedDeformConvPack(
#                        chn,
#                        self.feat_channels,
#                        3,
#                        stride=1,
#                        padding=1,
#                        dilation=1,
#                        deformable_groups=1,
#                    ))
#                if self.norm_cfg:
#                    self.mask_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
#                self.mask_convs.append(nn.ReLU(inplace=True))

        self.polar_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.polar_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)
        #self.angle_reg = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
#        self.polar_mask = nn.Conv2d(self.feat_channels, 36, 3, padding=1)
        self.polar_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales_bbox = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scales_mask = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.init_weights()
    def init_weights(self):
        if not self.use_dcn:
            for m in self.cls_convs:
                normal_init(m.conv, std=0.01)
            for m in self.reg_convs:
                normal_init(m.conv, std=0.01)
#            for m in self.mask_convs:
#                normal_init(m.conv, std=0.01)
        else:
            pass

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.polar_cls, std=0.01, bias=bias_cls)
        normal_init(self.polar_reg, std=0.01)
#       normal_init(self.polar_mask, std=0.01)
        normal_init(self.polar_centerness, std=0.01)


    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_bbox, self.scales_mask)

    def forward_single(self, x, scale_bbox, scale_mask):
        cls_feat = x
        reg_feat = x
#        mask_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.polar_cls(cls_feat)
        centerness = self.polar_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = scale_bbox(self.polar_reg(reg_feat))

        #bbox_pred[:,4,:,:] = F.relu(bbox_pred[:,4,:,:])

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        '''
        bbox_pred = self.polar_reg(reg_feat)
        print(bbox_pred.shape, flush=True)
        distances = bbox_pred[:,:4,:,:]
        distances = self.scales_bbox(distances).float().exp()
        angles = bbox_pred[:,4,:,:]
        final_pred = torch.cat([distances,angles.unsqueeze(1),1])
        print(final_pred.shape,'..fin',flush=True)
        '''
#        for mask_layer in self.mask_convs:
#            mask_feat = mask_layer(mask_feat)
#        mask_pred = scale_mask(self.polar_mask(mask_feat)).float().exp()

        return cls_score, bbox_pred, centerness # originally 4nd output = mask_pred

    def bbox2vis(self,img,bboxes,fn):
        bboxes = bboxes.detach().cpu().numpy()
 
        if bboxes.shape[-1] == 5:
            bboxes = RotBox2Polys(bboxes)
            plt.imshow(img)
            for i in range(bboxes.shape[0]):
                plt.scatter(bboxes[i,::2],bboxes[i,1::2])
            plt.savefig(fn)
            plt.close()
        elif bboxes.shape[-1] == 4:
            plt.imshow(img)
            for i in range(bboxes.shape[0]):
                plt.scatter(bboxes[i,::2],bboxes[i,1::2])
            plt.savefig(fn)
            plt.close()


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'mask_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
#             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
#             gt_masks,
             gt_bboxes_ignore=None,
             extra_data=None,
             img=None):
        assert len(cls_scores) == len(centernesses) # == len(mask_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        labels, bbox_targets  = self.polar_target(all_level_points, extra_data) # 3rd ouput = mask_targets
#        print("Polar Target",labels[0].shape, bbox_targets[0].shape,type(labels),type(bbox_targets),len(labels),len(bbox_targets),flush=True)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
#        flatten_mask_preds = [
#            mask_pred.permute(0, 2, 3, 1).reshape(-1, 36)
#            for mask_pred in mask_preds
#        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)  # [num_pixel, 80]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)  # [num_pixel, 5]
 #       flatten_mask_preds = torch.cat(flatten_mask_preds)  # [num_pixel, 36]
        flatten_centerness = torch.cat(flatten_centerness)  # [num_pixel]

        flatten_labels = torch.cat(labels).long()  # [num_pixel]
        flatten_bbox_targets = torch.cat(bbox_targets)  # [num_pixel, 5]
#        flatten_mask_targets = torch.cat(mask_targets)  # [num_pixel, 36]
        flatten_points = torch.cat([points.repeat(num_imgs, 1)
                                    for points in all_level_points])  # [num_pixel,2]
        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
#        pos_mask_preds = flatten_mask_preds[pos_inds]
        
 #       print("almost there\n\n",pos_inds,"!!!!!!!!!!!\n\n", flush=True)
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
#            pos_mask_targets = flatten_mask_targets[pos_inds]
            pos_centerness_targets = self.polar_centerness_target(pos_bbox_targets)

            pos_points = flatten_points[pos_inds]
            '''
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
            '''

#            print("before distance2obb",pos_bbox_preds.shape, pos_bbox_targets.shape,pos_centerness_targets, pos_centerness,flush=True)
            #print(pos_bbox_targets,flush= True)
#            loss_bbox = self.loss_bbox(
#                pos_bbox_preds,
#                pos_bbox_targets,
#                weight=pos_centerness_targets,
#                avg_factor=pos_centerness_targets.sum())
#            pos_decoded_bbox_preds =   distance2obb(pos_points, pos_bbox_preds)
#            pos_decoded_bbox_targets = distance2obb(pos_points, pos_bbox_targets)

   #         print('\n\n\n\n\n\n',pos_bbox_preds,"\n_____________________\n", pos_decoded_bbox_preds,"\n\n\n\n",flush=True)
#            print('\n\n\n\n\n\nGT_BBOX',gt_bboxes,"\n\n",pos_bbox_targets.shape,pos_decoded_bbox_preds.shape,'\n\n',pos_bbox_targets,"\n____________________\n", pos_decoded_bbox_targets,"\n\n\n\n",flush=True)
 #           self.bbox2vis(np.transpose(img[0].detach().cpu().numpy(),(1,2,0)),pos_decoded_target_preds,"after_d2obb.png")

#            self.bbox2vis(np.transpose(img[1].detach().cpu().numpy(),(1,2,0)),pos_decoded_target_preds,"after_d2obb_2.png")
#            self.bbox2vis(np.transpose(img[0].detach().cpu().numpy(),(1,2,0)),gt_bboxes[0],"gt_bbox.png")
#            self.bbox2vis(np.transpose(img[1].detach().cpu().numpy(),(1,2,0)),gt_bboxes[1],"gt_bboxes2.png")

           # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                pos_points,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())

#            loss_mask = self.loss_mask(pos_mask_preds,
#                                       pos_mask_targets,
#                                       weight=pos_centerness_targets,
#                                       avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
 #           loss_mask = pos_mask_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
#            loss_mask=loss_mask,
            loss_centerness=loss_centerness)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def polar_target(self, points, extra_data):
        assert len(points) == len(self.regress_ranges)

        num_levels = len(points)

        labels_list, bbox_targets_list  = extra_data.values() # , mask_targets_list

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
#        mask_targets_list = [
#            mask_targets.split(num_points, 0)
#            for mask_targets in mask_targets_list
#        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
#        concat_lvl_mask_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
 #           concat_lvl_mask_targets.append(
 #               torch.cat(
 #                   [mask_targets[i] for mask_targets in mask_targets_list]))

        return concat_lvl_labels, concat_lvl_bbox_targets # , concat_lvl_mask_targets

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets[:,:4].min(dim=-1)[0] / pos_mask_targets[:,:4].max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
#                   mask_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        #print(self.scales_bbox.state_dict(),"????????????",flush=True)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        if len(img_metas) > 1:
            print("Error",len(img_metas),flush=True)
            exit()
        elif len(img_metas) == 1:
            cls_score_list = [
                        cls_scores[i][0].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][0].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][0].detach() for i in range(num_levels)
            ]
#            mask_pred_list = [
#                mask_preds[i][0].detach() for i in range(num_levels)
#            ]
            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
#                                                mask_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            return det_bboxes, det_labels           

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
 #                         mask_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip( # mask_pred,
                cls_scores, bbox_preds, centernesses, mlvl_points): # mask_preds,
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
          
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
#            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, 36)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
#                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            #print(bbox_pred,'\n\n\n',flush=True)
#            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            bboxes = torch.stack([ dist2obb(box,p) for p, box in zip(points,bbox_pred)])
#            masks = distance2mask(points, mask_pred, self.angles, max_shape=img_shape)
#            print(points,bbox_pred, bboxes," distance2obb",flush=True)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
#            mlvl_masks.append(masks)

        mlvl_bboxes = torch.cat(mlvl_bboxes).float()
#        mlvl_masks = torch.cat(mlvl_masks)
        _mlvl_bboxes = mlvl_bboxes   #;  _mlvl_masks = mlvl_masks
        if rescale:
            _mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
#            try:
#                scale_factor = torch.Tensor(scale_factor)[:2].cuda().unsqueeze(1).repeat(1, 36)
#                _mlvl_masks = mlvl_masks / scale_factor
#            except:
#                _mlvl_masks = mlvl_masks / mlvl_masks.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        centerness_factor = 0.5  # mask centerness is smaller than origin centerness, so add a constant is important or the score will be too low.
        #  Rbox NMS
        det_bboxes , det_labels= multiclass_nms_rbbox(_mlvl_bboxes, mlvl_scores,cfg.score_thr, cfg.nms, cfg.max_per_img)
#        if self.mask_nms:
#            '''1 mask->min_bbox->nms, performance same to origin box'''
#            a = _mlvl_masks
#            _mlvl_bboxes = torch.stack([a[:, 0].min(1)[0],a[:, 1].min(1)[0],a[:, 0].max(1)[0],a[:, 1].max(1)[0]],-1)
 #           det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
#                _mlvl_bboxes,
#                mlvl_scores,
#                _mlvl_masks,
#                cfg.score_thr,
#                cfg.nms,
#                cfg.max_per_img,
#                score_factors=mlvl_centerness + centerness_factor)

#        else:
#            '''2 origin bbox->nms, performance same to mask->min_bbox'''
#            det_bboxes, det_labels, det_masks = multiclass_nms_with_mask(
#                _mlvl_bboxes,
#                mlvl_scores,
#                _mlvl_masks,
#               cfg.score_thr,
#               cfg.nms,
#                cfg.max_per_img,
#                score_factors=mlvl_centerness + centerness_factor)
        #print(det_bboxes,'\n\n\n___________________',flush=True)
        return det_bboxes, det_labels #, det_masks


def dist2obb(dist_obb,pos_point):
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
    
    pts = np.array([[x-b_cos-l_sin, y-b_sin+l_cos, x+r_sin-b_cos, y-r_cos-b_sin, x+t_cos+ r_sin, y+t_sin-r_cos, x-l_sin + t_cos, y + l_cos + t_sin]])
    return torch.tensor(polygonToRotRectangle_batch(pts)).squeeze(0)


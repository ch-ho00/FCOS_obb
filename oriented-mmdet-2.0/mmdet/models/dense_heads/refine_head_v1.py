import numpy as np
import torch.nn as nn
import torch.functional
from mmcv.cnn import normal_init, ConvModule, bias_init_with_prob
from ..builder import HEADS, build_loss

from mmdet.core import force_fp32, multi_apply
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

@HEADS.register_module()
class RefineHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_strides=[4, 8, 16, 32, 64],
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 stacked_convs=4,
                 pos_th = 0.5,
                 neg_th = 0.5,
                 conv_cfg=None,
                 norm_cfg=None,                 
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(RefineHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        # self.anchor_scales = anchor_scales
        # self.anchor_ratios = anchor_ratios
        self.strides = anchor_strides
        # self.anchor_angle = anchor_angle
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']

        #if self.use_sigmoid_cls:
        #    self.cls_out_channels = num_classes - 1
        #else:
        self.cls_out_channels = num_classes

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        # self.anchor_generators = []
        # for anchor_base in self.anchor_strides:
        #     self.anchor_generators.append(
        #         AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios, self.anchor_angle))
        # self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales) * len(self.anchor_angle)
        # self.octave_base_scale = octave_base_scale
        # self.scales_per_octave = scales_per_octave
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.pos_th = pos_th 
        self.neg_th = neg_th
        self._init_layers()

    def _init_layers(self):
        #self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.refine_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.refine_reg = nn.Conv2d(
            self.feat_channels, 
            5, 
            3, 
            padding=1)
        self.feat_refine_1_5 = nn.Conv2d(
            chn,
            self.feat_channels,
            [1,5],
            stride=1,
            padding=(0,2))
        self.feat_refine_5_1 = nn.Conv2d(
                chn,
                self.feat_channels,
                [5,1],
                stride=1,
                padding=(2,0))
        self.feat_refine_1_1 = nn.Conv2d(
                chn,
                self.feat_channels,
                [1,1],
                stride=1,
                padding=0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.refine_cls, std=0.01, bias=bias_cls)
        normal_init(self.refine_reg, std=0.01)
        normal_init(self.feat_refine_1_5, std=0.01)
        normal_init(self.feat_refine_5_1, std=0.01)
        normal_init(self.feat_refine_1_1, std=0.01)

    def rbbox_transform(self,ex_rois, gt_rois, scale_factors=None):

        targets_dx = (gt_rois[:, 0] - ex_rois[:, 0]) / ex_rois[:, 2]
        targets_dy = (gt_rois[:, 1] - ex_rois[:, 1]) / ex_rois[:, 3]
        targets_dw = torch.log(gt_rois[:, 2] / ex_rois[:, 2])
        targets_dh = torch.log(gt_rois[:, 3] / ex_rois[:, 3])
    #    print("rbbox_trans:",gt_rois[:5],ex_rois[:5],'---------------\n',flush=True)
        targets_dtheta = (gt_rois[:, 4] - ex_rois[:, 4])

        if scale_factors:
            targets_dx *= scale_factors[0]
            targets_dy *= scale_factors[1]
            targets_dw *= scale_factors[2]
            targets_dh *= scale_factors[3]
            targets_dtheta *= scale_factors[4]

        targets = torch.cat((targets_dx.unsqueeze(1), targets_dy.unsqueeze(1), targets_dw.unsqueeze(1), targets_dh.unsqueeze(1), targets_dtheta.unsqueeze(1)),1)
        return targets

    def rbbox_transform_inv(self, boxes, deltas, scale_factors=None):

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

    def refine_target(self,
                      gt_bboxes,
                      gt_labels,
                      anchors,
                      pos_threshold,
                      neg_threshold):

        anchor_states = torch.zeros((anchors.shape[0],))
        labels = torch.zeros((anchors.shape[0],)).long().cuda()
        if gt_bboxes.shape[0]:
            # [N, M]
            overlaps = bbox_overlaps(anchors,gt_bboxes)
            print("overlap", torch.max(overlaps), anchors.shape, gt_bboxes.shape, overlaps.shape, flush=True)
            argmax_overlaps_inds = torch.argmax(overlaps, 1)
            print(argmax_overlaps_inds.shape,flush=True)
            max_overlaps = overlaps[torch.arange(overlaps.shape[0]), argmax_overlaps_inds]

            # compute box regression targets
            target_boxes = gt_bboxes[argmax_overlaps_inds]
            target_labels = gt_labels[argmax_overlaps_inds]

            positive_indices = max_overlaps >= pos_threshold
            print("????????????",torch.max(max_overlaps), flush=True)
            ignore_indices = (max_overlaps > neg_threshold) & ~positive_indices
            anchor_states[ignore_indices] = -1
            anchor_states[positive_indices] = 1

            # compute target class labels
            labels[positive_indices] = target_labels[positive_indices].long() - 1
        else:
            # no annotations? then everything is background
            target_boxes = torch.zeros((anchors.shape[0], gt_bboxes.shape[1]))

        target_delta = self.rbbox_transform(ex_rois=anchors, gt_rois=target_boxes)
        return labels, target_delta,anchor_states

    def refine_feature_op(self, points, feat_map):
        #num_imgs = 1 if feat_map.dim() == 3 else feat_map.shape[0] 
        h,w = feat_map.shape[-2:]
        h = torch.tensor(h).int()
        w = torch.tensor(w).int()
        
        xmin = points[:,0].min().clamp(min=0.0)
        ymin = points[:,1].min().clamp(min=0.0)
        xmax = points[:,0].max().clamp(max=w-1)
        ymax = points[:,1].max().clamp(max=h-1)

        left_top = torch.stack([xmin, ymin], 0).int()
        right_bottom = torch.stack([xmax, ymax], 0).int()
        left_bottom = torch.stack([xmin, ymax], 0).int()
        right_top = torch.stack([xmax, ymin], 0).int()

        feature_1_5 = self.feat_refine_1_5(feat_map)
        feature_5_1 = self.feat_refine_5_1(feature_1_5)
        feature_1_1 = self.feat_refine_1_1(feat_map)

        feature = feature_5_1 + feature_1_1
        left_top_feature = feature[:,:,left_top[0],left_top[1]].reshape(-1)
        right_bottom_feature = feature[:,:,right_top[0],right_top[1]].reshape(-1)
        left_bottom_feature = feature[:,:,left_bottom[0], left_bottom[1]].reshape(-1)
        right_top_feature = feature[:,:,right_top[0], right_top[1]].reshape(-1)
        
        refine_feature = right_bottom_feature * torch.abs((points[:,0]-xmin) * (points[:,1] - ymin)).reshape(-1,1).repeat(1,self.feat_channels) \
                        + left_top_feature * torch.abs((xmax- points[:,0]) *(ymax-points[:,1])).reshape(-1,1).repeat(1,self.feat_channels) \
                        + right_top_feature * torch.abs((points[:,0]-xmin) *(ymax-points[:,1])).reshape(-1,1).repeat(1,self.feat_channels) \
                        + left_bottom_feature * torch.abs((xmax- points[:,0]) *(points[:,1] - ymin)).reshape(-1,1).repeat(1,self.feat_channels)
        refine_feature = refine_feature.reshape(1,self.feat_channels,h,w) 

        return refine_feature + feature


    def feature_refine(self,
                       feature_py,
                       proposal,
                       box_pred,
                       cls_prob,
                       stride,
                       num_anchors,
                       proposal_filter=False):
        if proposal_filter:                
            box_pred = box_pred.reshape(-1, num_anchors, 5)
            proposal = proposal.reshape(-1, num_anchors, 5)
            cls_prob = cls_prob.reshape(-1, num_anchors, self.cls_out_channels)

            cls_max_prob, _ = torch.max(cls_prob, -1)
            box_pred_argmax = torch.max(cls_max_prob, -1)[1].reshape(-1, 1).long()
            indices = torch.ones_like(box_pred_argmax).cumsum(0).long() - torch.tensor(1).long()
            indices = torch.cat([indices, box_pred_argmax], -1)
            box_pred = torch.stack([box_pred[idx[0],idx[1]] for idx in indices]).reshape(-1,5)
            proposal = torch.stack([proposal[idx[0],idx[1]] for idx in indices]).reshape(-1,5)
            #box_pred = box_pred[indices.tolist()].reshape(-1,5)
            #proposal = proposal[indices.tolist()].reshape(-1,5)
        else:
            box_pred = box_pred.reshape(-1, 5)
            proposal = proposal.reshape(-1, 5)
        filtered_bboxes = self.rbbox_transform_inv(boxes=proposal,deltas=box_pred)
        center_point = filtered_bboxes[:,:2] / stride
        refine_feature_py = self.refine_feature_op(points=center_point, feat_map=feature_py)
        
        return refine_feature_py, filtered_bboxes

    def forward(self,
                feats,
                cls_scores,
                bbox_pred,
                anchor_list,
                img_metas,
                stage,
                num_anchors): 
        stage = [stage] * len(self.strides)
        num_anchors = [num_anchors] * len(self.strides)
        return multi_apply(self.forward_single,
                           feats,
                           cls_scores,
                           bbox_pred,
                           anchor_list,
                           img_metas,
                           self.strides,
                           stage,
                           num_anchors)

    def forward_single(self, 
                       x,
                       cls_scores,
                       bbox_preds,
                       anchor_list,
                       img_metas,
                       stride,
                       stage,
                       num_anchors):
        cls_scores = cls_scores.reshape(-1,self.cls_out_channels)
        bbox_preds = bbox_preds.reshape(-1,5)

        cls_prob = torch.sigmoid(cls_scores)
        #print("forward",x.shape,anchor_list.shape, bbox_preds.shape, cls_scores.shape, flush=True)
        x, refine_boxes = self.feature_refine(x, anchor_list, 
            bbox_preds, cls_scores, stride, num_anchors,
            proposal_filter=True if stage == 0 else False)

        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.refine_cls(cls_feat)
        bbox_pred = self.refine_reg(reg_feat)

        cls_prob = torch.sigmoid(cls_score)
        return refine_boxes, bbox_pred, cls_score

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))    
    def loss(self,
             cls_scores,
             bbox_preds,
             proposals,
             gt_bboxes,
             gt_labels,
             img_metas,
             train_cfg):        
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            proposals,
            gt_bboxes,
            gt_labels,
            cfg=train_cfg)
        # print('loss_single in ', str(datetime.timedelta(seconds=time.time() - t3)))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self,
                    cls_score, 
                    bbox_pred,
                    proposals, 
                    gt_bboxes,
                    gt_labels,
                    cfg=None):
        cls_score = cls_score.permute(0,2,3,
                                      1).reshape(-1,self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0,2,3,1).reshape(-1,5)
        cls_reg_targets = self.refine_target(
            gt_bboxes,
            gt_labels,
            proposals,
            self.pos_th, 
            self.neg_th)

        if cls_reg_targets is None:
            return None
        (labels, deltas, anchor_states) = cls_reg_targets
        inds = (anchor_states == 1).nonzero().squeeze(1)
        cls_score = cls_score[inds,:]
        bbox_pred = bbox_pred[inds,:]
        deltas = deltas[inds,:]
        labels = labels[inds] 
        # classification loss
        labels = labels.reshape(-1)
        num_total_samples = deltas.shape[0]
        print(cls_score.shape, bbox_pred.shape, deltas.shape, num_total_samples, flush=True)
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        if num_total_samples == 0:
            return None
        assert labels.shape[0] == cls_score.shape[0]
        loss_cls = self.loss_cls(
            cls_score, labels, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.reshape(-1, 5)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            deltas,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox



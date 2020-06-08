import numpy as np
import torch.nn as nn
import torch.functional
from mmcv.cnn import normal_init, ConvModule, bias_init_with_prob
from ..builder import HEADS, build_loss

from mmdet.core import (force_fp32, multi_apply, build_assigner,multiclass_nms,  
                        build_bbox_coder, build_sampler, anchor_inside_flags,
                        unmap, images_to_levels)
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

@HEADS.register_module()
class RefineHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_strides=[4, 8, 16, 32, 64],
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 background_label=None,
                 reg_decoded_bbox = False,
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
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):

        super(RefineHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.strides = anchor_strides

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.background_label = (
            num_classes if background_label is None else background_label)

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.cls_out_channels = num_classes

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

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

      
        target_delta = self.bbox_coder.encode(anchors,target_boxes)
        print(target_delta.shape, flush =True)
        return labels, target_delta,anchor_states

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 6
#        print('flages inside' ,inside_flags.shape,inside_flags.int().nonzero().shape, flush=True)
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        # TODO: fix gt_bboxes_ignore later
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, None,
            gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes,gt_labels)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
                #print("refine head target ",anchors.shape, sampling_result.pos_bboxes.shape, sampling_result.neg_bboxes.shape,sampling_result.pos_gt_bboxes.shape,flush=True) 
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        num_imgs = len(img_metas)
#        assert len(anchor_list) == len(valid_flag_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = results[:6]
        rest_results = list(results[6:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg) \
            + tuple(rest_results)

#    @force_fp32(apply_to=('feat_map'))
    def refine_feature_op(self, points, feat_map):
        num_imgs = 1 if feat_map.dim() == 3 else feat_map.shape[0] 
        
        h,w = feat_map.shape[-2:]
        h = torch.tensor(h).int()
        w = torch.tensor(w).int()
        
        xmin = torch.clamp(torch.floor(points[:,0]),min=0.0, max = w-1)
        ymin = torch.clamp(torch.floor(points[:,1]), min=0.0, max=h-1)
        xmax = torch.clamp(torch.ceil(points[:,0]), min=0.0,max=w-1)
        ymax = torch.clamp(torch.ceil(points[:,1]), min=0.0, max=h-1)

        left_top = torch.stack([xmin, ymin], 0).t().long()
        right_bottom = torch.stack([xmax, ymax], 0).t().long()
        left_bottom = torch.stack([xmin, ymax], 0).t().long()
        right_top = torch.stack([xmax, ymin], 0).t().long()


        feature_1_5 = self.feat_refine_1_5(feat_map)
        feature_5_1 = self.feat_refine_5_1(feature_1_5)
        feature_1_1 = self.feat_refine_1_1(feat_map)

        feature = feature_5_1 + feature_1_1
        anchor_per_img = h * w
        refine_feature = torch.zeros_like(feature)
        for i in range(num_imgs):
            start = anchor_per_img * i
            end = anchor_per_img * (i+1) 
            left_top_feature = feature[i,:,left_top[start:end,0],left_top[start:end,1]].contiguous()
            right_bottom_feature = feature[i,:,right_bottom[start:end,0],right_bottom[start:end,1]].contiguous()
            left_bottom_feature = feature[i,:,left_bottom[start:end,0], left_bottom[start:end,1]].contiguous()
            right_top_feature = feature[i,:,right_top[start:end,0], right_top[start:end,1]].contiguous()
 
            refine_feature = right_bottom_feature * torch.abs((points[start:end,0]-xmin[start:end]) * (points[start:end,1] - ymin[start:end])).reshape(-1,1).repeat(1,self.feat_channels).t() \
                            + left_top_feature * torch.abs((xmax[start:end]- points[start:end,0]) * (ymax[start:end]-points[start:end,1])).reshape(-1,1).repeat(1,self.feat_channels).t() \
                            + right_top_feature * torch.abs((points[start:end,0]-xmin[start:end]) * (ymax[start:end]-points[start:end,1])).reshape(-1,1).repeat(1,self.feat_channels).t() \
                            + left_bottom_feature * torch.abs((xmax[start:end] - points[start:end,0]) * (points[start:end,1] - ymin[start:end])).reshape(-1,1).repeat(1,self.feat_channels).t()
            refine_feature = refine_feature.reshape(self.feat_channels,h,w).to(feature.device)
            feature[i,...] = feature[i,...] + refine_feature
        return feature

#    @force_fp32(apply_to=('feature_py','cls_prob', 'box_pred'))
    def feature_refine(self,
                       feature_py,
                       proposal,
                       valid_flag,
                       box_pred,
                       cls_prob,
                       stride,
                       num_anchors,
                       proposal_filter=False):
        if proposal_filter:                
            box_pred = box_pred.reshape(-1, num_anchors, 5)
            proposal = proposal.reshape(-1, num_anchors, 5)
            valid_flag = valid_flag.reshape(-1, num_anchors)
            cls_prob = cls_prob.reshape(-1, num_anchors, self.cls_out_channels)
          
            cls_max_prob, _ = torch.max(cls_prob, -1)
            box_pred_argmax = torch.max(cls_max_prob, -1)[1].reshape(-1, 1).long()
            indices = torch.ones_like(box_pred_argmax).cumsum(0).long() - torch.tensor(1).long()
            indices = torch.cat([indices, box_pred_argmax], -1)

            box_pred = torch.stack([box_pred[idx[0],idx[1]] for idx in indices]).reshape(-1,5)
            proposal = torch.stack([proposal[idx[0],idx[1]] for idx in indices]).reshape(-1,5)
            valid_flag = torch.stack([valid_flag[idx[0], idx[1]] for idx in indices]).reshape(-1)

            #box_pred = box_pred[indices.tolist()].reshape(-1,5)
            #proposal = proposal[indices.tolist()].reshape(-1,5)
        else:
            box_pred = box_pred.reshape(-1, 5)
            proposal = proposal.reshape(-1, 5)
        filtered_bboxes = self.bbox_coder.decode(proposal, box_pred)

        center_point = filtered_bboxes[:,:2] / stride
        refine_feature_py = self.refine_feature_op(points=center_point, feat_map=feature_py)
        return refine_feature_py, filtered_bboxes, valid_flag

    def forward(self,
                feats,
                cls_scores,
                bbox_pred,
                anchor_list,
                valid_flag_list,
                stage,
                num_anchors): 
        stage = [stage] * len(self.strides)
        num_anchors = [num_anchors] * len(self.strides)

        concat_anchor_list = []
        concat_valid_list = []
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_list.append(torch.cat(valid_flag_list[i]))

        anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        valid_flag_list = images_to_levels(concat_valid_list,
                                           num_level_anchors)
       
        return multi_apply(self.forward_single,
                           feats,
                           cls_scores,
                           bbox_pred,
                           anchor_list,  
                           valid_flag_list,
                           self.strides,
                           stage,
                           num_anchors)
#    @force_fp32(apply_to=('x','cls_scores', 'bbox_preds'))
    def forward_single(self, 
                       x,
                       cls_scores,
                       bbox_preds,
                       anchor,
                       valid_flag,
                       stride,
                       stage,
                       num_anchors):
        cls_scores = cls_scores.permute(0,2,3,1).reshape(-1,self.cls_out_channels)
        bbox_preds = bbox_preds.permute(0,2,3,1).reshape(-1,5)

        cls_prob = torch.sigmoid(cls_scores)
        x, refine_boxes, valid_flags = self.feature_refine(x, anchor, valid_flag, 
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
        return refine_boxes, valid_flags, bbox_pred, cls_score

 #   @force_fp32(apply_to=('cls_scores', 'bbox_preds'))    
    def loss(self,
             cls_scores,
             bbox_preds,
             anchor_list,
             valid_flag_list,
             gt_bboxes,
             gt_labels,
             img_metas,
             train_cfg):       
#        anchor_list = [anchor_list]
#        valid_flag_list = [valid_flag_list]
        label_channels = self.cls_out_channels #if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=None, #gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    def loss_single(self, 
                    cls_score, 
                    bbox_pred, 
                    anchors, 
                    labels, 
                    label_weights,
                    bbox_targets, 
                    bbox_weights, 
                    num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   mlvl_anchors,
                   img_metas,
                   cfg=None,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.annotation)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
# from .anchor_head import AnchorHead
from .rot_AnchorHeadRbbox import rot_AnchorHeadRbbox
from mmdet.core.anchor import rot_AnchorGenerator
from ..builder import build_loss
import random
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from mmdet.core import multi_apply,RotBox2Polys,force_fp32, multi_apply, multiclass_nms, dbbox2delta,delta2dbbox,multiclass_nms_rbbox
from mmdet.core.utils import anchor_target_layer, refinebox_target_layer, rbbox_transform_inv,filter_detections
import torch

@HEADS.register_module
class R3Det_Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_channels=256,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64,128],
                #  anchor_base_sizes=None,
                 anchor_angles = [-90, -75, -60, -45, -30, -15],
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 num_refine=2,
                 level= ['P3', 'P4', 'P5', 'P6', 'P7'],
                 vis_score= 0.4,
                 filter_score= 0.05,
                 nms_iou_th= 0.1,
                 nms_pre= 1000,
                 max_per_img=100,
                 with_module=True,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),                 
                 train_cfg =None,
                 **kwargs):
        super(R3Det_Head, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = anchor_angles
        self.anchor_strides = anchor_strides
        self.level = level
        self.num_refine= num_refine
        self.nms_pre = nms_pre
        self.losses_dict = {}
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
 
        self.train_cfg = train_cfg
#        octave_scales = np.array(
#            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
#        anchor_scales = octave_scales * octave_base_scale

        self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        # to radians
        anchor_angles = [(a*np.pi/180)%(2*np.pi) for a in anchor_angles]
        self.anchor_generator = rot_AnchorGenerator(anchor_scales, anchor_ratios,anchor_angles)
        self.vis_score= vis_score 
        self.filter_score = filter_score
        self.nms=True
        self.nms_iou_th = nms_iou_th
        self.max_per_img = max_per_img
        self.num_anchors = len(self.anchor_angles)* len(self.anchor_ratios) * len(self.anchor_scales)
        self.with_module = with_module
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.refine_cls_net = nn.ModuleList()
        self.refine_reg_net = nn.ModuleList()
        
        self.feat_refine_1_5 = nn.ModuleList()
        self.feat_refine_5_1 = nn.ModuleList()
        self.feat_refine_1_1 = nn.ModuleList()
        self.refine_cls_fin = nn.ModuleList()
        self.refine_reg_fin = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    
                    ))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    
                    ))
        self.rpn_cls = ConvModule(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3, padding=1)

        self.rpn_reg = ConvModule(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)


        for i in range(self.num_refine):
            chn = self.feat_channels
            for j in range(4):
                self.refine_cls_net.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1
                    )
                )
    #            chn = self.num_anchors * 5 if i ==0 else self.feat_channels
                self.refine_reg_net.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1
                    )
                )
            chn = self.in_channels if i == 0 else self.feat_channels
            self.feat_refine_1_5.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    [1,5],
                    stride=1,
                    padding=(0,2)
                )
            )
            self.feat_refine_5_1.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    [5,1],
                    stride=1,
                    padding=(2,0)
                )
            )
            self.feat_refine_1_1.append(
                nn.Conv2d(
                    chn,
                    self.feat_channels,
                    [1,1],
                    stride=1,
                    padding=0
                )
            )

            self.refine_cls_fin.append(ConvModule(
                self.feat_channels, self.cls_out_channels, 3, padding =1))

            self.refine_reg_fin.append(ConvModule(
                self.feat_channels, 5, 3, padding =1))

            
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.refine_cls_net:
            normal_init(m.conv, std=0.01)
        for m in self.refine_reg_net:
            normal_init(m.conv, std=0.01)
        for m in self.feat_refine_1_5:
            normal_init(m, std=0.01)
        for m in self.feat_refine_5_1:
            normal_init(m, std=0.01)
        for m in self.feat_refine_1_1:
            normal_init(m, std=0.01)
        for m in self.refine_cls_fin:
            normal_init(m.conv, std=0.01)
        for m in self.refine_reg_fin:
            normal_init(m.conv, std=0.01)
        # FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.rpn_cls.conv, std=0.01)
        normal_init(self.rpn_reg.conv, std=0.01)
    '''    
    def make_anchors(self,feature_pyramid,img_metas):
        anchor_list = []
        level_list = self.level
            # cfg.base_anchor_size_list, cfgs.anchor_stride 
        for i,(level, base_anchor_size, stride) in enumerate(zip(level_list,self.train_cfg.base_size_list,self.anchor_strides)):
            feat_h, feat_w = feature_pyramid[i].shape[-2:]
            #feat_h = torch.Tensor(feat_h) ; feat_w = torch.Tensor(feat_w)
            tmp_anchors = self.anchor_generator.gen_base_anchors(base_anchor_size,feat_h, feat_w, stride)
            assert tmp_anchors.shape[-1] == 5
            anchor_list.append(tmp_anchors)
        return anchor_list
    '''

    def make_anchors(self,feature_pyramid,img_metas):
        anchor_list = []
        level_list = self.level
            # cfg.base_anchor_size_list, cfgs.anchor_stride 
        for i,(level, base_anchor_size, stride) in enumerate(zip(level_list,self.train_cfg.base_size_list,self.anchor_strides)):
            feat_h, feat_w = feature_pyramid[i].shape[-2:]
            #feat_h = torch.Tensor(feat_h) ; feat_w = torch.Tensor(feat_w)
            tmp_anchors = self.anchor_generator.gen_base_anchors(base_anchor_size,feat_h, feat_w, stride)
            assert tmp_anchors.shape[-1] == 5
            anchor_list.append(tmp_anchors)
        return anchor_list

    def get_bbox(self,refine_bbox_pred, cls_prob, anchors,is_training):
        # from libs.detection_oprations.refine_proposal_opr import postprocess_detctions
        boxes_pred = rbbox_transform_inv(boxes=anchors, deltas=refine_bbox_pred,scale_factors=None)
        nms_pre = self.nms_pre
    #    print('rbbox_trans_inv',boxes_pred.shape,flush=True)
        padding = cls_prob.new_zeros(cls_prob.shape[0], 1)
        cls_prob = torch.cat([padding, cls_prob], dim=1)
 
        if nms_pre >0 and boxes_pred.shape[0] > nms_pre:
            max_scores, _ = cls_prob.max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            boxes_pred = boxes_pred[topk_inds,:]
            cls_prob = cls_prob[topk_inds,:]
     #   print('before nms',boxes_pred.shape, cls_prob.shape,flush=True)
        det_boxes , det_labels = multiclass_nms_rbbox(boxes_pred, cls_prob, self.nms_iou_th, None, self.max_per_img)
        return det_boxes , det_labels
        '''
        return_boxes_pred = []
        return_scores = []
        return_labels = []
        for j in range(0, 15):
            indices = filter_detections(boxes_pred, refine_cls_prob[:, j], vis_score=self.vis_score, filter_score=self.filter_score, nms=self.nms,nms_iou_th=self.nms_iou_th,is_training=is_training)
            print('filtred indices',str(j)," ",indices.shape, flush=True)
            tmp_boxes_pred = boxes_pred[indices.tolist()].reshape(-1,5)
            # tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, indices), [-1, 5])
            tmp_scores = refine_cls_prob[:,j][indices.tolist()].reshape(-1)
            # tmp_scores = tf.reshape(tf.gather(refine_cls_prob[:, j], indices), [-1, ])
            print("get bboxes per class",j,":",tmp_boxes_pred.shape, tmp_scores.shape, flush=True)
            return_boxes_pred.append(tmp_boxes_pred)
            return_scores.append(tmp_scores)
            return_labels.append(torch.ones_like(tmp_scores)*(j+1))

        return_boxes_pred = torch.cat(return_boxes_pred, 0)
        return_scores = torch.cat(return_scores, 0)
        return_labels = torch.cat(return_labels, 0)

        return return_boxes_pred, return_scores, return_labels
        '''
    def rpn_net(self,x,gt_rbboxes=None, gt_hbboxes=None, anchor_list=None,gt_labels=None):
        
        rpn_bbox_pred_list = [] ; rpn_cls_score_list = [] ; rpn_bbox_cls_probs_list = []
        for i in range(len(self.level)):
            # cls branch
            cls_feat = x[i]
            reg_feat = x[i]

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            rpn_cls_score = self.rpn_cls(cls_feat)
            rpn_cls_score = rpn_cls_score.reshape(-1,self.cls_out_channels)
            rpn_bbox_cls_probs = F.sigmoid(rpn_cls_score) 

            # bbox branch
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            rpn_bbox_pred = self.rpn_reg(reg_feat)
            rpn_bbox_pred = rpn_bbox_pred.reshape(-1,5)
            
            rpn_bbox_pred_list.append(rpn_bbox_pred)
            rpn_cls_score_list.append(rpn_bbox_cls_probs)
            rpn_bbox_cls_probs_list.append(rpn_cls_score)

        rpn_bbox_pred = torch.cat(rpn_bbox_pred_list, 0)
        rpn_cls_score = torch.cat(rpn_cls_score_list, 0)
        #rpn_bbox_cls_probs = torch.cat(rpn_bbox_cls_probs_list,0)
        anchors = torch.cat(anchor_list, 0)
        if self.is_training:
            # anchor target generation
            #labels, target_delta, anchor_states, target_boxes = anchor_target_layer(gt_hbboxes.cpu().numpy(), gt_rbboxes.cpu().numpy(), anchors.cpu().numpy(),gt_labels.cpu().numpy(),self.train_cfg.assigner['pos_iou_thr'], self.train_cfg.assigner['neg_iou_thr']) 
            labels, target_delta, anchor_states, target_boxes = anchor_target_layer(gt_hbboxes, gt_rbboxes, anchors,gt_labels,self.train_cfg.assigner['pos_iou_thr'], self.train_cfg.assigner['neg_iou_thr']) 
            rpn_cls_score = rpn_cls_score[(anchor_states != -1).nonzero(),:].squeeze(1)
            rpn_bbox_pred = rpn_bbox_pred[(anchor_states != -1).nonzero(),:].squeeze(1)
            target_delta = target_delta[(anchor_states != -1).nonzero(),:].squeeze(1)

            self.losses_dict['cls_loss'] = self.loss_cls(rpn_cls_score,labels)
            # if smmooth_l1_loss
            self.losses_dict['reg_loss'] = self.loss_bbox(rpn_bbox_pred,target_delta)
            # if iou_smooth_l1_loss
            # reg_loss = losses.iou_smooth_l1_loss(target_delta, rpn_box_pred, anchor_states, target_boxes, anchors)
        return rpn_cls_score_list, rpn_bbox_pred_list, rpn_bbox_cls_probs_list

    def refine_net(self,x,stage=None):
        refine_feature_py = {}
        refine_bbox_pred_list = [] ; refine_cls_score_list = [] ; refine_bbox_cls_probs_list = []
        for i in range(len(self.level)):
            cls_feat = x[i]
            reg_feat = x[i]

            for j in range(4):
                cls_feat = self.refine_cls_net[stage*4+j](cls_feat)
            refine_cls_score = self.refine_cls_fin[stage](cls_feat)
            refine_cls_score = refine_cls_score.reshape(-1,self.cls_out_channels)
            refine_bbox_cls_probs = F.sigmoid(refine_cls_score) 

            for j in range(4):
                reg_feat = self.refine_reg_net[stage*4+j](reg_feat)
            refine_bbox_pred = self.refine_reg_fin[stage](reg_feat)
            refine_bbox_pred = refine_bbox_pred.reshape(-1,5)
            
            refine_bbox_pred_list.append(refine_bbox_pred)
            refine_cls_score_list.append(refine_bbox_cls_probs)
            refine_bbox_cls_probs_list.append(refine_cls_score)
                
        return refine_cls_score_list, refine_bbox_pred_list, refine_bbox_cls_probs_list

    def refine_feature_op(self,points,feat_map,stage=None):
        h,w = feat_map.shape[-2:]
        h = torch.tensor(h).int(); w = torch.tensor(w).int()
        xmin = points[:,0].min().clamp(min=0.0)
        ymin = points[:,1].min().clamp(min=0.0)
        xmax = points[:,0].max().clamp(max=w-1)
        ymax = points[:,1].max().clamp(max=h-1)

        left_top = torch.stack([xmin, ymin], 0).int()
        right_bottom = torch.stack([xmax, ymax], 0).int()
        left_bottom = torch.stack([xmin, ymax], 0).int()
        right_top = torch.stack([xmax, ymin], 0).int()
        feature_1_5 = self.feat_refine_1_5[stage](feat_map)
        feature_5_1 = self.feat_refine_5_1[stage](feature_1_5)
        feature_1_1 = self.feat_refine_1_1[stage](feat_map)

        feature = feature_5_1 + feature_1_1
        
        # squeeze which dim ? 
        left_top_feature = feature[:,:,left_top[0],left_top[1]]
        right_bottom_feature = feature[:,:,right_top[0],right_top[1]]
        left_bottom_feature = feature[:,:,left_bottom[0], left_bottom[1]]
        right_top_feature = feature[:,:,right_top[0], right_top[1]]

        refine_feature = right_bottom_feature * torch.abs((points[:,0]-xmin)*(points[:,1] - ymin)).reshape(-1,1).repeat(1,self.feat_channels) \
                        + left_top_feature * torch.abs((xmax- points[:,0]) *(ymax-points[:,1])).reshape(-1,1).repeat(1,self.feat_channels) \
                        + right_top_feature * torch.abs((points[:,0]-xmin) *(ymax-points[:,1])).reshape(-1,1).repeat(1,self.feat_channels) \
                        + left_bottom_feature * torch.abs((xmax- points[:,0]) *(points[:,1] - ymin)).reshape(-1,1).repeat(1,self.feat_channels)
        refine_feature = refine_feature.reshape(1,self.feat_channels,h,w) 

        return refine_feature + feature
    def refine_stage(self,feature_py, gt_rbboxes, gt_labels, box_pred_list, cls_prob_list, proposal_list, pos_th, neg_th,proposal_filter=False,stage=None):
        refine_boxes_list = []
        refine_feature_py = {}
        for i,(box_pred, cls_prob, proposal, stride, level) in \
                enumerate(zip(box_pred_list, cls_prob_list, proposal_list, self.anchor_strides, self.level)):
            if proposal_filter:                
                box_pred = box_pred.reshape(-1, self.num_anchors, 5)
                proposal = proposal.reshape(-1, self.num_anchors, 5 )
                cls_prob = cls_prob.reshape(-1, self.num_anchors, self.cls_out_channels)

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
            bboxes = rbbox_transform_inv(boxes=proposal,deltas=box_pred)
            refine_boxes_list.append(bboxes)
            center_point = bboxes[:,:2] / stride
            refine_feature_py[i] = self.refine_feature_op(points=center_point, feat_map=feature_py[i],stage=stage)

        refine_cls_score_list, refine_bbox_pred_list, refine_cls_prob_list = self.refine_net(refine_feature_py,stage)
        #print("refine_net ouput",[r.shape for r in refine_cls_score_list],[ r.shape for r in refine_bbox_pred_list], [r.shape for r in refine_cls_prob_list],flush=True)
        refine_bbox_pred = torch.cat(refine_bbox_pred_list, 0)
        refine_cls_score = torch.cat(refine_cls_score_list, 0)
        refine_boxes = torch.cat(refine_boxes_list, 0)

        #loss
        if self.is_training:
            #refine_labels, refine_target_delta, refine_box_states, refine_target_boxes = refinebox_target_layer(gt_rbboxes.detach().cpu().numpy(), refine_boxes.detach().cpu().numpy(), gt_labels.detach().cpu().numpy(), pos_th, neg_th)
            refine_labels, refine_target_delta, refine_box_states, refine_target_boxes = refinebox_target_layer(gt_rbboxes, refine_boxes, gt_labels, pos_th, neg_th)

            refine_cls_score = refine_cls_score[(refine_box_states != -1).nonzero(),:].squeeze(1)
            refine_bbox_pred = refine_bbox_pred[(refine_box_states != -1).nonzero(),:].squeeze(1)
            refine_target_delta = refine_target_delta[(refine_box_states != -1).nonzero(),:].squeeze(1)                 
            self.losses_dict['refine_cls_loss_%d'%(stage)] = self.loss_cls(refine_cls_score,refine_labels) 
            # smooth l1 loss
            self.losses_dict['refine_reg_loss_%d'%(stage)] = self.loss_bbox(refine_bbox_pred,refine_target_delta)

        return refine_boxes_list, refine_bbox_pred_list, refine_cls_prob_list
    '''
    def forward(self,x, img_metas, gt_rbboxes, gt_hbboxes, gt_labels,training,train_cfg):

        self.train_cfg = train_cfg
        self.is_training = training

        return multi_apply(self.forward_single,x, img_metas, gt_rbboxes, gt_hbboxes, gt_labels)
    ''' 
    def forward(self,x, img_metas,
                        gt_rbboxes,
                        gt_hbboxes, 
                        gt_labels,
                        training=True,
                        train_cfg=None):
        self.train_cfg = train_cfg
        self.is_training = training
#        print("start",flush=True)
        gt_rbboxes = gt_rbboxes[0]
        gt_hbboxes = gt_hbboxes[0]
        gt_labels = gt_labels[0]
        # rpn net
        anchor_list = self.make_anchors(x, img_metas)
        rpn_cls_score_list, rpn_bbox_pred_list, rpn_cls_prob_list = self.rpn_net(x,gt_rbboxes,gt_hbboxes, anchor_list, gt_labels)
#        print("after rpn net",flush=True)

        bbox_pred_list, cls_prob_list, proposal_list = rpn_bbox_pred_list, rpn_cls_prob_list, anchor_list
        # refine net
        for i in range(self.num_refine):
            proposal_list, bbox_pred_list, cls_prob_list = self.refine_stage(x,
                                                                             gt_rbboxes,
                                                                             gt_labels,
                                                                             bbox_pred_list,
                                                                             cls_prob_list,
                                                                             proposal_list,
                                                                             pos_th=self.train_cfg.refine_iou_p_th[i],
                                                                             neg_th=self.train_cfg.refine_iou_n_th[i],
                                                                             proposal_filter=True if i == 0 else False,
                                                                             stage=i)
#            print("after refine net ",str(i),flush=True)

            if not self.is_training:
                all_box_pred_list.extend(bbox_pred_list)
                all_cls_prob_list.extend(cls_prob_list)
                all_proposal_list.extend(proposal_list)
            else:
                all_box_pred_list, all_cls_prob_list, all_proposal_list = bbox_pred_list, cls_prob_list, proposal_list
#            print('refine stage',str(i+1),flush=True)
        # post-process
        # stack or cat? 
        box_pred = torch.cat(all_box_pred_list,0)
        cls_prob = torch.cat(all_cls_prob_list,0)
        proposal = torch.cat(all_proposal_list,0)
#        print('done with refine net', flush=True)
        boxes, labels = self.get_bbox(box_pred,cls_prob, proposal,training)
#        print('post process', flush=True)
        if random.random() < 0.01:
            print('Sample output : GT\n',RotBox2Polys(gt_rbboxes[:5].detach().cpu()),'\n\npred',boxes[:5],'\n----------\n\n',flush=True)
        if self.is_training:
            return boxes, labels, self.losses_dict
        else:
            return boxes, labels



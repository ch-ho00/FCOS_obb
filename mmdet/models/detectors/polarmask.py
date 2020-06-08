from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import bbox2result, bbox_mask2result, RotBox2Polys,dbbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch

#remove later
import numpy as np
import matplotlib.pyplot as plt 

import random
import string

def randomString(stringLength=8):

    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(stringLength))


@DETECTORS.register_module
class PolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)


    def rand_visual(self,img, det_bboxes):
        print("random result generated",flush=True)
        plt.imshow(img)
        for i in range(det_bboxes.shape[0]):
            plt.scatter(det_bboxes[i,:-1][::2],det_bboxes[i,:-1][1::2])
        plt.savefig('./inference_tests/'+randomString(8)+'.png')
        plt.close()
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_hbboxes,
                      gt_labels,
#                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None
#                      _gt_masks=None
                      ):
        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_bboxes=_gt_bboxes)
#                              _gt_masks=_gt_masks)
        else:
            extra_data = None

        enter = 0
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if sum([torch.isnan(outs[0][i]).int().sum() for i in range(len(outs[0]))]) != 0:
            print("Nan in cls scores", flush=True)
            enter = 1
        if sum([torch.isnan(outs[1][i]).int().sum() for i in range(len(outs[1]))]) !=0:
            print('Nan in bbox pred', flush=True)
            enter =1
        if sum([torch.isnan(outs[2][i]).int().sum() for i in range(len(outs[2]))]) != 0:
            print('Nan in centerness pred', flush=True)
            enter = 1 
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
#            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            extra_data=extra_data,
        )

        if torch.isnan(losses['loss_cls']).int().sum() !=0:
            print('loss cls nan!!',flush=True)
            enter = 1
        if torch.isnan(losses['loss_centerness']).int().sum() !=0:
            print('loss center nan!!',flush=True)
            enter =1
        if torch.isnan(losses['loss_bbox']).int().sum() !=0:
            print('loss bbox nan!!',flush=True)
            print(losses['loss_bbox'],"\n\n_______________\n",torch.isnan(losses['loss_bbox']).int().nonzero(),'\n\n_________\n',outs[1][0].shape,'\n\n_________________________\n', gt_bboxes,flush=True)
            enter =1 
        
        return losses


    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*bbox_inputs)
#        if np.random.uniform(0,1) < 0.1:
#            self.rand_visual(img.detach().squeeze(0).permute(1,2,0).cpu().numpy(), det_bboxes.detach().cpu().numpy())
        rbbox_results =  dbbox2result(det_bboxes, det_labels ,self.bbox_head.num_classes)
        return rbbox_results               

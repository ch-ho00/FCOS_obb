#from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import torch

@DETECTORS.register_module
class R3Det(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 refine_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(R3Det, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        for module_ in refine_head:
            module_.update(train_cfg=train_cfg)
            module_.update(test_cfg=test_cfg)

        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.refine_head = build_head(refine_head) 
        self.init_weights(pretrained=pretrained)

    def init_weights(self,pretrained=None):
        super(R3Det,self).init_weights(pretrained)
        for refine_mod in self.refine_head:
            refine_mod.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # img_name = img_metas[0]['filename'].split('/')[-1]
        # mean = img_metas[0]['img_norm_cfg']['mean']
        # std = img_metas[0]['img_norm_cfg']['std']
        # img_array = (img.cpu().numpy()[0].transpose((1, 2, 0)) * np.array(std) + np.array(mean))
        num_img = len(img_metas)
        x = self.extract_feat(img)
        outs = self.bbox_head(x) #cls_score, bbox_pred
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        featmap_sizes = [featmap.size()[-2:] for featmap in outs[0]]
        proposal_list , valid_flag_list  = self.bbox_head.get_anchors(featmap_sizes, img_metas)
        num_anchor = self.bbox_head.num_anchors

        for stage, refine_module in enumerate(self.refine_head):
            refine_proposal_list, refine_valid_flag_list , refine_box_pred_list, \
                refine_cls_score_list  = refine_module(x,*outs,proposal_list, valid_flag_list,stage, num_anchor)
            proposal_list = []
            valid_flag_list = []

            for i in range(num_img):
                proposal_list.append([lvl[i*lvl.shape[0]//num_img: \
                    (i+1) * lvl.shape[0]//num_img] for lvl in refine_proposal_list])
                valid_flag_list.append([lvl[i*lvl.shape[0]//num_img : \
                    (i+1) * lvl.shape[0]//num_img].squeeze(0) for lvl in refine_valid_flag_list])
            outs = (refine_cls_score_list, refine_box_pred_list)
            loss_inputs =  outs + (proposal_list, valid_flag_list,gt_bboxes, gt_labels, img_metas, self.train_cfg) 
            tmp_loss = refine_module.loss(*loss_inputs)
            for type_ in tmp_loss:
                losses[type_+"_refine_"+str(stage)] = tmp_loss[type_]
        #print("LOSS :",losses,flush=True)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        featmap_sizes = [featmap.size()[-2:] for featmap in outs[0]]
        proposal_list , valid_flag_list  = self.bbox_head.get_anchors(featmap_sizes, img_metas)
        for stage, refine_module in enumerate(self.refine_head):
            proposal_list, valid_flag_list , refine_box_pred_list, \
                refine_cls_score_list  = refine_module(x,*outs,proposal_list, valid_flag_list,stage, num_anchor)
            outs = (refine_cls_score_list, refine_box_pred_list)

        bbox_inputs = outs + ([proposal_list],img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]


    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

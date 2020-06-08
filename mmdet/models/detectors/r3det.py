from .single_stage_rbbox import SingleStageDetectorRbbox
from ..registry import DETECTORS
from mmdet.core.bbox import dbbox2result

@DETECTORS.register_module
class R3Det(SingleStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(R3Det, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_hbboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None
                      ):
        x = self.backbone(img)
        x = self.neck(x)
        box_pred, labels,loss_dict = self.rbbox_head(x,img_metas,gt_bboxes,gt_hbboxes,gt_labels,True,train_cfg=self.train_cfg)
        return loss_dict
    
    def simple_test(self,img,img_meta,rescale=False):
        x = self.backbone(img)
        x = self.neck(x)
        box_pred, cls_prob, labels = self.rbbox_head(x,img_metas,None,None,False,test_cfg=self.test_cfg)
        bbox_inputs = (box_pred, cls_prob, labels) # + ??
        
        box_pred, labels = self.rbbox_head(*bbox_inputs)

        rbbox_results= dbbox2result(box_pred, labels,self.rbbox_head.num_classes)

        return rbbox_results


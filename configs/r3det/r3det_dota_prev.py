# model settings
# retinanet_obb_r50_fpn_2x.py 
model = dict(
    type='R3Det',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    rbbox_head=dict(
        type='R3Det_Head',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_scales = [2 ** 0,  2 ** (2.0 / 3.0)], # originally 2**(1.0/3.0) included 
        anchor_ratios=[1, 1 / 2, 2., 1 / 3., 3.], # orignally 5. ,1/5. also included
        # anchor_strides=[8, 16, 32, 64, 128],
        anchor_angles = [-90,  -60,  -30], # originally range(-90,0,15) 
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        num_refine=2,
        with_module=False,
        vis_score= 0.4,
        filter_score= 0.05,
        nms_iou_th= 0.1,
        nms_pre = 1000, 
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)
    ))
# faster_rcnn_RoITrans_r50_fpn_1x_dota.py
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssignerRbbox',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    refine_iou_p_th= [0.5, 0.6],
    refine_iou_n_th= [0.4, 0.5],
    base_size_list= [32, 64, 128, 256, 512],
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr = 0.05, 
    nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), 
    max_per_img = 2000),

dataset_type = 'DOTADataset'
data_root = 'data/dota1_1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA_test1024.json',
        img_prefix=data_root + 'test1024/images',
        # ann_file=data_root + 'test1024_ms/DOTA_test1024_ms.json',
        # img_prefix=data_root + 'test1024_ms/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
# R3det
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/r3det'
load_from = None
resume_from = None
workflow = [('train', 1)]


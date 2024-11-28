_base_ = [
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py',
    '../_base_/datasets/whu-opt-sar-RGBS_512x512.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='RGBXDataPreProcessor',
    size=crop_size,
    mean=[80.475, 98.217, 106.364, 53.837, 53.837, 53.837],
    std=[29.48, 26.953, 24.829, 48.475, 48.475, 48.475],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

checkpoint_file = '/home/pengbc/project/multi_seg/pretrain/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth'
model = dict(
    type='EarlyFusionSegmentor',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='VFuseNetBackbone',
                  backbone=dict(type='VGGBackbone',
                                depth=16,
                                num_stages=5,
                                out_indices=(0, 1, 2, 3, 4),
                                norm_cfg=norm_cfg,
                                init_cfg=dict(type='Pretrained',
                                              checkpoint=checkpoint_file,
                                              prefix='backbone.')),
                  stage_channels=[64, 128, 256, 512, 512]),
    decode_head=dict(type='FCNHead',
                     in_channels=64,
                     in_index=4,
                     channels=16,
                     num_convs=1,
                     concat_input=False,
                     num_classes=7,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     dropout_ratio=-1,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      avg_non_ignore=True,
                                      loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=64,
                        in_index=3,
                        channels=16,
                        num_convs=1,
                        concat_input=False,
                        num_classes=7,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        dropout_ratio=-1,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         avg_non_ignore=True,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(
    checkpoint=dict(save_best='mIoU', save_last=False, max_keep_ckpts=1))

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='WHURGBSFusion',
             name='vfusenet_vgg16',
         ))
]
visualizer = dict(vis_backends=vis_backends)

find_unused_parameters = True

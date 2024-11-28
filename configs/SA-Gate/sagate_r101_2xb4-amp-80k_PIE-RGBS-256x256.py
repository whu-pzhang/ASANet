_base_ = [
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py',
    '../_base_/datasets/pie-rgb-sar_256x256.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-5, momentum=0.1)
crop_size = (256, 256)
data_preprocessor = dict(
    type='RGBXDataPreProcessor',
    size=crop_size,
    # RGB,SAR
    mean=[73.422, 89.751, 83.405, 84.869, 84.869, 84.869],
    std=[29.48, 26.953, 24.829, 57.769, 57.769, 57.769],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EarlyFusionSegmentor',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='DualResNet',
                  depth=101,
                  deep_stem=True,
                  stem_width=64,
                  norm_cfg=norm_cfg,
                  init_cfg=dict(type='Pretrain',
                                checkpoint='/data1/torch/hub/checkpoints/resnet101_v1c-e67eebb6.pth')),
    decode_head=dict(
        type='SAGateHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         avg_non_ignore=True,
                         loss_weight=1.0),
    ),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=2048,
                        in_index=3,
                        channels=512,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=6,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         avg_non_ignore=True,
                                         loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(checkpoint=dict(save_best='mIoU', max_keep_ckpts=1))

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='PIERGBSFusion',
             name='sagate_r101',
         ))
]
visualizer = dict(vis_backends=vis_backends)

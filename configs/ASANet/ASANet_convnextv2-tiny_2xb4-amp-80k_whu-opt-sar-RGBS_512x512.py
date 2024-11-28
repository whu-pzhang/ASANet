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
class_weight = [0.9183, 1.0143, 1.0057, 0.9574, 0.9175, 1.1098, 1.0770]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-d8579f84.pth'  # noqa
model = dict(
    type='EarlyFusionSegmentor',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ASANet',
        backbone=dict(
            type='mmpretrain.ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=0.,  # disable layer scale when using GRN
            gap_before_final_norm=False,
            use_grn=True,  # V2 uses GRN
            init_cfg=dict(type='Pretrained',
                          checkpoint=checkpoint_file,
                          prefix='backbone.')),
        frm_cfg=dict(type='SFM'),
        ffm_cfg=dict(type='CFM'),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss',
                 use_sigmoid=False,
                 avg_non_ignore=True,
                 loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         avg_non_ignore=True,
                         loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(_delete_=True,
                     type='AmpOptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.0001,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.05),
                     paramwise_cfg={
                         'decay_rate': 0.9,
                         'decay_type': 'stage_wise',
                         'num_layers': 6
                     },
                     constructor='CustomLearningRateDecayOptimizerConstructor',
                     loss_scale='dynamic')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,
         end=1500),
    dict(type='PolyLR',
         power=1.0,
         begin=1500,
         end=80000,
         eta_min=0.0,
         by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(save_best='mIoU', save_last=False, max_keep_ckpts=1))

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='WHURGBSFusion',
             name='upernet_convnextv2-tiny-SFM-CFM',
         ))
]
visualizer = dict(vis_backends=vis_backends)

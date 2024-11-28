_base_ = [
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py',
    '../_base_/datasets/ddrh-rgb-sar_256x256.py',
]

crop_size = (256, 256)
data_preprocessor = dict(
    type='RGBXDataPreProcessor',
    size=crop_size,
    # RGB,SAR
    mean=[80.475, 98.217, 106.364, 53.837, 53.837, 53.837],
    std=[29.48, 26.953, 24.829, 48.475, 48.475, 48.475],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    _scope_='mmseg',
    type='EarlyFusionSegmentor',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='RGBXTransformer',
                  embed_dims=[64, 128, 320, 512],
                  num_heads=[1, 2, 5, 8],
                  mlp_ratios=[4, 4, 4, 4],
                  qkv_bias=True,
                  norm_cfg=dict(type='LN', eps=1e-6),
                  depths=[3, 4, 6, 3],
                  sr_ratios=[8, 4, 2, 1],
                  drop_rate=0.0,
                  drop_path_rate=0.1,
                  init_cfg=dict(
                      type='Pretrain',
                      checkpoint='/data1/torch/hub/checkpoints/mit_b2.pth')),
    decode_head=dict(
        type='MLPDecoderHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', eps=1e-3),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         avg_non_ignore=True,
                         loss_weight=1.0),
    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)),
)

optim_wrapper = dict(_delete_=True,
                     type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.00006,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.01),
                     paramwise_cfg=dict(
                         norm_decay_mult=0.,
                         bias_decay_mult=0.,
                     ))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,
         end=1500),
    dict(type='PolyLR',
         eta_min=0.0,
         power=1.0,
         begin=1000,
         end=80000,
         by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(save_best='mIoU', save_last=False, max_keep_ckpts=1))

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='DDRHFusion',
             name='cmx_mit-b2',
         ))
]
visualizer = dict(vis_backends=vis_backends)

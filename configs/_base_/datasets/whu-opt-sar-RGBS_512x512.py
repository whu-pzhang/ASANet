dataset_type = 'WHUOptSarDataset'
data_root = 'data/whu-opt-sar/crop512'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadMultipleImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MultiRandomResize',
         scale=(2048, 512),
         ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='MultiRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='MultiRandomFlip', direction='vertical', prob=0.5),
    dict(type='mmseg.ConcatCDInput'),
    dict(type='mmseg.PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultipleImageFromFile'),
    dict(type='MultiResize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='mmseg.ConcatCDInput'),
    dict(type='mmseg.PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadMultipleImageFromFile'),
    dict(type='TestTimeAug',
         transforms=[[
             dict(type='MultiResize', scale_factor=r, keep_ratio=True)
             for r in img_ratios
         ],
                     [
                         dict(type='MultiRandomFlip',
                              prob=0.,
                              direction='horizontal'),
                         dict(type='MultiRandomFlip',
                              prob=1.,
                              direction='horizontal')
                     ], [dict(type='LoadAnnotations')],
                     [dict(type='mmseg.ConcatCDInput')],
                     [dict(type='mmseg.PackSegInputs')]])
]

train_dataloader = dict(batch_size=4,
                        num_workers=4,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     data_prefix=dict(
                                         img_path='train/opt_dir',
                                         img_path2='train/sar_dir',
                                         seg_map_path='train/ann_dir'),
                                     pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   data_prefix=dict(
                                       img_path='test/opt_dir',
                                       img_path2='test/sar_dir',
                                       seg_map_path='test/ann_dir'),
                                   pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='CustomIoUMetric',
                     iou_metrics=['mIoU', 'mFscore', 'Kappa'])
test_evaluator = val_evaluator

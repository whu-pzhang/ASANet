dataset_type = 'DDRHHGOptDataset'
data_root = 'data/DDHR-DATA/korea'
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize',
         scale=(1024, 256),
         ratio_range=(1, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='mmseg.PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TestTimeAug',
         transforms=[[
             dict(type='Resize', scale_factor=r, keep_ratio=True)
             for r in img_ratios
         ],
                     [
                         dict(type='RandomFlip',
                              prob=0.,
                              direction='horizontal'),
                         dict(type='RandomFlip',
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
                                     ann_file='train.txt',
                                     data_prefix=dict(img_path='GF2',
                                                      seg_map_path='label'),
                                     pipeline=train_pipeline))

val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='val.txt',
                                   data_prefix=dict(img_path='GF2',
                                                    seg_map_path='label'),
                                   pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='CustomIoUMetric',
                     iou_metrics=['mIoU', 'mFscore', 'Kappa'])
test_evaluator = val_evaluator

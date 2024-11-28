from src import *

from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.utils import register_all_modules

import matplotlib.pyplot as plt

register_all_modules()

# dataset_type = 'NYUv2Dataset'
# data_root = 'data/NYUDepthv2'

dataset_type = 'WHUOptSarDataset'
data_root = 'data/whu-opt-sar/crop512'

albu_train_transforms = [
    # dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]

train_pipeline = [
    dict(type='LoadMultipleImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MultiRandomResize',
         scale=(2048, 640),
         ratio_range=(0.5, 1.75),
         keep_ratio=True),
    dict(type='MultiRandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='MultiRandomFlip', prob=0.5),
    dict(type='MultiRandomFlip', direction='vertical', prob=0.5),
    # dict(type='MultiResize', scale=(2048, 640), keep_ratio=True),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

# dataset = dict(type=dataset_type,
#                data_root=data_root,
#                ann_file='train.txt',
#                data_prefix=dict(img_path='RGB',
#                                 img_path2='Depth_jpg',
#                                 seg_map_path='Label'),
#                pipeline=train_pipeline)
dataset = dict(type=dataset_type,
               data_root=data_root,
               data_prefix=dict(img_path='train/opt_dir',
                                img_path2='train/sar_dir',
                                seg_map_path='train/ann_dir'),
               pipeline=train_pipeline)

ds = DATASETS.build(dataset)

sample = ds[100]

inputs = sample['inputs'].numpy().transpose(1, 2, 0)
data_samples = sample['data_samples']

img = inputs[..., :3][..., ::-1]
img2 = inputs[..., 3:]
gt = data_samples.gt_sem_seg.data.numpy().squeeze()

f, ax = plt.subplots(1, 3)

ax[0].imshow(img)
ax[1].imshow(img2)
ax[2].imshow(gt)

plt.tight_layout()
# plt.show()
plt.savefig('junk.jpg')

from typing import Callable, List, Optional, Sequence, Union
import os.path as osp
from pathlib import Path

import mmengine
import mmengine.fileio as fileio
from mmseg.datasets import BaseCDDataset, BaseSegDataset
from mmseg.registry import DATASETS


class BaseMultiModalDataset(BaseCDDataset):

    def load_data_list(self) -> List[dict]:
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        img_dir2 = self.data_prefix.get('img_path2', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(self.ann_file,
                                            backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(img_path=osp.join(img_dir,
                                                   img_name + self.img_suffix),
                                 img_path2=osp.join(
                                     img_dir2, img_name + self.img_suffix2))

                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(dir_path=img_dir,
                                               list_dir=False,
                                               suffix=self.img_suffix,
                                               recursive=True,
                                               backend_args=self.backend_args):

                data_info = dict(img_path=osp.join(img_dir, img),
                                 img_path2=osp.join(
                                     img_dir2,
                                     img[:-_suffix_len] + self.img_suffix2))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list


@DATASETS.register_module()
class PIEOptDataset(BaseSegDataset):
    METAINFO = dict(classes=('backbroad', 'city', 'road', 'water', 'forest',
                             'farmland'),
                    palette=[[128, 128, 128], [235, 51, 36], [255, 253, 85],
                             [66, 236, 245], [56, 168, 0], [168, 112, 0]])

    def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=False,
                         **kwargs)


@DATASETS.register_module()
class PIEOptSarDataset(BaseMultiModalDataset):
    METAINFO = dict(classes=('backbroad', 'city', 'road', 'water', 'forest',
                             'farmland'),
                    palette=[[128, 128, 128], [235, 51, 36], [255, 253, 85],
                             [66, 236, 245], [56, 168, 0], [168, 112,
                                                            0]])  # rgb

    def __init__(self,
                 img_suffix='.tif',
                 img_suffix2='.tif',
                 seg_map_suffix='.tif',
                 **kwargs):
        super().__init__(img_suffix=img_suffix,
                         img_suffix2=img_suffix2,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=False,
                         **kwargs)


@DATASETS.register_module()
class DDRHHGOptDataset(BaseSegDataset):
    METAINFO = dict(classes=('buildings', 'roads', 'greenery', 'water',
                             'farmland'),
                    palette=[[255, 0, 0], [0, 0, 255], [255, 255, 0],
                             [255, 0, 255], [0, 255, 0]])

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=False,
                         **kwargs)


@DATASETS.register_module()
class DDRHHGOptSarDataset(BaseMultiModalDataset):
    METAINFO = dict(classes=('buildings', 'roads', 'greenery', 'water',
                             'farmland'),
                    palette=[[255, 0, 0], [0, 0, 255], [255, 255, 0],
                             [255, 0, 255], [0, 255, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 img_suffix2='.jpg',
                 seg_map_suffix='.png',
                 **kwargs):
        super().__init__(img_suffix=img_suffix,
                         img_suffix2=img_suffix2,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=False,
                         **kwargs)


@DATASETS.register_module()
class WHUOptSarDataset(BaseMultiModalDataset):
    METAINFO = dict(classes=('farmland', 'city', 'village', 'water', 'forest',
                             'road', 'other'),
                    palette=[[255, 195, 128], [255, 0, 0], [159, 129, 183],
                             [0, 0, 255], [0, 255, 0], [255, 255, 0],
                             [255, 255, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 img_suffix2='.tif',
                 seg_map_suffix='.tif',
                 **kwargs):
        super().__init__(img_suffix=img_suffix,
                         img_suffix2=img_suffix2,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=True,
                         **kwargs)


@DATASETS.register_module()
class WHUOptSarRGBDataset(BaseSegDataset):
    METAINFO = dict(classes=('farmland', 'city', 'village', 'water', 'forest',
                             'road', 'other'),
                    palette=[[255, 195, 128], [255, 0, 0], [159, 129, 183],
                             [0, 0, 255], [0, 255, 0], [255, 255, 0],
                             [255, 255, 255]])

    def __init__(self, img_suffix='.tif', seg_map_suffix='.tif', **kwargs):
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         reduce_zero_label=True,
                         **kwargs)

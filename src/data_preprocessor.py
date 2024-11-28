from typing import Any, Dict

import torch

from mmseg.models import SegDataPreProcessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class RGBXDataPreProcessor(SegDataPreProcessor):

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        num_imgs = inputs[0].size(0) // 3
        if self.channel_conversion and inputs[0].size(0) % 3 == 0:
            # inputs = [_input[[2, 1, 0], ...] for _input in inputs]
            color_index = []
            ori_index = range(inputs[0].size(0))
            for i in range(num_imgs):
                color_index.extend(ori_index[3 * i:3 * i + 3][::-1])
            inputs = [_input[color_index, ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(inputs=inputs,
                                               data_samples=data_samples,
                                               size=self.size,
                                               size_divisor=self.size_divisor,
                                               pad_val=self.pad_val,
                                               seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)

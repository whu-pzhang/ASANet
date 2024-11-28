from typing import List, Optional

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList, SampleList
from mmseg.utils import add_prefix

import torch
import torch.nn as nn
from torch import Tensor


@MODELS.register_module()
class LateFusionSegmentor(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 backbone2: ConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs):
        super().__init__(backbone=backbone,
                         decode_head=decode_head,
                         data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if backbone2 is not None:
            self.backbone2 = MODELS.build(backbone2)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    @property
    def with_backbone2(self):
        return hasattr(self, 'backbone2') and self.backbone2 is not None

    def extract_feat(self, inputs) -> List[Tensor]:
        img = inputs[:, :3]
        img2 = inputs[:, 3:]

        if self.with_backbone2:
            x1 = self.backbone(img)
            x2 = self.backbone2(img2)
        else:
            x1, x2 = self.backbone(img, img2)

        return x1, x2

    def encode_decode(self, inputs, batch_img_metas: List[dict]) -> Tensor:
        x1, x2 = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x1, x2, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x1, x2 = self.extract_feat(inputs)
        return self.decode_head.forward(x1, x2)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x1, x2 = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x1, x2, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x1, x2, data_samples)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, inputs: List[Tensor], inputs2,
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, inputs2, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor], inputs2,
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, inputs2, data_samples,
                                         self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, inputs2, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses


@MODELS.register_module()
class EarlyFusionSegmentor(EncoderDecoder):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        img = inputs[:, :3]
        img2 = inputs[:, 3:]
        x = self.backbone(img, img2)
        return x

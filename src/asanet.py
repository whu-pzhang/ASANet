import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from .layers import SimpleFusionModule
from .layers import SFM, CFM



@MODELS.register_module()
class ASANet(BaseModule):
    '''
    ASANet.
    '''

    def __init__(self,
                 backbone,
                 frm_cfg=dict(act_cfg=dict(type='ReLU'), reduction=1),
                 ffm_cfg=dict(dict(type='mmpretrain.LN2d', eps=1e-6)),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)

        self.depths = self.backbone.depths
        self.channels = self.backbone.channels
        self.num_stages = self.backbone.num_stages
        self.out_indices = self.backbone.out_indices

        self.frm_cfg = frm_cfg
        self.ffm_cfg = ffm_cfg
        # FRM
        if frm_cfg is not None:
            if frm_cfg['type'] =='SFM':
                self.FRMs = nn.ModuleList([
                    SFM(self.channels[i],reduction=16)
                    for i in range(self.num_stages)
                ])
        # FFM
        self.FFMs = nn.ModuleList()
        for i in range(self.num_stages):
            channels = self.channels[i]
            if ffm_cfg is not None:
                if ffm_cfg['type'] == 'CFM':
                    self.FFMs.append(CFM(channels))
            else:
                self.FFMs.append(
                    SimpleFusionModule(in_channels=channels, mode='sum'))
                
    def forward(self, x_opt, x_sar):
        outs = []

        for i in range(self.num_stages):
            x_opt = self.backbone.downsample_layers[i](x_opt)
            x_opt = self.backbone.stages[i](x_opt)

            x_sar = self.backbone2.downsample_layers[i](x_sar)
            x_sar = self.backbone2.stages[i](x_sar)

            # interactive
            if self.frm_cfg is not None:
                x_opt, x_sar = self.FRMs[i](x_opt, x_sar)

            if i in self.out_indices:
                opt_norm_layer = getattr(self.backbone, f'norm{i}')
                sar_norm_layer = getattr(self.backbone2, f'norm{i}')
                x_opt = opt_norm_layer(x_opt)
                x_sar = sar_norm_layer(x_sar)
                x_fused = self.FFMs[i](x_opt, x_sar)
                outs.append(x_fused)

        return tuple(outs)

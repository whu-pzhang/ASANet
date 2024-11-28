import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmengine.model import BaseModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class VFuseNetBackbone(BaseModule):

    def __init__(self,
                 backbone,
                 stage_channels=[64, 128, 256, 512, 512],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)

        self.num_stages = len(self.backbone.stage_blocks)
        self.out_indices = self.backbone.out_indices

        self.fuse_convs = nn.ModuleList([
            ConvModule(2 * stage_channels[i],
                       stage_channels[i],
                       kernel_size=3,
                       padding=1,
                       norm_cfg=dict(type='BN'),
                       act_cfg=dict(type='ReLU'))
            for i in range(self.num_stages)
        ])

        self.unpool = nn.MaxUnpool2d(2)
        self.decoder_convs = nn.ModuleList()
        # 64,128,256,512,512
        stage_blocks = self.backbone.stage_blocks
        enc_in_channels = [64 * 2**i for i in range(self.num_stages) if i < 4]
        enc_in_channels.append(512)
        for i in range(self.num_stages):  # 0,1,2,3,4
            in_channels = enc_in_channels[i]
            out_channels = 64 if i == 0 else enc_in_channels[i - 1]
            num_block = stage_blocks[i]
            if i == 0:
                num_block -= 1
            layers = self.make_decoder_layer(in_channels, out_channels,
                                             num_block, conv_cfg, norm_cfg,
                                             act_cfg)

            self.decoder_convs.append(nn.Sequential(*layers))

    def forward(self, x1, x2):
        x1_outs, pool_masks = self.vgg_forward(self.backbone, x1)
        x2_outs, _ = self.vgg_forward(self.backbone2, x2)

        enc_outs = []
        for i in range(self.num_stages):
            x = torch.cat([x1_outs[i], x2_outs[i]], dim=1)
            fuse = self.fuse_convs[i](x)
            enc_outs.append(x1_outs[i] + x2_outs[i] + fuse)

        outs = []
        for i in reversed(range(self.num_stages)):
            x = self.unpool(enc_outs[i], pool_masks[i])
            x = self.decoder_convs[i](x)
            outs.append(x)

        return tuple(outs)

    def vgg_forward(self, module, x):
        outs = []
        pool_masks = []
        vgg_layers = getattr(module, 'features')
        pool_layer = getattr(module, 'pool')
        for i in range(len(module.stage_blocks)):
            for j in range(*module.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            x, mask = pool_layer(x)

            if i in module.out_indices:
                outs.append(x)
                pool_masks.append(mask)

        return tuple(outs), pool_masks

    def make_decoder_layer(self,
                           in_channels,
                           out_channels,
                           num_blocks,
                           conv_cfg=None,
                           norm_cfg=None,
                           act_cfg=dict(type='ReLU')):
        layers = []
        for i in range(num_blocks):
            channels = in_channels if i < num_blocks - 1 else out_channels
            layer = ConvModule(in_channels,
                               channels,
                               kernel_size=3,
                               padding=1,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)
            layers.append(layer)

        return layers


@MODELS.register_module()
class VGGBackbone(BaseModule):
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 num_classes=-1,
                 num_stages=5,
                 out_indices=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1., layer=['_BatchNorm']),
                     dict(type='Normal', std=0.01, layer=['Linear'])
                 ]):
        super().__init__(init_cfg=init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        with_norm = norm_cfg is not None

        if out_indices is None:
            out_indices = (5, ) if num_classes > 0 else (4, )
        self.out_indices = out_indices

        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1
            # num_modules = num_blocks
            end_idx = start_idx + num_modules
            out_channels = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(self.in_channels,
                                       out_channels,
                                       num_blocks,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg,
                                       dilation=1,
                                       with_norm=with_norm)
            vgg_layers.extend(vgg_layer)
            self.in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        self.pool = nn.MaxPool2d(2, ceil_mode=ceil_mode, return_indices=True)

    def forward(self, x):
        outs = []
        pool_masks = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)

            x, mask = self.pool(x)
            if i in self.out_indices:
                outs.append(x)
                pool_masks.append(mask)

        return tuple(outs), pool_masks

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def make_vgg_layer(in_channels,
                   out_channels,
                   num_blocks,
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=dict(type='ReLU'),
                   dilation=1,
                   with_norm=False):
    layers = []
    for _ in range(num_blocks):
        layer = ConvModule(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           dilation=dilation,
                           padding=dilation,
                           bias=True,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels
    layers.append(nn.Identity())

    return layers


if __name__ == '__main__':

    from mmpretrain.models.backbones import VGG

    b = VFuseNetBackbone(
        backbone=dict(
            type='VGGBackbone',
            depth=16,
            num_stages=5,
            out_indices=(0, 1, 2, 3, 4),
            norm_cfg=dict(type='BN'),
            # init_cfg=dict(type='Pretrained',
            #               checkpoint=checkpoint_file,
            #               prefix='backbone.'),
        ),
        stage_channels=[64, 128, 256, 512, 512])

    img1 = torch.randn(4, 3, 512, 512)
    img2 = torch.randn(4, 3, 512, 512)
    outs = b(img1, img2)

    loss = 0
    for f in outs:
        loss += torch.sum(f)

    loss.backward()

    for name, param in b.named_parameters():
        if param.grad is None:
            print(name)

    # print(m)

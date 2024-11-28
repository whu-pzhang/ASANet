import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from mmcv.cnn import ConvModule, build_norm_layer, build_activation_layer
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize


@MODELS.register_module()
class SAGateBackbone(nn.Module):
    '''
    
    Bi-directional Cross-Modality Feature Propagation with 
    Separation-and-Aggregation Gate for RGB-D Semantic Segmentation
    '''

    def __init__(self, backbone, sagate_channels, bn_momentum=0.1):
        super().__init__()
        assert backbone.get('type', None) == 'ResNetV1c'
        self.backbone1 = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)

        self.sagates = nn.ModuleList()
        for c in sagate_channels:
            self.sagates.append(
                SAGate(in_channels=c, out_channels=c, bn_momentum=bn_momentum))

    def forward(self, x1, x2):
        x1 = self.backbone1.stem(x1)
        x1 = self.backbone1.maxpool(x1)

        x2 = self.backbone2.stem(x2)
        x2 = self.backbone2.maxpool(x2)

        merges = []
        for i, layer_name in enumerate(self.backbone1.res_layers):
            res_layer1 = getattr(self.backbone1, layer_name)
            x1 = res_layer1(x1)

            res_layer2 = getattr(self.backbone2, layer_name)
            x2 = res_layer2(x2)

            (x1, x2), merge = self.sagates[i]([x1, x2])

            merges.append(merge)

        return tuple(merges)


class FilterLayer(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.out_channels = out_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_channels, 1, 1)
        return y


class FSP(nn.Module):
    '''Feature Separation Part'''

    def __init__(self, in_channels, out_channels, reduction=16) -> None:
        super().__init__()

        self.filter = FilterLayer(2 * in_channels, out_channels, reduction)

    def forward(self, guide_path, main_path):
        combined = torch.cat([guide_path, main_path], dim=1)
        channel_weight = self.filter(combined)
        out = main_path + channel_weight * guide_path
        return out


class SAGate(nn.Module):
    ''' SAGate '''

    def __init__(self, in_planes, out_planes, reduction=16):
        super().__init__()
        self.in_planes = in_planes

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_hha = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb, hha = x
        b, c, h, w = rgb.size()

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r],
                                     dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:
                                                                  1, :, :], attention_vector[:,
                                                                                             1:
                                                                                             2, :, :]
        merge_feature = rgb * attention_vector_l + hha * attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        return [rgb_out, hha_out], merge_feature


@MODELS.register_module()
class SAGateHead(BaseDecodeHead):

    def __init__(self,
                 dilations=(6, 12, 18),
                 c1_in_channels=256,
                 c1_channels=48,
                 **kwargs):
        super().__init__(**kwargs)

        self.aspp = ASPP(self.in_channels,
                         self.channels,
                         dilations,
                         norm_cfg=self.norm_cfg)

        self.c1_bottleneck = ConvModule(c1_in_channels,
                                        c1_channels,
                                        1,
                                        conv_cfg=self.conv_cfg,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg)

        self.last_conv = nn.Sequential(
            ConvModule(self.channels + c1_channels,
                       self.channels,
                       3,
                       padding=1,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            ConvModule(self.channels,
                       self.channels,
                       3,
                       padding=1,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        aspp_out = self.aspp(x)
        c1_output = self.c1_bottleneck(inputs[0])
        output = resize(input=aspp_out,
                        size=c1_output.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
        output = torch.cat([output, c1_output], dim=1)
        output = self.last_conv(output)
        output = self.cls_seg(output)
        return output


class ASPP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 norm_cfg=dict(type='BN')):
        super(ASPP, self).__init__()
        self.pooling_size = None

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      bias=False,
                      dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = build_norm_layer(norm_cfg, out_channels * 4)[1]

        self.global_pooling_conv = nn.Conv2d(in_channels,
                                             out_channels,
                                             1,
                                             bias=False)
        self.global_pooling_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.red_conv = nn.Conv2d(out_channels * 4,
                                  out_channels,
                                  1,
                                  bias=False)
        self.pool_red_conv = nn.Conv2d(out_channels,
                                       out_channels,
                                       1,
                                       bias=False)
        self.red_bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)  # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = ((pooling_size[1] - 1) // 2, (pooling_size[1] - 1) //
                       2 if pooling_size[1] % 2 == 1 else
                       (pooling_size[1] - 1) // 2 + 1,
                       (pooling_size[0] - 1) // 2, (pooling_size[0] - 1) //
                       2 if pooling_size[0] % 2 == 1 else
                       (pooling_size[0] - 1) // 2 + 1)

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool


class DualBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 downsample=None,
                 inplace=True):
        super(DualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = nn.Conv2d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.hha_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.hha_bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.hha_conv2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   bias=False)
        self.hha_bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.hha_conv3 = nn.Conv2d(planes,
                                   planes * self.expansion,
                                   kernel_size=1,
                                   bias=False)
        self.hha_bn3 = build_norm_layer(norm_cfg, planes * self.expansion)[1]
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv3(out2)
        out2 = self.hha_bn3(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)

        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


@MODELS.register_module()
class DualResNet(BaseModule):

    arch_settings = {
        50: (DualBottleneck, (3, 4, 6, 3)),
        101: (DualBottleneck, (3, 4, 23, 3)),
        152: (DualBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 norm_cfg=dict(type='BN', eps=1e-5, momentum=0.1),
                 deep_stem=False,
                 stem_width=32,
                 inplace=True,
                 init_cfg=None):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super().__init__(init_cfg=init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block, layers = self.arch_settings[depth]

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,
                          stem_width,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                build_norm_layer(norm_cfg, stem_width)[1],
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width,
                          stem_width,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                build_norm_layer(norm_cfg, stem_width)[1],
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width,
                          stem_width * 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
            )
            self.hha_conv1 = nn.Sequential(
                nn.Conv2d(3,
                          stem_width,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                build_norm_layer(norm_cfg, stem_width)[1],
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width,
                          stem_width,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
                build_norm_layer(norm_cfg, stem_width)[1],
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width,
                          stem_width * 2,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3,
                                   64,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.hha_conv1 = nn.Conv2d(3,
                                       64,
                                       kernel_size=7,
                                       stride=2,
                                       padding=3,
                                       bias=False)

        self.bn1 = build_norm_layer(norm_cfg,
                                    stem_width * 2 if deep_stem else 64)[1]
        self.hha_bn1 = build_norm_layer(norm_cfg,
                                        stem_width * 2 if deep_stem else 64)[1]
        self.relu = nn.ReLU(inplace=inplace)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hha_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_cfg, 64, layers[0], inplace)
        self.layer2 = self._make_layer(block,
                                       norm_cfg,
                                       128,
                                       layers[1],
                                       inplace,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       norm_cfg,
                                       256,
                                       layers[2],
                                       inplace,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       norm_cfg,
                                       512,
                                       layers[3],
                                       inplace,
                                       stride=2)

        self.sagates = nn.ModuleList([
            SAGate(in_planes=256, out_planes=256),
            SAGate(in_planes=512, out_planes=512),
            SAGate(in_planes=1024, out_planes=1024),
            SAGate(in_planes=2048, out_planes=2048)
        ])

    def _make_layer(self,
                    block,
                    norm_cfg,
                    planes,
                    blocks,
                    inplace=True,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, norm_cfg, downsample,
                  inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      norm_cfg=norm_cfg,
                      inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.hha_conv1(x2)
        x2 = self.hha_bn1(x2)
        x2 = self.hha_relu(x2)
        x2 = self.hha_maxpool(x2)

        x = [x1, x2]
        blocks = []
        merges = []
        x = self.layer1(x)
        x, merge = self.sagates[0](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer2(x)
        x, merge = self.sagates[1](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer3(x)
        x, merge = self.sagates[2](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer4(x)
        x, merge = self.sagates[3](x)
        blocks.append(x)
        merges.append(merge)

        return tuple(merges)

    def init_weights(self):
        if self.init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]
        else:
            ckpt_file = self.init_cfg.get('checkpoint')
            checkpoint = _load_checkpoint(ckpt_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('.bn.', '.')] = v
                if k.find('conv1') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('conv1', 'hha_conv1')] = v
                if k.find('conv2') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('conv2', 'hha_conv2')] = v
                if k.find('conv3') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('conv3', 'hha_conv3')] = v
                if k.find('bn1') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('bn1', 'hha_bn1')] = v
                if k.find('bn2') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('bn2', 'hha_bn2')] = v
                if k.find('bn3') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('bn3', 'hha_bn3')] = v
                if k.find('downsample') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('downsample',
                                             'hha_downsample')] = v

            _load_checkpoint_to_model(self, new_state_dict)


if __name__ == '__main__':
    m = DualResNet(depth=101,
                   deep_stem=True,
                   stem_width=64,
                   init_cfg=dict(type='Pretrain',
                                 checkpoint='pretrain/resnet101_v1c.pth'))
    m.init_weights()

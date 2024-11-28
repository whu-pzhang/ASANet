import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from mmcv.cnn.bricks import DropPath
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_dropout
from mmpretrain.models.utils import build_norm_layer
from mmseg.models.utils import resize

from mmseg.registry import MODELS


# =================== ConvNeXtV2 =================
class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """

        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
            1, -1, 1, 1) + x
        return x


class ConvNeXtBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='mmpretrain.LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        groups=in_channels,
                                        **dw_conv_cfg)
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        self.pointwise_conv1 = nn.Conv2d(in_channels,
                                         mid_channels,
                                         kernel_size=1)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = nn.Conv2d(mid_channels,
                                         in_channels,
                                         kernel_size=1)
        self.grn = GRN(mid_channels)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            x = self.norm(x, data_format='channel_first')
            x = self.pointwise_conv1(x)
            x = self.act(x)

            x = self.grn(x, data_format='channel_first')
            x = self.pointwise_conv2(x)

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
    
# ====================== ScConv ================================
## SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy.
class GroupBatchnorm2d(nn.Module):

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    '''Spatial and Reconstruct Unit'''

    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False):
        super().__init__()

        self.gn = nn.GroupNorm(
            num_channels=oup_channels,
            num_groups=group_num) if torch_gn else GroupBatchnorm2d(
                c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = torch.sigmoid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    Channel Reduction Unit.
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel,
                                  up_channel // squeeze_radio,
                                  kernel_size=1,
                                  bias=False)
        self.squeeze2 = nn.Conv2d(low_channel,
                                  low_channel // squeeze_radio,
                                  kernel_size=1,
                                  bias=False)
        #up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio,
                             op_channel,
                             kernel_size=group_kernel_size,
                             stride=1,
                             padding=group_kernel_size // 2,
                             groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio,
                              op_channel,
                              kernel_size=1,
                              bias=False)
        #low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio,
                              op_channel - low_channel // squeeze_radio,
                              kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):

    def __init__(
        self,
        op_channel: int,
        group_num: int = 4,
        gate_treshold: float = 0.5,
        alpha: float = 1 / 2,
        squeeze_radio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


# ======================= FaPN ==========================
class FSM(nn.Module):
    """Feature Selection Module."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        w = self.conv1(self.avg_pool(x))
        out = self.conv2(x + x * w)
        return out


class DCNv2(nn.Module):

    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FAM(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s,
                                    size=feat_l.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm

# ==================== CMX =======================
class MLPChannelWeights(nn.Module):

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2), nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1,
                                    1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class CNNChannelWeights(nn.Module):

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 4,
                      self.dim * 4 // reduction,
                      kernel_size=1,
                      bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim * 4 // reduction,
                      self.dim * 2,
                      kernel_size=1,
                      bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat([x1, x2], dim=1)
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        y = torch.cat([avg, max], dim=1)  # B,4C,1,1
        y = self.mlp(y)  # B,2C,1,1
        channel_weights = y.reshape(B, 2, self.dim, 1, 1)
        # B,2,C,1,1 -> 2,B,C,1,1
        channel_weights = channel_weights.permute(1, 0, 2, 3, 4).contiguous()
        return channel_weights


class CNNSpatialWeights(nn.Module):

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat([x1, x2], dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W)
        # B,2,1,H,W -> 2,B,1,H,W
        spatial_weights = spatial_weights.permute(1, 0, 2, 3, 4).contiguous()
        return spatial_weights


class FeatureRectifyModule(nn.Module):

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.channel_weights = CNNChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = CNNSpatialWeights(dim=dim, reduction=reduction)

        # self.lambda_c = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.lambda_s = nn.Parameter(torch.ones(1), requires_grad=True)
        self.lambda_c = 0.5
        self.lambda_s = 0.5

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)

        out_x1 = x1 + self.lambda_c * channel_weights[
            1] * x2 + self.lambda_s * spatial_weights[1] * x2

        out_x2 = x2 + self.lambda_c * channel_weights[
            0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class FeatureFusionModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scconv = ScConv(2 * dim)
        self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1)

        # GRN: increase the contrast and selectivity of channels
        # self.grn = GRN(dim)
        # self.point_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x1, x2):
        # x1, x2 = self.cross_attn(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.scconv(x)
        x = self.proj(x)

        # x = self.attn(x)
        # x = self.grn(x)
        # x = self.point_conv(x)

        return x


# ====================== custom modules =========================
class SimpleFusionModule(nn.Module):

    def __init__(self, in_channels, mode='sum') -> None:
        super().__init__()
        assert mode in ['sum', 'concat']
        self.mode = mode
        if mode == 'concat':
            self.proj = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x1, x2):
        
        if self.mode == 'sum':
            x = x1 + x2
        elif self.mode == 'concat':
            x = torch.cat([x1, x2], dim=1)

        x = self.proj(x)
        return x

    
# ======================= ASANet ==========================

class SFM(nn.Module):
    '''
    The Semantic Focusing Module
    '''
    def __init__(self,channels, reduction=16) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(32, channels//reduction)
        self.sconv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1)
        )
        
    def forward(self, x1, x2):
        
        # for rgb diff
        diff_rgb = x1 - x2
        s_rgb = self.maxpool(diff_rgb)
        z_rgb = self.sconv(s_rgb)
        attn_rgb = torch.sigmoid(z_rgb)
        x11 = x1 + x1 * (attn_rgb)
        
        # for sar diff
        diff_sar = x2 - x1
        s1_sar = self.maxpool(diff_sar)
        z_sar = self.sconv(s1_sar)
        attn_sar = torch.sigmoid(z_sar)
        x22 = x2 + x2 * (attn_sar)
        
        return x11, x22

        
class MLP(nn.Module):
    """the Multilayer Perceptron.

    Args:
        in_features (int): the input feature.
        hidden_features (int): the hidden feature.
        out_features (int): the output feature.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        drop (float): Probability of an element to be zeroed.
            Default 0.0
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvModule(
            in_features,
            hidden_features,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.dwconv = ConvModule(
            hidden_features,
            hidden_features,
            kernel_size=3,
            padding=1,
            groups=hidden_features,
            norm_cfg=None,
            act_cfg=act_cfg,
        )

        self.fc2 = ConvModule(
            hidden_features,
            out_features,
            1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.drop = build_dropout(dict(type='Dropout', drop_prob=drop))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CFM(nn.Module):
    "The Cascade Fusion Module"
    
    def __init__(self, channels, reduction=16, drop=0.0) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.catconv1 = nn.Conv2d(2 * channels, channels, kernel_size=1)
        mlp_hidden_dim = int(channels * 4)
        act_cfg = dict(type='ReLU6')
        self.mlp1 = MLP(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop,
        )
        self.mlp2 = MLP(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop,
        )
        self.catconv2 = nn.Conv2d(2 * channels, channels, kernel_size=1)

        self.fc1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x1, x2):
        x_concat = self.catconv1(torch.cat([x1, x2], dim=1))
        # x_sum = x1 + x2

        x_gap = self.avgpool(x_concat)
        mlp1 = torch.sigmoid(self.mlp1(x_gap))
        mlp2 = torch.sigmoid(self.mlp2(x_gap))
        w_x1 = x1*mlp1
        w_x2 = x2*mlp2
        
        w_concat = self.catconv2(torch.cat([w_x1, w_x2], dim=1))
        
        fc_1 = F.softmax(self.fc1(w_concat), dim=1)
        fc_2 = F.softmax(self.fc2(w_concat), dim=1)
        
        out_x1 = w_x1 * fc_1
        out_x2 = w_x2 * fc_2
        
        return out_x1 + out_x2
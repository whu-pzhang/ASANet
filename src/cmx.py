import math

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.utils import to_2tuple
from mmengine.model.weight_init import trunc_normal_init, constant_init, normal_init, kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model

from mmseg.registry import MODELS
from mmseg.models.utils import nlc_to_nchw, resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class CMXBackbone(BaseModule):
    '''
    CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers.
    '''

    def __init__(self, backbone, fuse_norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)

        self.num_heads = self.backbone.num_heads
        self.embed_dims = self.backbone.embed_dims
        self.num_layers = self.backbone.num_layers
        self.out_indices = self.backbone.out_indices

        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()

        for i, num_layer in enumerate(self.num_layers):
            embed_dims_i = self.embed_dims * self.num_heads[i]

            self.FRMs.append(
                FeatureRectifyModule(dim=embed_dims_i, reduction=1))
            self.FFMs.append(
                FeatureFusionModule(dim=embed_dims_i,
                                    reduction=1,
                                    num_heads=self.num_heads[i],
                                    norm_cfg=fuse_norm_cfg))

    def forward(self, x_opt, x_sar):
        outs = []
        for i, (layer1, layer2) in enumerate(
                zip(self.backbone.layers, self.backbone2.layers)):
            # optical branch
            x_opt, hw_shape = layer1[0](x_opt)  # pacth_embed
            for block in layer1[1]:  # layer
                x_opt = block(x_opt, hw_shape)
            x_opt = layer1[2](x_opt)  # norm
            x_opt = nlc_to_nchw(x_opt, hw_shape)
            # sar branch
            x_sar, hw_shape = layer2[0](x_sar)  # pacth_embed
            for block in layer2[1]:  # layer
                x_sar = block(x_sar, hw_shape)
            x_sar = layer2[2](x_sar)  # norm
            x_sar = nlc_to_nchw(x_sar, hw_shape)

            # interactive
            x_opt, x_sar = self.FRMs[i](x_opt, x_sar)
            x_fused = self.FFMs[i](x_opt, x_sar)

            if i in self.out_indices:
                outs.append(x_fused)

        return tuple(outs)


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim,
                                dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True,
                                groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2,
                      1).reshape(B, C, H,
                                 W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim,
                                dim,
                                kernel_size=sr_ratio,
                                stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            #  norm_layer=nn.LayerNorm,
            norm_cfg=dict(type='LN', eps=1e-6),
            sr_ratio=1):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W


@MODELS.register_module()
class RGBXTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 fuse_norm_cfg=dict(type='BN'),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size,
                                              patch_size=7,
                                              stride=4,
                                              in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size,
                                                    patch_size=7,
                                                    stride=4,
                                                    in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4,
                                                    patch_size=3,
                                                    stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8,
                                                    patch_size=3,
                                                    stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16,
                                                    patch_size=3,
                                                    stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0],
                  num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = build_norm_layer(norm_cfg, embed_dims[0])[1]

        self.extra_block1 = nn.ModuleList([
            Block(dim=embed_dims[0],
                  num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.extra_norm1 = build_norm_layer(norm_cfg, embed_dims[0])[1]
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1],
                  num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = build_norm_layer(norm_cfg, embed_dims[1])[1]

        self.extra_block2 = nn.ModuleList([
            Block(dim=embed_dims[1],
                  num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + 1],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.extra_norm2 = build_norm_layer(norm_cfg, embed_dims[1])[1]

        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2],
                  num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = build_norm_layer(norm_cfg, embed_dims[2])[1]

        self.extra_block3 = nn.ModuleList([
            Block(dim=embed_dims[2],
                  num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.extra_norm3 = build_norm_layer(norm_cfg, embed_dims[2])[1]

        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3],
                  num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = build_norm_layer(norm_cfg, embed_dims[3])[1]

        self.extra_block4 = nn.ModuleList([
            Block(dim=embed_dims[3],
                  num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_cfg=norm_cfg,
                  sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.extra_norm4 = build_norm_layer(norm_cfg, embed_dims[3])[1]

        cur += depths[3]

        self.FRMs = nn.ModuleList([
            FeatureRectifyModule(dim=embed_dims[0], reduction=1),
            FeatureRectifyModule(dim=embed_dims[1], reduction=1),
            FeatureRectifyModule(dim=embed_dims[2], reduction=1),
            FeatureRectifyModule(dim=embed_dims[3], reduction=1)
        ])

        self.FFMs = nn.ModuleList([
            FeatureFusionModule(dim=embed_dims[0],
                                reduction=1,
                                num_heads=num_heads[0],
                                norm_cfg=fuse_norm_cfg),
            FeatureFusionModule(dim=embed_dims[1],
                                reduction=1,
                                num_heads=num_heads[1],
                                norm_cfg=fuse_norm_cfg),
            FeatureFusionModule(dim=embed_dims[2],
                                reduction=1,
                                num_heads=num_heads[2],
                                norm_cfg=fuse_norm_cfg),
            FeatureFusionModule(dim=embed_dims[3],
                                reduction=1,
                                num_heads=num_heads[3],
                                norm_cfg=fuse_norm_cfg)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def init_weights(self):
        if self.init_cfg is None:
            self.apply(self._init_weights)
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
                if k.find('patch_embed') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('patch_embed',
                                             'extra_patch_embed')] = v
                elif k.find('block') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('block', 'extra_block')] = v
                elif k.find('norm') >= 0:
                    new_state_dict[k] = v
                    new_state_dict[k.replace('norm', 'extra_norm')] = v

            _load_checkpoint_to_model(self, new_state_dict)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x N x H x W
        """
        B = x_rgb.shape[0]
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        x_fused = self.FFMs[0](x_rgb, x_e)
        outs.append(x_fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        x_fused = self.FFMs[1](x_rgb, x_e)
        outs.append(x_fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        x_fused = self.FFMs[2](x_rgb, x_e)
        outs.append(x_fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        x_fused = self.FFMs[3](x_rgb, x_e)
        outs.append(x_fused)

        return outs

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


@MODELS.register_module()
class MLPDecoderHead(BaseDecodeHead):

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(input_transform='multiple_select',
                         init_cfg=init_cfg,
                         **kwargs)

        num_inputs = len(self.in_channels)

        self.layers = nn.ModuleList()
        for i in range(num_inputs):
            self.layers.append(
                MLP(input_dim=self.in_channels[i], embed_dim=self.channels))

        self.fusion_conv = ConvModule(in_channels=self.channels * num_inputs,
                                      out_channels=self.channels,
                                      kernel_size=1,
                                      bias=True,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)

        self.apply(self._init_weights)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        outs = []
        for idx in reversed(range(len(inputs))):
            x = inputs[idx]
            b, _, h, w = x.shape
            layer = self.layers[idx]
            x = layer(x).permute(0, 2, 1).reshape(b, -1, h, w)
            outs.append(
                resize(input=x,
                       size=inputs[0].shape[2:],
                       mode='bilinear',
                       align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m,
                         mode='fan_in',
                         nonlinearity='relu',
                         distribution='normal')


class MLP(nn.Module):
    """
    Linear Embedding: 
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # N,C,H,W -> N,C,L -> N,L,C
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ChannelWeights(nn.Module):

    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

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


class SpatialWeights(nn.Module):

    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H,
                                              W).permute(1, 0, 2, 3,
                                                         4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):

    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[
            1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[
            0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads,
                        C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads,
                        C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads,
                                      C // self.num_heads).permute(
                                          2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads,
                                      C // self.num_heads).permute(
                                          2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):

    def __init__(self,
                 dim,
                 reduction=1,
                 num_heads=None,
                 norm_cfg=dict(type='LN')):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]

    def forward(self, x1, x2):
        # x_res, x_inter
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=1,
                 norm_cfg=dict(type='BN')):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels // reduction,
                      kernel_size=1,
                      bias=True),
            nn.Conv2d(out_channels // reduction,
                      out_channels // reduction,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True,
                      groups=out_channels // reduction), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction,
                      out_channels,
                      kernel_size=1,
                      bias=True),
            build_norm_layer(norm_cfg, out_channels)[1])
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):

    def __init__(self,
                 dim,
                 reduction=1,
                 num_heads=None,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.cross = CrossPath(dim=dim,
                               reduction=reduction,
                               num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2,
                                        out_channels=dim,
                                        reduction=reduction,
                                        norm_cfg=norm_cfg)

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # B,C,H,W -> B,L,C
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02, bias=0.)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m, val=1.0, bias=0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

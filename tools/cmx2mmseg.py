import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_mit(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        # patch embedding conversion
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i-1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')

        # transformer encoder layer conversion
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i-1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                new_k = new_k.replace('fc2.', '4.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        # norm layer conversion
        elif k.startswith('norm'):
            stage_i = int(k.split('.')[0].replace('norm', ''))
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i-1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt


def convert_cmx_head(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('linear_c'):
            stage_i = int(k[8])
            new_k = k.replace(f'linear_c{stage_i}', f'layers.{stage_i-1}')
            new_v = v

        elif 'fuse' in k:
            new_k = k.replace('linear_fuse', 'fusion_conv')
            new_k = new_k.replace('0', 'conv')
            new_k = new_k.replace('1', 'bn')
            new_v = v
        elif 'pred' in k:
            new_k = k.replace('linear_pred', 'conv_seg')
            new_v = v
        else:
            new_k = k
            new_v = v

        new_ckpt[new_k] = new_v

    return new_ckpt


def convert_cmx(ckpt):
    backbone1 = OrderedDict()
    backbone2 = OrderedDict()
    head = OrderedDict()

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if 'decode_head' in k:
            new_k = k.replace('decode_head.', '')
            head[new_k] = v
        else:
            if 'extra' in k:
                new_k = k.replace('backbone.', '').replace('extra_', '')
                backbone2[new_k] = v
            elif ('FRMs' in k) or ('FFMs' in k):
                new_ckpt[k] = v
            else:
                new_k = k.replace('backbone.', '')
                backbone1[new_k] = v

    backbone1 = convert_mit(backbone1)
    backbone2 = convert_mit(backbone2)
    head = convert_cmx_head(head)
    # print(head.keys())

    for k, v in backbone1.items():
        new_k = 'backbone.backbone.' + k
        new_ckpt[new_k] = v

    for k, v in backbone2.items():
        new_k = 'backbone.backbone2.' + k
        new_ckpt[new_k] = v

    for k, v in head.items():
        new_k = 'decode_head.' + k
        new_ckpt[new_k] = v

    return new_ckpt


def load_ckpt(pth_path):
    checkpoint = CheckpointLoader.load_checkpoint(pth_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    return state_dict


def main():

    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    ori_ckpt = load_ckpt(args.src)

    ckpt = convert_cmx(ori_ckpt)

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(ckpt, args.dst)


if __name__ == '__main__':
    main()

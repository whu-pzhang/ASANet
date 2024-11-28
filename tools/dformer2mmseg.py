import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def load_ckpt(pth_path):
    checkpoint = CheckpointLoader.load_checkpoint(pth_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    return state_dict


def convert_dformer(ckpt):
    return ckpt


def main():

    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained segformer to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    ori_ckpt = load_ckpt(args.src)

    ckpt = convert_dformer(ori_ckpt)

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(ckpt, args.dst)

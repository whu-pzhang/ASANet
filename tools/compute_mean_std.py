from pathlib import Path
import os.path as osp

from tqdm import tqdm
from skimage.io import imread
import numpy as np
import cv2

from mmengine.utils import scandir


def calcuate_mean_std(root, suffix, num_channels=None, max_value=None):
    root = Path(root).expanduser().resolve()
    files = list(scandir(root, suffix=(suffix)))

    pixel_num = 0  # store all pixel number in the dataset
    if num_channels is None:
        num_channels = cv2.imread(osp.join(root, files[0]), -1).shape[-1]
    print(f'num_channels = {num_channels}')

    channel_sum = np.zeros(num_channels)
    channel_sum_squared = np.zeros(num_channels)
    channel_min = np.zeros(num_channels)
    channel_max = np.zeros(num_channels)

    for img_path in tqdm(files):
        img_path = osp.join(root, img_path)
        # image in M*N*num_channels shape, channel in BGR order
        im = cv2.imread(img_path, -1).astype(np.float32)
        # im = im / 255.0
        if max_value is not None:
            im = im / max_value
        pixel_num += im.shape[0] * im.shape[1]
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))

    return mean, std


def get_channel_infos(img_file, num_channels=None, max_value=None):
    img = cv2.imread(img_file).astype(np.float32)
    if num_channels is None:
        num_channels = img.shape[-1]

    if max_value is not None:
        im = im / max_value
    pixel_num = im.shape[0] * im.shape[1]
    channel_sum = np.sum(im, axis=(0, 1))
    channel_sum_squared = np.sum(np.square(im), axis=(0, 1))

    img_info = dict(pixel_num=pixel_num,
                    channel_sum=channel_sum,
                    channel_sum_squared=channel_sum_squared)

    return img_info


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='training label directory')
    parser.add_argument('--suffix', type=str, help='suffix of mask file')
    parser.add_argument('-c',
                        '--channels',
                        type=int,
                        default=None,
                        help='number of channels')
    parser.add_argument("--maxvalue",
                        default=None,
                        type=float,
                        help="max value of all images default: {255}")

    args = parser.parse_args()
    print(args)

    np.set_printoptions(precision=3, suppress=True)
    mean, std = calcuate_mean_std(args.root, args.suffix, args.channels,
                                  args.maxvalue)
    print(f"mean = {mean.tolist()}\nstd = {std.tolist()}")


if __name__ == "__main__":
    main()

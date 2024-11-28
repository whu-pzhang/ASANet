from pathlib import Path
from math import ceil
from itertools import product
from functools import partial

import cv2
from tqdm import tqdm
from mmengine.utils import (mkdir_or_exist, track_parallel_progress,
                            track_progress)


def split_single_image(filename,
                       opt_dir,
                       sar_dir,
                       label_dir,
                       out_opt_dir,
                       out_sar_dir,
                       out_lbl_dir,
                       x_size=512,
                       y_size=512):
    opt_file = opt_dir / filename
    opt = cv2.imread(str(opt_file), flags=cv2.IMREAD_UNCHANGED)
    height, width = opt.shape[:2]

    x_num = 1 if width <= x_size else ceil((width - x_size) / x_size + 1)
    x_offsets = [x_size * i for i in range(x_num)]
    if len(x_offsets) > 1 and x_offsets[-1] + x_size > width:
        x_offsets[-1] = width - x_size

    y_num = 1 if height <= y_size else ceil((height - y_size) / y_size + 1)
    y_offsets = [y_size * i for i in range(y_num)]
    if len(y_offsets) > 1 and y_offsets[-1] + y_size > height:
        y_offsets[-1] = height - y_size

    left_top_xy = list(product(x_offsets, y_offsets))

    sar_file = sar_dir / opt_file.name
    lbl_file = label_dir / opt_file.name
    basename = opt_file.stem
    suffix = opt_file.suffix

    sar = cv2.imread(str(sar_file), cv2.IMREAD_UNCHANGED)
    lbl = cv2.imread(str(lbl_file), cv2.IMREAD_UNCHANGED)
    lbl = (lbl / 10).astype(lbl.dtype)
    # pbar = tqdm(left_top_xy)
    for idx, (xoff, yoff) in enumerate(left_top_xy):
        opt_patch = opt[yoff:yoff + y_size, xoff:xoff + x_size]
        sar_patch = sar[yoff:yoff + y_size, xoff:xoff + x_size]
        lbl_patch = lbl[yoff:yoff + y_size, xoff:xoff + x_size]

        out_file = f'{basename}_{idx:03d}{suffix}'
        opt_file = out_opt_dir / out_file
        sar_file = out_sar_dir / out_file
        lbl_file = out_lbl_dir / out_file

        cv2.imwrite(str(opt_file), opt_patch)
        cv2.imwrite(str(sar_file), sar_patch)
        cv2.imwrite(str(lbl_file), lbl_patch)


def main():

    x_size = 512
    y_size = 512
    nproc = 4

    data_root = Path('data/whu-opt-sar')

    opt_dir = data_root / 'optical'
    sar_dir = data_root / 'sar'
    label_dir = data_root / 'lbl'

    for mode in ['train', 'test']:
        print(mode)
        with open(f'splits/{mode}_list.txt', 'r') as fp:
            file_list = [line.strip() for line in fp]

        out_dir = data_root / f'crop{x_size}' / mode
        out_opt_dir = out_dir / 'opt_dir'
        out_sar_dir = out_dir / 'sar_dir'
        out_lbl_dir = out_dir / 'ann_dir'
        mkdir_or_exist(out_opt_dir)
        mkdir_or_exist(out_sar_dir)
        mkdir_or_exist(out_lbl_dir)

        if nproc > 1:
            track_parallel_progress(partial(split_single_image,
                                            opt_dir=opt_dir,
                                            sar_dir=sar_dir,
                                            label_dir=label_dir,
                                            out_opt_dir=out_opt_dir,
                                            out_sar_dir=out_sar_dir,
                                            out_lbl_dir=out_lbl_dir,
                                            x_size=x_size,
                                            y_size=y_size),
                                    file_list,
                                    nproc=nproc)
        else:
            track_progress(
                partial(split_single_image,
                        opt_dir=opt_dir,
                        sar_dir=sar_dir,
                        label_dir=label_dir,
                        out_opt_dir=out_opt_dir,
                        out_sar_dir=out_sar_dir,
                        out_lbl_dir=out_lbl_dir,
                        x_size=x_size,
                        y_size=y_size), file_list)


if __name__ == '__main__':
    main()

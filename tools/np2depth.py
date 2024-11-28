import argparse
from pathlib import Path
from functools import partial

import numpy as np
from PIL import Image
from mmengine.utils import (mkdir_or_exist, track_parallel_progress,
                            track_progress)


def rerange(img, min_value=0, max_value=255):
    img_min_value = np.min(img)
    img_max_value = np.max(img)

    img = (img - img_min_value) / (img_max_value - img_min_value)
    img = img * (max_value - min_value) + min_value

    return img


def npy2depth(input_path, out_dir):
    img = np.load(input_path)
    img = rerange(img)

    img = Image.fromarray(img.astype(np.uint8))
    out_file = out_dir / f'{input_path.stem}.png'
    img.save(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--nproc', type=int, default=4)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    npy_files = list(Path(args.input_dir).glob('*.npy'))
    nproc = args.nproc

    if nproc > 1:
        track_parallel_progress(partial(npy2depth, out_dir=out_dir),
                                npy_files,
                                nproc=nproc)
    else:
        track_progress(partial(npy2depth, out_dir=out_dir), npy_files)


if __name__ == '__main__':
    main()

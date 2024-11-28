import argparse
from pathlib import Path
from functools import partial

import numpy as np
import rasterio as rio
from mmengine.utils import (mkdir_or_exist, track_parallel_progress,
                            track_progress)

def LinearEnhancement(input, channels, percent=2):
    out = []
    for i in range(channels):
        band = input[i,:,:] 

        low_value = np.percentile(band, percent)
        high_value = np.percentile(band, 100-percent)
        tmp = np.clip(band, a_min=low_value, a_max=high_value)
        band_out = ((tmp - low_value) / (high_value - low_value)) * 255
        band_out = np.squeeze(band_out)
        out.append(band_out)

    output = np.stack(out, axis=0)
    return output

def rerange(img, min_value=0, max_value=255):
    img_min_value = np.min(img)
    img_max_value = np.max(img)

    img = (img - img_min_value) / (img_max_value - img_min_value)
    img = img * (max_value - min_value) + min_value

    return img


def npy2depth(input_path, out_dir):
    src = rio.open(input_path)
    img = src.read()
    kwds = src.meta
    kwds.update(dtype='uint8')
    # img = np.load(input_path)
    img = LinearEnhancement(img, 1)

    out_file = out_dir / f'{input_path.stem}.tif'
    out_raster = rio.open(out_file, 'w', **kwds)
    out_raster.write(img)
    out_raster.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data1/pengbc/data122/pengbc/SAR_RGB_seg/case01/GF3_SAR', type=str)
    parser.add_argument('--output_dir',default='/data1/pengbc/code/dataset/PIEM-DATA/with_code/SAR8', type=str)
    parser.add_argument('--nproc', type=int, default=4)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    npy_files = list(Path(args.input_dir).glob('*.tif'))
    nproc = args.nproc

    if nproc > 1:
        track_parallel_progress(partial(npy2depth, out_dir=out_dir),
                                npy_files,
                                nproc=nproc)
    else:
        track_progress(partial(npy2depth, out_dir=out_dir), npy_files)


if __name__ == '__main__':
    main()

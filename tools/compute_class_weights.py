import os.path as osp
from pathlib import Path

import mmcv
import numpy as np
from mmengine.utils import scandir
from tqdm import tqdm


def calculate_class_weights(ann_roots,
                            num_classes,
                            reducez_zero_labels=False,
                            backend='pillow'):
    assert isinstance(ann_roots, (list, tuple))

    class_freq = np.zeros(num_classes)
    for root in ann_roots:
        root = Path(root).expanduser().resolve()
        label_list = list(scandir(root, suffix=('.jpg', '.png', '.tif')))
        for label_path in tqdm(label_list):
            label_path = osp.join(root, label_path)
            # label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            # label = np.array(Image.open(label_path))
            label = mmcv.imread(label_path, flag='unchanged', backend=backend)

            if reducez_zero_labels:
                label[label == 0] = 255
                label = label - 1
                label[label == 254] = 255

            mask = (label >= 0) & (label < num_classes)
            label = label[mask]

            cur_class_freq = np.bincount(label, minlength=num_classes)

            class_freq += cur_class_freq

    # https://github.com/openseg-group/OCNet.pytorch/issues/14
    # weights = 1 / np.log1p(class_freq)
    # weights = num_classes * weights / np.sum(weights)
    weights = class_freq / np.sum(class_freq)

    return weights


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir',
                        nargs='+',
                        type=str,
                        help='training set directory')
    parser.add_argument('--num-classes',
                        type=int,
                        help='number of classes(include background)')
    parser.add_argument('--reduce-zero-labels',
                        action='store_true',
                        default=False,
                        help='reduce zero labels')

    args = parser.parse_args()

    class_weights = calculate_class_weights(args.gt_dir, args.num_classes,
                                            args.reduce_zero_labels)
    weights_str = ', '.join(f'{v:.4f}' for v in class_weights)
    print(f'class_weights: \n[{weights_str}]')


if __name__ == '__main__':
    main()

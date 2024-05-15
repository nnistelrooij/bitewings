import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils


def main(in_dir: Path, coco_path: Path):
    coco = COCO(coco_path)

    images = list(coco.imgs.items())
    images = sorted(images, key=lambda img: img[1]['file_name'])
    for img_id, img_dict in images:
        print(img_dict['file_name'])
        img = cv2.imread(str(in_dir / img_dict['file_name']))

        anns = coco.imgToAnns[img_id]
        for ann in anns:
            mask = maskUtils.decode(ann['segmentation'])
            mask = np.concatenate((np.zeros_like(mask), mask), axis=1)
            rle = maskUtils.encode(np.asfortranarray(mask))
            ann['segmentation'] = {
                'size': rle['size'],
                'counts': rle['counts'].decode(),
            }
            ann['bbox'][0] += img_dict['width']


        _, ax = plt.subplots(figsize=(12, 6))

        side_by_side = np.concatenate((img, img), axis=1)
        
        ax.imshow(side_by_side)
        ax.set_title(img_dict['file_name'])
        coco.showAnns(anns, draw_bbox=False)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_dir', default='.', type=Path,
        help='Path to folder with images for prediction.',
    )
    parser.add_argument(
        'method', choices=['hierarchical', 'maskdino', 'maskrcnn', 'sparseinst'],
        help='Method that has done predictions to convert to COCO format.',
    )
    args = parser.parse_args()

    main(
        args.in_dir,
        Path(f'work_dirs/chart_filing_{args.method}/pred.json'),
    )

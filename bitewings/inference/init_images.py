import argparse
import json
from pathlib import Path
from typing import List

import cv2


def main(
    in_dir: Path,
    img_suffixes: List[str]=[
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    ],
):
    out_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    idx = 0
    for file_path in in_dir.glob('*'):
        if file_path.suffix not in img_suffixes:
            continue

        img = cv2.imread(str(file_path))
        img_dict = {
            'file_name': file_path.name,
            'width': img.shape[1],
            'height': img.shape[0],
            'id': idx + 1,
        }

        out_dict['images'].append(img_dict)
        idx += 1

    with open(in_dir / 'coco.json', 'w') as f:
        json.dump(out_dict, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_dir', default='.', type=Path,
        help='Path to folder with images for prediction.',
    )
    args = parser.parse_args()

    main(args.in_dir)

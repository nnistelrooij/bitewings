import json
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

import os, sys
sys.path.append(os.getcwd())

from bitewings.preprocess.add_fdis import (
    add_categories_with_fdi,
    determine_tooth_matches,
)
from bitewings.preprocess.hierarchy_to_flat import process_image
from bitewings.preprocess.split_images import split


if __name__ == '__main__':
    root = Path('../data/Netherlands')

    # Match tooth finding to corresponding FDI number
    coco = COCO(root / 'annotations.json')
    coco = add_categories_with_fdi(coco)

    out_anns = []
    for img_id in tqdm(list(coco.imgs.keys())):
        anns = determine_tooth_matches(coco, coco.imgToAnns[img_id])
        out_anns.extend(anns)

    coco.dataset['annotations'] = out_anns
    with open(root / 'annotations_fdi.json', 'w') as f:
        json.dump(coco.dataset, f, indent=2)

    # go back to flat representation for comparative models
    coco = COCO(root / 'annotations_fdi.json')    
    classes = [
        'tooth_11', 'tooth_12', 'tooth_13', 'tooth_14', 'tooth_15', 'tooth_16', 'tooth_17', 'tooth_18',
        'tooth_21', 'tooth_22', 'tooth_23', 'tooth_24', 'tooth_25', 'tooth_26', 'tooth_27', 'tooth_28',
        'tooth_31', 'tooth_32', 'tooth_33', 'tooth_34', 'tooth_35', 'tooth_36', 'tooth_37', 'tooth_38',
        'tooth_41', 'tooth_42', 'tooth_43', 'tooth_44', 'tooth_45', 'tooth_46', 'tooth_47', 'tooth_48',
        'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus_left', 'calculus_right',
    ]

    out_dict = {
        'images': coco.dataset['images'],
        'categories': [{'id': i, 'name': label} for i, label in enumerate(classes, 1)],
        'annotations': [],
    }
    for img_dict in tqdm(list(coco.imgs.values())):
        out_dict = process_image(coco, img_dict, out_dict)

    with open(root / 'annotations_flat.json', 'w') as f:
        json.dump(out_dict, f, indent=2)

    # split images with stratification for k-fold cross-validation
    coco = COCO(root / 'annotations_fdi.json')
    split(root, coco, 'fdi', n_splits=5)

    coco = COCO(root / 'annotations_flat.json')
    split(root, coco, 'flat', n_splits=5)

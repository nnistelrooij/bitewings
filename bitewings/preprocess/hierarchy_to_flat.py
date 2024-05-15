import copy
import logging

import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from scipy import ndimage

from bitewings.preprocess.add_fdis import coco_to_rle


def split_segments(
    img_dict: dict,
    tooth_ann: dict,
    finding_ann: dict,
    min_pixels: int=32,
):
    rle = coco_to_rle(finding_ann['segmentation'], img_dict['height'], img_dict['width'])
    mask = maskUtils.decode(rle)
    labels, max_label = ndimage.label(mask)
    counts = ndimage.sum_labels(np.ones_like(labels), labels, range(1, max_label + 1))
    if counts.sum() < min_pixels:
        logging.debug(' '.join(['<', str(min_pixels), img_dict['file_name'], str(counts.sum())]))
        return []

    tooth_x = tooth_ann['bbox'][0] + tooth_ann['bbox'][2] / 2
    if max_label == 1:
        finding_x = finding_ann['bbox'][0] + finding_ann['bbox'][2] / 2

        return [{
            'direction': 'right' if finding_x > tooth_x else 'left', 
            'segmentation': {
                'size': rle['size'],
                'counts': rle['counts'].decode() if isinstance(rle['counts'], bytes) else rle['counts'],
            },
        }]
    
    out_segments = []
    for label in range(1, max_label + 1):
        mask = labels == label
        if mask.sum() < min_pixels:
            logging.debug(' '.join(['<', str(min_pixels), img_dict['file_name'], str(counts.sum())]))
            continue
  
        finding_x = np.nonzero(mask)[1].mean()
        rle = maskUtils.encode(np.asfortranarray(mask))
        segment = {
            'direction': 'right' if finding_x > tooth_x else 'left',
            'segmentation': {
                'size': rle['size'],
                'counts': rle['counts'].decode() if isinstance(rle['counts'], bytes) else rle['counts'],
            },
        }
        out_segments.append(segment)

    return out_segments
            

def process_image(coco: COCO, img_dict: dict, out_dict: dict):
    cat_id2name = {cat_id: cat['name'] for cat_id, cat in coco.cats.items()}
    ann_name2id = {cat['name']: cat['id'] for cat in out_dict['categories']}

    tooth_fdi2ann = {
        cat_id2name[ann['category_id']][-2:]: ann 
        for ann in coco.imgToAnns[img_dict['id']]
        if 'tooth' in cat_id2name[ann['category_id']].lower()
    }
    
    for finding_ann in coco.imgToAnns[img_dict['id']]:
        cat_name = cat_id2name[finding_ann['category_id']].lower()
        if 'tooth' in cat_name:
            ann = copy.deepcopy(finding_ann)
            ann['id'] = len(out_dict['annotations']) + 1
            ann['category_id'] = ann_name2id[cat_name]
            ann['iscrowd'] = 0
            out_dict['annotations'].append(ann)
            continue

        fdi = cat_name[-2:]
        ann_segments = split_segments(img_dict, tooth_fdi2ann[fdi], finding_ann)
        for segment in ann_segments:
            ann_name = cat_name[:-3]
            if ann_name == 'calculus':
                ann_name = f'calculus_{segment["direction"]}'

            ann = copy.deepcopy(finding_ann)
            ann['id'] = len(out_dict['annotations']) + 1
            ann['category_id'] = ann_name2id[ann_name]
            ann['segmentation'] = segment['segmentation']
            ann['area'] = maskUtils.area(segment['segmentation']).item()
            ann['bbox'] = maskUtils.toBbox(segment['segmentation']).tolist()
            ann['iscrowd'] = 0
            out_dict['annotations'].append(ann)

    return out_dict

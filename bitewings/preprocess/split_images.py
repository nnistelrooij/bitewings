import json
from pathlib import Path
import shutil
from typing import List, Optional

import numpy as np
from pycocotools.coco import COCO
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
from tqdm import tqdm


def copy_annotations(
    coco: COCO,
    files: List[Path],
    split,
):
    train_idxs, val_idxs = split
    train_stems = np.array([Path(files[idx]).stem for idx in train_idxs])
    val_stems = np.array([Path(files[idx]).stem for idx in val_idxs])

    id2stem = {img['id']: Path(img['file_name']).stem for img in coco.dataset['images']}
    img_stems = np.array(list(id2stem.values()))
    ann_stems = np.array([id2stem[ann['image_id']] for ann in coco.dataset['annotations']])

    train_coco_dict = {'categories': coco.dataset['categories'], 'annotations': []}
    val_coco_dict = {'categories': coco.dataset['categories'], 'annotations': []}

    mask = np.any(img_stems[np.newaxis] == train_stems[:, np.newaxis], axis=0)
    train_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]
    mask = np.any(img_stems[np.newaxis] == val_stems[:, np.newaxis], axis=0)
    val_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]

    train_stems = set(train_stems.tolist())
    val_stems = set(val_stems.tolist())
    for ann, ann_stem in zip(coco.dataset['annotations'], ann_stems):
        if ann_stem in train_stems:
            train_coco_dict['annotations'].append(ann)
        elif ann_stem in val_stems:
            val_coco_dict['annotations'].append(ann)

    return train_coco_dict, val_coco_dict


def determine_one_hots(
    coco: COCO,
    files: List[str],
):
    filename2id = {img['file_name']: img['id'] for img in coco.dataset['images']}
    file2cats = []
    cats = set()
    for file in tqdm(files):
        file_cats = []
        for ann in coco.imgToAnns[filename2id[file]]:
            cat_name = coco.cats[ann['category_id']]['name'].lower()
            cat_name = cat_name[:-3] if cat_name[-1] in '12345678' else cat_name
            cats.add(cat_name)
            file_cats.append(cat_name)

        file2cats.append(file_cats)

    cats = sorted(list(cats))
    one_hots = np.zeros((len(files), len(cats)), dtype=int)
    for i, file_cats in enumerate(file2cats):
        for cat in file_cats:
            one_hots[i, cats.index(cat)] = 1
            
    return one_hots


def test_split(
    out_dir: Path,
    files: List[Path],
    one_hots: np.ndarray,
    test_path: Path,
):
    coco = COCO(test_path)
    img_paths = set([img['file_name'] for img in coco.dataset['images']])

    split = [[], []]
    for i, file in enumerate(files):
        if file in img_paths:
            split[1].append(i)
        else:
            split[0].append(i)

    _, test_coco = copy_annotations(coco, files, split)

    with open(out_dir / 'test.json', 'w') as f:
        json.dump(test_coco, f, indent=2)

    trainval_idxs = np.array(split[0])
    return [files[idx] for idx in trainval_idxs], one_hots[trainval_idxs]


def kfold_split(
    out_dir: Path,
    coco: COCO,
    name: str,
    n_splits: int,
    files: List[Path],
    one_hots: np.ndarray,
):
    splits = [None]
        
    if n_splits > 1:
        splitter = IterativeStratification(n_splits, order=2, shuffle=True, random_state=1234)
        splits += list(splitter.split(one_hots, one_hots))

    for i, split in enumerate(splits):
        if split is None:
            with open(out_dir / f'trainval_{name}.json', 'w') as f:
                json.dump(coco.dataset, f, indent=2)
            continue

        train_coco, val_coco = copy_annotations(coco, files, split)

        with open(out_dir / f'train_{name}_{i}.json', 'w') as f:
            json.dump(train_coco, f, indent=2)

        with open(out_dir / f'val_{name}_{i}.json', 'w') as f:
            json.dump(val_coco, f, indent=2) 


def split(
    root: Path,
    coco: COCO,
    name: str,
    n_splits: int,
    test_path: Optional[Path]=None,
):
    files = sorted([img['file_name'] for img in coco.dataset['images']])
    one_hots = determine_one_hots(coco, files)

    out_dir = root / 'splits'
    shutil.rmtree(root / 'splits', ignore_errors=True)
    out_dir.mkdir(exist_ok=True)

    if test_path is not None:
        files, one_hots = test_split(
            out_dir, files, one_hots, test_path,
        )

    kfold_split(out_dir, coco, name, n_splits, files, one_hots)

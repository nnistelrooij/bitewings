import copy
import json
from pathlib import Path
import shutil
from typing import List

import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import KFold


def combine_train_val(root: Path, out_dir: Path):
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)

    out_dict = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    for split in ['train', 'val']:
        split_dir = root / split
        for img_path in (split_dir / 'images').glob('*'):
            shutil.copy(img_path, out_dir / 'images' / img_path.name)

        with open(split_dir / f'{split}.json', 'r') as f:
            coco_dict = json.load(f)

        out_dict['images'].extend(coco_dict['images'])
        out_dict['annotations'].extend(coco_dict['annotations'])
        out_dict['categories'] = coco_dict['categories']

    with open(out_dir / 'odonto.json', 'w') as f:
        json.dump(out_dict, f, indent=2)

    return out_dir / 'odonto.json'


def is_bitewing_suitable(
    coco: COCO,
    catid2name: dict,
    img_id: int,
):
    anns = coco.imgToAnns[img_id]
    fdis = np.array([int(catid2name[ann['category_id']][-2:]) for ann in anns])

    if np.any(fdis > 50):
        return False

    fdis = fdis[fdis > 10]
    fdis = fdis[(fdis % 10) >= 3]
    _, counts = np.unique(fdis // 10, return_counts=True)

    if counts.shape[0] != 4:
        return False

    return counts.min() >= 3


def select_bitewings(coco: COCO, out_dir: Path):
    catid2name = {cat['id']: cat['name'] for cat in coco.cats.values()}

    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories'],
    }
    for i, img_id in enumerate(coco.imgs):        
        if not is_bitewing_suitable(coco, catid2name, img_id):
            continue

        img_dict = copy.deepcopy(coco.imgs[img_id])
        coco_dict['images'].append(img_dict)
        coco_dict['annotations'].extend(coco.imgToAnns[img_id])

    with open(out_dir / 'odonto_bitewings.json', 'w') as f:
        json.dump(coco_dict, f, indent=2)

    return out_dir / 'odonto_bitewings.json'


def copy_annotations(
    coco: COCO,
    files: List[Path],
    split,
):
    if split is None:
        train_idxs = []
        val_idxs = range(len(files))
    else:
        train_idxs, val_idxs = split
    train_stems = np.array([Path(files[idx]).stem for idx in train_idxs])
    val_stems = np.array([Path(files[idx]).stem for idx in val_idxs])

    id2stem = {img['id']: Path(img['file_name']).stem for img in coco.dataset['images']}
    img_stems = np.array(list(id2stem.values()))
    ann_stems = np.array([id2stem[ann['image_id']] for ann in coco.dataset['annotations']])

    train_coco_dict = {
        'categories': coco.dataset['categories'],
        'tag_categories': coco.dataset['tag_categories'],
    }
    val_coco_dict = copy.deepcopy(train_coco_dict)

    mask = np.any(img_stems[np.newaxis] == train_stems[:, np.newaxis], axis=0)
    train_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]
    mask = np.any(img_stems[np.newaxis] == val_stems[:, np.newaxis], axis=0)
    val_coco_dict['images'] = [img for img, b in zip(coco.dataset['images'], mask) if b]

    mask = np.any(ann_stems[np.newaxis] == train_stems[:, np.newaxis], axis=0)
    train_coco_dict['annotations'] = [img for img, b in zip(coco.dataset['annotations'], mask) if b]
    mask = np.any(ann_stems[np.newaxis] == val_stems[:, np.newaxis], axis=0)
    val_coco_dict['annotations'] = [img for img, b in zip(coco.dataset['annotations'], mask) if b]

    return train_coco_dict, val_coco_dict


def test_split(
    ann_dir: Path,
    files: List[Path],
    tags: List[List[str]],
    test_size: float,
    name: str,
    keep_tags: List[str]=['Bitewing'],
):
    keep_mask = np.ones(len(files), dtype=bool)
    for i, file_tags in enumerate(tags):
        if not np.all(np.any(np.array(file_tags)[None] == np.array(keep_tags)[:, None], axis=1)):
            keep_mask[i] = False
            continue

    keep_files = [file for file, b in zip(files, keep_mask) if b]

    splitter = KFold(
        n_splits=int(1 / test_size),
        shuffle=True,
        random_state=1234,
    )
    split = next(splitter.split(keep_files, keep_files))
    _, test_coco = copy_annotations(coco, keep_files, split)

    with open(ann_dir / f'test_{name}.json', 'w') as f:
        json.dump(test_coco, f, indent=2)

    files = [
        *[file for file, b in zip(files, keep_mask) if not b],
        *[keep_files[idx] for idx in split[0]],
    ]

    return files


def kfold_split(
    root: Path,
    coco: COCO,
    n_splits: int,
    files: List[Path],
    name: str,
):
    splits = [None]
    if n_splits > 1:
        splitter = KFold(
            n_splits, shuffle=True, random_state=1234,
        )
        splits.extend(list(splitter.split(files, files)))

    out_dir = root / 'splits'
    out_dir.mkdir(exist_ok=True)

    for i, split in enumerate(splits):
        train_coco, val_coco = copy_annotations(coco, files, split)

        if split is None:
            with open(out_dir / f'trainval_{name}.json', 'w') as f:
                json.dump(val_coco, f, indent=2)
            continue

        with open(out_dir / f'train_{name}_{i}.json', 'w') as f:
            json.dump(train_coco, f, indent=2)

        with open(out_dir / f'val_{name}_{i}.json', 'w') as f:
            json.dump(val_coco, f, indent=2) 


def split(
    root: Path,
    coco: COCO,
    n_splits: int,
    test_size: float=0.0,
    name: str='',
):
    files = sorted([img['file_name'] for img in coco.dataset['images']])
    filename2id = {img['file_name']: img['id'] for img in coco.dataset['images']}
    tagid2name = {tag['id']: tag['name'] for tag in coco.dataset['tag_categories']}
    tags = [[tagid2name[tag_id] for tag_id in coco.imgs[filename2id[file]]['tag_ids']] for file in files]

    if test_size > 0:
        files = test_split(
            root, files, tags, test_size, name,
        )

    kfold_split(root, coco, n_splits, files, name)


if __name__ == '__main__':
    root = Path('odontoai')
    out_dir = Path('data/odonto')

    coco_path = combine_train_val(root, out_dir)
    coco = COCO(coco_path)
    coco_path = select_bitewings(coco, out_dir)

    coco = COCO(root / 'odonto_bitewings.json')
    split(
        root,
        coco,
        n_splits=5,
        test_size=0.0,
        name='odonto_enumeration',
    )

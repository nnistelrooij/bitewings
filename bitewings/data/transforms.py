import numpy as np

import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import RandomFlip
from mmdet.structures.bbox import autocast_box_type

@TRANSFORMS.register_module()
class RandomBitewingFlip(RandomFlip):

    label_map = np.array([
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,  # 1st-2nd quadrant
        24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,  # 3rd-4th quadrant
        32, 33, 34, 35, 36, 37,  # findings
        39, 38,  # calculus
    ])

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, labels, and semantic segmentation."""
        labels = results['gt_bboxes_labels']

        results['gt_bboxes_labels'] = self.label_map[labels]

        super()._flip(results)


@TRANSFORMS.register_module()
class CLAHETransform(BaseTransform):

    def __init__(self, clip_limit: float=2.0, tile_grid_size: tuple[int, int]=(32, 32)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def transform(self, results: dict) -> dict:
        grayscale_img = results['img'][..., 0]

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        clahe_img = clahe.apply(grayscale_img)

        results['img'] = np.tile(clahe_img[..., None], (1, 1, 3))

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '
        repr_str += f'tile_grid_size={self.tile_grid_size})'

        return repr_str

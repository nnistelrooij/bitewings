from typing import Tuple, Union

from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import RandomCrop
import numpy as np


@TRANSFORMS.register_module()
class CropBitewing(RandomCrop):

    def __init__(
        self,
        height_range: Tuple[int, int]=(0.75, 1),
        aspect_ratio_range: Tuple[int, int]=(1, 1.33),
        offset_sigma: float=0.05,
    ):
        super().__init__(crop_size=(1, 1))

        self.height_range = height_range
        self.aspect_ratio_range = aspect_ratio_range
        self.offset_sigma = offset_sigma

    def _rand_bitewing_teeth(self, results: dict):
        crop_right = np.random.rand() < 0.5

        bboxes = []
        for instance in results['instances']:
            label = instance['bbox_label']

            instance_right = (8 <= label) & (label < 24)
            if crop_right != instance_right:
                continue

            instance_anterior = (label % 8) <= 1
            if instance_anterior:
                continue

            bboxes.append(np.concatenate((
                instance['bbox'], [instance['bbox_label']],
            )))

        return np.stack(bboxes)
    
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        max_tooth_height = np.max(self.bboxes[:, 3] - self.bboxes[:, 1])
        self.height = max_tooth_height * np.random.uniform(*self.height_range)
        self.width = self.height * np.random.uniform(*self.aspect_ratio_range)

        return int(self.height), int(self.width)

    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        center_x = (
            np.max(self.bboxes[:, 2]) + 
            np.min(self.bboxes[:, 0])
        ) / 2

        maxilla_mask = self.bboxes[:, -1] < 16
        center_y = (
            np.max(self.bboxes[maxilla_mask, 3]) + 
            np.min(self.bboxes[~maxilla_mask, 1])
        ) / 2

        max_tooth_height = np.max(self.bboxes[:, 3] - self.bboxes[:, 1])
        offset_x = max_tooth_height * np.random.normal(scale=self.offset_sigma)
        offset_y = max_tooth_height * np.random.normal(scale=self.offset_sigma)

        margin_x = center_x + offset_x - self.width / 2
        margin_y = center_y + offset_y - self.height / 2

        return int(margin_y), int(margin_x)

    def transform(self, results: dict) -> Union[dict, None]:
        if 'tags' in results and 'OPG' not in results['tags']:
            return results

        self.bboxes = self._rand_bitewing_teeth(results)
        
        return super().transform(results)

    def __repr__(self) -> str:
        repr_str = super().__repr__()[:-1]
        repr_str += f', height_range={self.height_range}'
        repr_str += f', aspect_ratio_range={self.aspect_ratio_range})'

        return repr_str

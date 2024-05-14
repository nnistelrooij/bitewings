# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results

from mmdet.evaluation import CocoMetric
from projects.DENTEX.evaluation import CocoMulticlassMetric


@METRICS.register_module()
class RelevantCocoMetric(CocoMetric):

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                    outfile_prefix: str) -> str:
        remove_labels = torch.tensor([0, 1, 8, 9, 16, 17, 24, 25])
        for gt_dict in gt_dicts:
            labels = gt_dict['anns']['labels']
            rles = encode_mask_results(gt_dict['anns']['masks'])
            widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in rles])

            keep_mask = (
                (labels < 32)
                & (widths >= gt_dict['width'] * 0.1)
                & (~torch.any(labels[:, None] == remove_labels, axis=1))
            )

            annotations = []
            for i in range(labels.shape[0]):
                if not keep_mask[i]:
                    continue

                ann = {
                    'bbox_label': labels[i].item() % 8,
                    'bbox': gt_dict['anns']['bboxes'][i].numpy().tolist(),
                    'mask': rles[i],
                }
                annotations.append(ann)

            gt_dict['anns'] = annotations

        return super().gt_to_coco_json(gt_dicts, outfile_prefix)

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        remove_labels = np.array([0, 1, 8, 9, 16, 17, 24, 25])
        for result in results:
            labels = result['labels']
            rles = result['masks']
            widths = np.array([maskUtils.toBbox(rle)[2] for rle in rles])

            keep = (
                (labels < 32)
                & (~np.any(labels[:, None] == remove_labels, axis=1))
                & (widths >= rles[0]['size'][1] * 0.1)
            )

            result['labels'] = result['labels'][keep] % 8
            result['scores'] = result['scores'][keep]
            result['bboxes'] = result['bboxes'][keep]
            result['masks'] = [result['masks'][i] for i in np.nonzero(keep)[0]]

        return super().results2json(results, outfile_prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            data_sample['instances'] = data_sample['gt_instances']

        super().process(data_batch, data_samples)



@METRICS.register_module()
class CocoRelevantMulticlassMetric(CocoMulticlassMetric):

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                    outfile_prefix: str) -> str:
        remove_labels = torch.tensor([0, 1, 8, 9, 16, 17, 24, 25])
        for gt_dict in gt_dicts:
            labels = torch.tensor([ann['bbox_label'] for ann in gt_dict['anns']])
            rles = [ann['mask'] for ann in gt_dict['anns']]
            widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in rles])

            keep_mask = (
                (widths >= gt_dict['width'] * 0.1)
                & (~torch.any(labels[:, None] == remove_labels, axis=1))
            )
            gt_dict['anns'] = [gt_dict['anns'][i] for i in torch.nonzero(keep_mask)[:, 0]]
            # for ann in gt_dict['anns']:
            #     ann['bbox_label'] = ann['bbox_label'] % 8

        return super().gt_to_coco_json(gt_dicts, outfile_prefix)

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        remove_labels = np.array([0, 1, 8, 9, 16, 17, 24, 25])
        for result in results:
            labels = result['labels']
            rles = result['masks']
            widths = np.array([maskUtils.toBbox(rle)[2] for rle in rles])

            keep = (
                (~np.any(labels[:, None] == remove_labels, axis=1))
                & (widths >= rles[0]['size'][1] * 0.1)
            )

            result['labels'] = result['labels'][keep]
            result['scores'] = result['scores'][keep]
            result['bboxes'] = result['bboxes'][keep]
            result['masks'] = [result['masks'][i] for i in np.nonzero(keep)[0]]

        return super().results2json(results, outfile_prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            data_sample['instances'] = data_sample['gt_instances']

        super().process(data_batch, data_samples)

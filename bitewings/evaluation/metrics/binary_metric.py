from collections import defaultdict
from typing import List, Sequence

import pycocotools.mask as maskUtils
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
import numpy as np
import torch

from projects.DENTEX.evaluation import SingleLabelMetric


@METRICS.register_module()
class SingleLabelInstanceMetric(SingleLabelMetric):
    
    FDIs = [
        '11', '12', '13', '14', '15', '16', '17', '18',
        '21', '22', '23', '24', '25', '26', '27', '28',
        '31', '32', '33', '34', '35', '36', '37', '38',
        '41', '42', '43', '44', '45', '46', '47', '48',
    ]

    def match_tooth_findings(self, instances):
        if not isinstance(instances['masks'], torch.Tensor):
            instances['masks'] = torch.from_numpy(instances['masks'].masks)
            instances['labels'] = instances['labels'].cpu()

        tooth_idxs = torch.nonzero(instances['labels'] < 32)[:, 0]
        tooth_masks = instances['masks'][tooth_idxs].cpu().numpy()
        tooth_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in tooth_masks]
        tooth_areas = tooth_masks.sum(axis=(1, 2))

        results = [{
            'fdi': self.FDIs[instances['labels'][idx]],
            'bbox': instances['bboxes'][idx],
            'mask': rle,
            'label': [0],
            'score': [1.0, 0.0],
        } for idx, rle in zip(tooth_idxs, tooth_rles)]

        if self.idx == 38:  # calculus
            finding_idxs = torch.nonzero(instances['labels'] >= 38)[:, 0]
            if finding_idxs.shape[0] == 0:
                return results

            finding_masks = instances['masks'][finding_idxs].cpu().numpy()
            finding_masks = np.stack([
                np.roll(mask, -50 + 100 * (label == 38), axis=1)
                for mask, label in zip(finding_masks, instances['labels'][finding_idxs].cpu())
            ])
        else:
            finding_idxs = torch.nonzero(instances['labels'] == self.idx)[:, 0]
            finding_masks = instances['masks'][finding_idxs].cpu().numpy()

        finding_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in finding_masks]
        finding_areas = finding_masks.sum(axis=(1, 2))

        if len(finding_rles) == 0:
            return results

        ious = maskUtils.iou(tooth_rles, finding_rles, [0]*len(finding_rles))
        overlaps = ious / (finding_areas / tooth_areas[:, None])

        for i, result in enumerate(results):
            if not np.any(overlaps[i] >= 0.5):
                continue

            result['label'] = [1]
            if 'scores' in instances:
                scores = instances['scores'][finding_idxs[overlaps[i] >= 0.5]]
                result['score'] = [1 - scores.max(), scores.max()]

        return results
    
    def match_teeth(self, gt_teeth, pred_teeth):
        if self.remove_irrelevant:
            gt_teeth = [
                tooth for tooth in gt_teeth if tooth['fdi'][1] not in '12'
                and (tooth['bbox'][2] - tooth['bbox'][0]) > tooth['mask']['size'][1] * 0.05
            ]
        
        gt_rles = [tooth['mask'] for tooth in gt_teeth]
        pred_rles = [tooth['mask'] for tooth in pred_teeth]

        ious = maskUtils.iou(gt_rles, pred_rles, [0]*len(pred_rles))
        results = []
        for gt_idx, pred_idx in zip(*np.nonzero(ious > 0.5)):
            result = {
                'gt_label': torch.tensor(gt_teeth[gt_idx]['label']),
                'pred_label': torch.tensor(pred_teeth[pred_idx]['label']),
                'pred_score': torch.tensor(pred_teeth[pred_idx]['score']),
                'num_classes': 2,
            }
            results.append(result)

        return results

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            pred_teeth = self.match_tooth_findings(pred)
            gt_teeth = self.match_tooth_findings(gt)

            results = self.match_teeth(gt_teeth, pred_teeth)
            self.results.extend(results)


@METRICS.register_module()
class AggregateLabelInstanceMetric(BaseMetric):

    def __init__(
        self,
        label_idxs: List[int],
        prefixes: List[str],
        remove_irrelevant: bool=False,
        **kwargs,
    ):
        super().__init__(prefix='aggregate')

        self.single_metrics = [
            SingleLabelInstanceMetric(label_idx, remove_irrelevant=remove_irrelevant, prefix=prefix) for
            label_idx, prefix in zip(label_idxs, prefixes)
        ]

    def process(self, data_batch, data_samples: Sequence[dict]):
        for metric in self.single_metrics:
            metric.process(data_batch, data_samples)

    def compute_metrics(self, results: List):
        metrics = {}
        agg_metrics = defaultdict(list)
        for metric in self.single_metrics:
            metric_dict = metric.compute_metrics(metric.results)

            metrics.update({f'{metric.prefix}/{k}': v for k, v in metric_dict.items()})

            for k, v in metric_dict.items():
                agg_metrics[k] = agg_metrics[k] + [v]

        agg_metrics = {f'aggregate/{k}': np.mean(v) for k, v in agg_metrics.items()}
        metrics.update(agg_metrics)
        
        return metrics

_base_ = '../../mmdetection/projects/DENTEX/configs/maskdino_r50_coco_multilabel.py'
# _base_ = './maskdino_swin-l_coco_multilabel.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX.datasets',
        'projects.DENTEX.datasets.dataset_wrappers',
        'projects.DENTEX.datasets.samplers',
        'projects.DENTEX.datasets.transforms.loading',
        'projects.DENTEX.datasets.transforms.formatting',
        'projects.DENTEX.datasets.transforms.transforms',
        'projects.DENTEX.evaluation',
        'projects.DENTEX.maskdino',
        'projects.DENTEX.visualization',
        'projects.DENTEX.hooks',
        'bitewings.data.transforms',
        'bitewings.evaluation.metrics',
    ],
    allow_failed_imports=False,
)

synmedico_root = '/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_train/'
nijmegen_root = '/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_test/'
ai_dental_root = '/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_extra/'
mihai_root = '/home/mkaailab/Documents/Xray/Mihai/'
split = 'lingyun_bitewings'
fold = 1
work_dir = f'work_dirs/lingyun_trainval_hierarchical/'
phase = 'train'
merge_layers = True
share_mlp = True

classes = [
    'tooth',
]

attributes = [
    'pontic', 'implants', 'crowns', 'fillings', 'roots', 'caries', 'calculus'
]
attributes = [
    'implants', 'crowns', 'pontic', 'fillings', 'roots', 'caries', 'calculus'
]


num_classes = 32, len(classes)
num_attributes = 1 + len(attributes)
num_upper_masks = 1 + len(set(classes) & set(attributes))
train_pipeline = [
    dict(type='RandomOPGFlip', prob=0.5),
    # dict(type='CLAHETransform'),
    dict(
        type='RandomChoice',
        transforms=[[{
                'type':
                'RandomChoiceResize',
                'scales': [(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
            [{
                'type': 'RandomChoiceResize',
                'scales': [(400, 4200), (500, 4200), (600, 4200)],
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                'RandomChoiceResize',
                'scales':
                [(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
        ]),
    dict(type='PackMultilabelDetInputs'),
]

num_workers = 0
train_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(dataset=dict(
        pipeline=train_pipeline,
        dataset=dict(_delete_=True, type='ConcatDataset', datasets=[
            dict(
                type='CocoMulticlassDataset',
                strict=True,
                decode_masks=False,
                ann_file=synmedico_root + f'trainval_synmedico.json',
                data_prefix=dict(img=synmedico_root + 'images'),
                data_root=synmedico_root,
                metainfo=dict(classes=classes, attributes=attributes),
                merge_layers=merge_layers,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMulticlassAnnotations', merge_layers=merge_layers, with_bbox=True, with_mask=True),
                ],
            ),
            dict(
                type='CocoMulticlassDataset',
                strict=True,
                decode_masks=False,
                ann_file=nijmegen_root + f'trainval_nijmegen.json',
                data_prefix=dict(img=nijmegen_root + 'images'),
                data_root=nijmegen_root,
                metainfo=dict(classes=classes, attributes=attributes),
                merge_layers=merge_layers,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadMulticlassAnnotations', merge_layers=merge_layers, with_bbox=True, with_mask=True),
                ],
            ),
        ],
    ))),
)

val_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),
    # dict(type='CLAHETransform'),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True),
    dict(type='LoadMulticlassAnnotations', merge_layers=merge_layers, with_bbox=True, with_mask=True),
    dict(
        type='PackMultilabelDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]
val_dataloader = dict(
    num_workers=num_workers,
    persistent_workers=False,
    dataset=dict(
        type='CocoMulticlassDataset',
        strict=True,
        decode_masks=False,
        # ann_file=ai_dental_root + f'val_ai-dental_1.json',
        ann_file='/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_diff2/diffs.json',
        data_prefix=dict(img=ai_dental_root + 'images'),
        data_root=ai_dental_root,
        metainfo=dict(classes=classes, attributes=attributes),
        merge_layers=merge_layers,
        pipeline=val_pipeline,
    ),
)
val_evaluator = [
    dict(
        type='CocoMulticlassMetric',
        metric=['bbox', 'segm'],
        class_agnostic=False,
        classwise=True,
        prefix='fdi_label',
    ),
    dict(
        type='AggregateLabelMetric',
        label_idxs=range(1, len(classes + attributes)),
        prefixes=(classes + attributes)[1:]
    ),
]

# val_dataloader = None
# val_evaluator = None
# val_cfg = None

test_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type='CocoMulticlassDataset',
        strict=True,
        num_workers=0,
        decode_masks=False,
        # ann_file=data_root + f'val_{split}_{fold}.json',
        ann_file='/home/mkaailab/.darwin/datasets/mucoaid/bitewingchartfiling_diff2/diffs.json',
        # ann_file=mihai_root + 'coco.json',
        data_prefix=dict(img=ai_dental_root + 'images'),
        # data_prefix=dict(img=mihai_root),
        data_root=ai_dental_root,
        # data_root=mihai_root,
        metainfo=dict(classes=classes, attributes=attributes),
        pipeline=val_pipeline,
    ),
)
test_evaluator = [
    # dict(
    #     type='CocoMulticlassMetric',
    #     metric=['bbox', 'segm'],
    #     class_agnostic=True,
    #     prefix='class_agnostic',
    # ),
    dict(
        type='CocoRelevantMulticlassMetric',
        metric=['bbox', 'segm'],
        class_agnostic=False,
        prefix='fdi_label',
        classwise=True,
    ),
    # dict(
    #     type='AggregateLabelMetric',
    #     label_idxs=range(1, len(classes + attributes)),
    #     prefixes=(classes + attributes)[1:]
    # ),
    dict(
        type='DumpMulticlassDetResults',
        score_thr=0.0,
        out_file_path=work_dir + 'detection.pkl',
    ),
]

custom_hooks = []
model = dict(
    train_cfg=dict(num_classes=num_classes[0], hnm_samples=2, use_fed_loss=False),
    test_cfg=dict(
        instance_postprocess_cfg=dict(max_per_image=75),
        max_per_image=75,
    ),
    panoptic_head=dict(
        num_things_classes=num_classes[0],
        num_stuff_classes=0,
        decoder=dict(
            num_classes=num_classes,
            num_attributes=num_attributes,
            enable_multilabel=not merge_layers,
            enable_multiclass=True,
            share_mlp=share_mlp,
            num_queries=80 if phase == 'train' else 100,
        ),
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes[0],
        num_stuff_classes=0,
        enable_multilabel=not merge_layers,
        enable_multiclass=True,
        num_upper_masks=num_upper_masks,
    ),
)

max_epochs = 36
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
param_scheduler = [dict(
    type='MultiStepLR',
    begin=0,
    end=max_epochs,
    by_epoch=True,
    milestones=[27, 33],
    gamma=0.1,
)]
optim_wrapper = dict(optimizer=dict(lr=1e-4))

tta_model = dict(
    type='DENTEXTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

default_hooks = dict(
    checkpoint=dict(save_best='aggregate/aggregate/f1-score'),
    visualization=dict(draw=False, show=False),
)

visualizer = dict(type='MulticlassDetLocalVisualizer')

load_from = 'work_dirs/odonto_bitewings_enumeration/epoch_50.pth'
# load_from = 'work_dirs/lingyun_trainval_hierarchical/epoch_27 (copy).pth'

# Automated bitewing chart filing

Comprehensive bitewing chart filing using hierarchical instance segmentation.

This code is implemented using MMDetection (v.3.0.0) based on PyTorch 2.0.1.


## Installation

See `INSTALL.md` for installation instructions.


## Data and checkpoints

The panoramic radiographs from the OdontoAI platform can be requested from [the platform's website](https://odontoai.com/). Furthermore, model checkpoints and the data collected for the study can be requested from Niels van Nistelrooij (Niels dot vanNistelrooij at radboudumc dot nl).


## Inference

To run the model on your own bitewings, first make an empty COCO file by running `bitewings/inference/init_images.py`. The model can be run on these images using this command:

```bash
export IN_DIR=`realpath <path>`
PYTHONPATH=. python \
  mmdetection/tools/test.py \
  bitewings/configs/config_<model>.py \
  ../checkpoints/<model>_chartfiling.pth \
  --cfg-options \
    test_dataloader.dataset.data_root=$IN_DIR \
    test_dataloader.dataset.data_prefix.img=$IN_DIR \
    test_dataloader.dataset.ann_file=$IN_DIR/coco.json
```

The predictions can be converted to COCO by running `bitewings/inference/mmdet2coco.py`. Afterward, the annotations can be visualized by running `bitewings/inference/show_anns.py`. One example bitewing has been added in the `test/` folder with which you can run the models and show the prediction results.


## OdontoAI pretraining

If you would like to skip this step, model checkpoints pre-trained on COCO and OdontoAI are made available on request.

**Preprocessing** Unzip the downloaded data from the OdontoAI platform to the `../data/` folder and run `bitewings/odonto/preprocess.py` to combine the train and validation images and to split the data.

**Training** Following the preprocessing, the Mask DINO, Mask R-CNN, and SparseInst models can be pretrained by running the following command:

```bash
PYTHONPATH=. python mmdetection/tools/train.py bitewings/odonto/config_<model>.py
```

choosing a model architecture for `<model>`. The training run will be logged using TensorBoard and the checkpoints and logging files will be stored in `work_dirs/odonto_bitewings_<model>`.


## Fine-tuning

### Preprocessing

Unzip the downloaded data collected from The Netherlands to the `../data/` folder. As the teeth and tooth findings were annotated independently, each tooth finding needs to be matched to a tooth with corresponding FDI number. Please run `bitewings/preprocess/preprocess.py` to automatically assign tooth findings to teeth and to split the images into train, validation, and test.

Furthermore, `bitewings/preprocess/intensities.py` and `bitewings/preprocess/prevalences.py` can be run to visualize the intensity distribution and to show the tooth finding prevalances of the bitewings from The Netherlands. Please note that the data from The Netherlands does not include implants.


### Training

After pretraining a model on the OdontoAI dataset, the model checkpoint in the working directory can be copied to `../checkpoints/<model>_odonto.pth`, after which it can be fine-tuned on the bitewings from The Netherlands using the following command:

```bash
PYTHONPATH=. python mmdetection/tools/train.py bitewings/configs/config_<model>.py
```

Please note that the hierarchical instance segmentation method requires at least 24GB of GPU memory and typically requires 20 GPU hours to complete training.


### Evaluation

A fine-tuned model can be evaluated on bitewings from Slovakia in an external validation using the following command:

```bash
PYTHONPATH=. python mmdetection/tools/test.py bitewings/configs/config_<model>.py work_dirs/lingyun_trainval_<model>/epoch_36.pth
```

This will produce mean average precision (mAP) metrics for tooth segmentation and labeling and the results will be written to a pickel file in the working directory.

Further metrics for tooth segmentation and labeling can be computed by running `bitewings/evaluation/tooth_segmentation.py`. Furthermore, metrics can be computed for tooth finding classification by running `bitewings/evaluation/tooth_findings.py`. Lastly, figures showing the original bitewing, the bitewing with annotations and the bitewing with model predictions can be visualized by running `bitewings/visualization/side_by_side.py`.


## Citation

```
@article{bitewing_chart_filing,
  title={Comprehensive Bitewing Chart Filing using Hierarchical End-to-End Vision Transformer: A Multi-Center Study},
  author={Cao, Lingyun and van Nistelrooij, Niels},
  year={2024},
  journal={Computers in Biology and Medicine},
  note={In press}
}
```

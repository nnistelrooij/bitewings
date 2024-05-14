## First, make a new conda environment
```bash
conda create -n bitewings python=3.9
conda activate bitewings
```

## Install PyTorch 2.0.1
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Install pip requirements
```bash
pip install -r pip_requirements.txt
```

## Install mim requirements
```bash
pip install -U openmim
mim install -r mim_requirements.txt
pip install -e mmdetection
```

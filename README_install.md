## Installation

This repository is built in PyTorch 1.10.0 and tested on Ubuntu 20.04 environment (Python3.8, CUDA11.3).
Follow these intructions

1. Clone our repository

2. Make conda environment
```
conda create -n UniProcessor_arxiv python=3.8 pip
conda activate UniProcessor_arxiv
```

3. Install dependencies
```
conda install pytorch==2.1.1 torchvision==0.16.1 cudatoolkit=11.3 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install -r requirements.txt
```
if in China, please use `--channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/` after `conda`, and `-i https://pypi.tuna.tsinghua.edu.cn/simple` after `pip`.

4. Install LAVIS
```
cd  models/lavis
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-contrib-python accelerate chardet cchardet huggingface_hub
```
download all required packages from `lavis` and put in `models/lavis/pretrained_weights` folder.
unzip `blip-diffusion` use the command.
```
cd models/lavis/pretrained_weights/
tar -xf blip-diffusion.tar.gz
```
test whether lavis is successfully installed
```
python test_lavis.py
```

5. If want to try distortion generation
```
cd ir_data_uni
python distortion_bank.py
```
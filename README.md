# Pytorch Detectron2 Detect

## Install
```bash
conda create -n dt2 python=3.8
conda activate dt2

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
or
cd ~
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

## Download Model
```bash
https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl
```

## Run
```bash
python Detectron2Detector.py
```

## Enjoy it~


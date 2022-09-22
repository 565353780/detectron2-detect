pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113

cd ..
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2


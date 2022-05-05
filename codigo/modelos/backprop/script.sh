#!/bin/bash
epocas=30
programa=backprop_qtp.py
python $programa --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 0
python $programa --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 1

python $programa --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 0
python $programa --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 1

python $programa --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 0
python $programa --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 1

python $programa --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 0
python $programa --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 1

python $programa --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 0
python $programa --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 1

python $programa --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 0
python $programa --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 1

./script1.sh

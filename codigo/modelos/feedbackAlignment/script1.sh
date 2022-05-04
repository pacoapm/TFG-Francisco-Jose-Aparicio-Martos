#!/bin/bash
epocas=30
python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 1 --modo 1

python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 1 --modo 1

python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 1 --modo 1

python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 1 --modo 1

python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 1 --modo 1

python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 0 --modo 1
python mnist_fa_qtp.py --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 1 --modo 1

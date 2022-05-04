#!/bin/bash
epocas=30
python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 8 --global-quantization 1

python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 6 --global-quantization 1

python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset MNIST --n-bits 4 --global-quantization 1

python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 8 --global-quantization 1

python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 6 --global-quantization 1

python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 0
python mnist_dni_qtp.py --epochs $epocas --dataset FMNIST --n-bits 4 --global-quantization 1

./script1.sh

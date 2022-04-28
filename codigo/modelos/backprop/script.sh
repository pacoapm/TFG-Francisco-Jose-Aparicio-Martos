#!/bin/bash

python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 8 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 8 --global-quantization 1

python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 6 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 6 --global-quantization 1

python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 4 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset MNIST --n-bits 4 --global-quantization 1

python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 8 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 8 --global-quantization 1

python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 6 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 6 --global-quantization 1

python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 4 --global-quantization 0
python mnist_backprop_qtp.py --epochs 10 --dataset FMNIST --n-bits 4 --global-quantization 1

./script1.sh

#!/bin/bash

python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 8 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 8 --global-quantization 1 --modo 1

python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 6 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 6 --global-quantization 1 --modo 1

python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 4 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset MNIST --n-bits 4 --global-quantization 1 --modo 1

python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 8 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 8 --global-quantization 1 --modo 1

python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 6 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 6 --global-quantization 1 --modo 1

python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 4 --global-quantization 0 --modo 1
python tests/HSIC_qtp.py --epochs 10 --dataset FMNIST --n-bits 4 --global-quantization 1 --modo 1

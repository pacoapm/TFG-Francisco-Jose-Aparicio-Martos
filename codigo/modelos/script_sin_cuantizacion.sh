#!/bin/bash
cd feedbackAlignment
python mnist_fa.py --dataset MNIST --n-layers 5 --hidden-width 20 --save-model
python mnist_fa.py --dataset FMNIST --n-layers 5 --hidden-width 20 --save-model
cd ..
cd HSIC
source env.sh
python tests/HSIC.py --dataset MNIST --n-layers 5 --hidden-width 20 --save-model
python tests/HSIC.py --dataset FMNIST --n-layers 5 --hidden-width 20 --save-model
cd ..
./script_cuantizacion


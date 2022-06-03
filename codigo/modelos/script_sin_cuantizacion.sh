#!/bin/bash
cd backprop
python backprop.py --dataset MNIST --n-layers $1 --hidden-width $2 --save-model
python backprop.py --dataset FMNIST --n-layers $1 --hidden-width $2 --save-model
cd ..
cd dni
python mnist_dni.py --dataset MNIST --n-layers $1 --hidden-width $2 --save-model
python mnist_dni.py --dataset FMNIST --n-layers $1 --hidden-width $2 --save-model
cd ..
cd feedbackAlignment
python mnist_fa.py --dataset MNIST --n-layers $1 --hidden-width $2 --save-model
python mnist_fa.py --dataset FMNIST --n-layers $1 --hidden-width $2 --save-model
cd ..
cd HSIC
source env.sh
python tests/HSIC.py --dataset MNIST --n-layers $1 --hidden-width $2 --save-model
python tests/HSIC.py --dataset FMNIST --n-layers $1 --hidden-width $2 --save-model
cd ..
./script_cuantizacion $1 $2


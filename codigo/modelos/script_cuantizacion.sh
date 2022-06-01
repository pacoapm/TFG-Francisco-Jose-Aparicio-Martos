#!/bin/bash
#cd backprop
#./pruebas.sh
#cd ..
cd dni
python mnist_dni.py --dataset MNIST --n-layers 5 --hidden-width 20 --save-model
python mnist_dni.py --dataset FMNIST --n-layers 5 --hidden-width 20 --save-model
./pruebas.sh
#cd ..

#cd feedbackAlignment
#./pruebas.sh
#cd ..
#cd HSIC
#source env.sh
#./pruebas.sh

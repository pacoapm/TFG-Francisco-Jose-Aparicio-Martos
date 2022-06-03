#!/bin/bash
cd backprop
./pruebas.sh $1 $2
cd ..
cd dni
./pruebas.sh $1 $2
cd ..

cd feedbackAlignment 
./pruebas.sh $1 $2
cd ..
cd HSIC
source env.sh
./pruebas.sh $1 $2

#!/bin/bash

cd dni
./script.sh
cd ..

cd feedbackAlignment 
./script.sh
cd ..
cd HSIC
source env.sh
./script.sh

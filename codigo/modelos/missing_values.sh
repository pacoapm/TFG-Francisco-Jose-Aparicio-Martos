cd feedbackAlignment 
echo "Arquitectura 5 capas 20 unidades;;;;;" >> datos/MNIST.csv
python mnist_fa_qtp.py --dataset MNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 5 --hidden-width 20
cd ..
cd dni

python mnist_dni_qtp.py --dataset FMNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 2 --hidden-width 100
python mnist_dni_qtp.py --dataset FMNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 1 --hidden-width 50
python mnist_dni_qtp.py --dataset FMNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 2 --hidden-width 50

python mnist_dni_qtp.py --dataset MNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 5 --hidden-width 20
python mnist_dni_qtp.py --dataset MNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 2 --hidden-width 100
python mnist_dni_qtp.py --dataset MNIST --modo 0 --n-bits 2 --global-quantization 0 --epochs 30 --n-layers 2 --hidden-width 50

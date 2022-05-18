funciones=(0 1)
bits=(2 3 4 5 6 7 8)
global=(1 0)
database=(MNIST FMNIST)
programa=mnist_dni_qtp.py

for k in "${database[@]}"
do
	for i in "${funciones[@]}"
	do
		for j in "${bits[@]}"
		do
			for z in "${global[@]}"
			do
				python $programa --dataset $k --modo $i --n-bits $j --global-quantization $z --epochs 30
			done
		done
	done
done

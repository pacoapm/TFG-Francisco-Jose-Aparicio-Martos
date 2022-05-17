funciones=(0 1)
bits=(3 2 1)
global=(1)
database=(MNIST FMNIST)
programa=tests/HSIC_qtp.py

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
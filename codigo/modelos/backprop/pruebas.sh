funciones=(0)
bits=(2 3 4 5 6 7 8)
global=(0 1)
database=(MNIST FMNIST)
programa=backprop_qtp.py

for k in "${database[@]}"
do
	for i in "${funciones[@]}"
	do
		for j in "${bits[@]}"
		do
			for z in "${global[@]}"
			do
				echo python $programa --dataset $k --modo $i --n-bits $j --global-quantization $z --epochs 30 --n-layers 5 --hidden-width 20
				python $programa --dataset $k --modo $i --n-bits $j --global-quantization $z --epochs 30 --n-layers 5 --hidden-width 20
			done
		done
	done
done

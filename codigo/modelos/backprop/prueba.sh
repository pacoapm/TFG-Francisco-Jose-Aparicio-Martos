funciones=(0 1)
bits=(8 6 4)
global=(0 1)
database=(MNIST FMNIST)
programa=backprop.py

for k in "${database[@]}"
do
	for i in "${funciones[@]}"
	do
		for j in "${bits[@]}"
		do
			for z in "${global[@]}"
			do
				echo "python" $programa "--database" $k "--modo" $i "--n-bits" $j "--global-quantization" $z
			done
		done
	done
done

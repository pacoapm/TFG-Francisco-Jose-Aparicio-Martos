#!/bin/bash
experimento(){
	echo Arquitectura $1 capas $2 unidades capa >> datos/MNIST.csv
	echo Arquitectura $1 capas $2 unidades capa >> datos/FMNIST.csv
	./pruebas.sh $1 $2
}

experimento 0 4
experimento 5 20
experimento 2 100
experimento 1 100
experimento 1 50
experimento 2 50

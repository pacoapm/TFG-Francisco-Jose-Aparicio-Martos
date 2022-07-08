# Trabajo Fin de Grado: Estudio de algoritmos de entrenamiento en redes neuronales cuantificadas

En este proyecto fin de grado se ha realizado un estudio del comportamiento de varios algoritmos de entrenamiento en redes neuronales cuantificadas. 

Los algoritmos estudiados son:  
-[Backpropagation](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/modelos/backprop)  [[1]](#1)   
-[HSIC](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/modelos/HSIC) [[2]](#2)  
-[Synthetic Gradients](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/modelos/dni)  [[3]](#3)  
-[Feedback Alignment](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/modelos/feedbackAlignment) [[4]](#4) 

La cuantificación de los pesos de la red se ha realizado con las funciones Uniform-ASYMM y Uniform-SYMM [[5]](#5). 

## Manual de usuario

A continuación se presenta las instrucciones para reproducir los experimentos realizados.

### Ejecución algoritmos

Para ejecutar cada uno de los algoritmos de forma manual se debe de acceder a las carpetas que contienen los algoritmos. Los scripts con los algoritmos son los siguientes: 
- Backpropagation: [backprop.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/backprop/backprop.py)  [backprop_qtp.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/backprop/backprop_qtp.py) 
- HSIC: [HSIC.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/HSIC/tests/HSIC.py) [HSIC_qtp.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/HSIC/tests/HSIC_qtp.py)
- Synthetic Gradients: [dni.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/dni/dni.py) [dni_qpt.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/dni/dni_qtp.py)  
- Feedback Alignment: [fa.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/feedbackAlignment/fa.py) [fa_qtp.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/modelos/feedbackAlignment/fa_qtp.py)

Los archivos finalizados en _qtp son las versiones cuantificadas de los algoritmos.

<br/>

Ubicado en la carpeta del algoritmo que queremos ejecutar introducimos el siguiente comando:

```
python nombre_archivo --epochs x --n-layers x --hidden-width x --input-widht x --output-width x

python nombre_archivo_qtp --epochs x --n-layers x --hidden-width x --input-widht x --output-width x --n-bits x --global-quantization 0/1 --modo 0/1 
```

Para obtener información de más opciones disponibles e información detallada de cada opción, usar el siguiente comando:

```
python nombre_archivo --help
```

Tras la ejecución del modelo sin cuantificación los pesos se alamacenan en la carpeta [pesosModelos](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/pesosModelos). Mientras que si se ejecuta los modelos cuantificados se guardará información sobre el entrenamiento en las carpetas especificadas en la memoria.

### Ejecución experimentos
Para poder ejecutar un gran número de pruebas de forma automática se pueden usar los scritps que se encuentran las carpetas de cada algoritmo. Se llaman script.sh y pruebas.sh.

El archivo script.sh se encarga de añadir la cabecera de entrada en el archivo de la carpeta datos y ejecuta el archivo pruebas.sh, especificando la anchura y profundidad de la red sobre la que ejecutar el algoritmo de entrenamiento. El archivo pruebas.sh se encargará de realizar de especificar la cantidad de experimentos a realizar. Dentro de este archivo se puede modificar:

- la cantidad de bits
- las funciones de cuantificación
- el nivel al que aplicar la cuantificación
- el conjunto de datos sobre el que hacer la ejecución

#### Ejemplo de ejecución
Si queremos hacer pruebas con arquitecturas de 2 capas ocultas y 50 neuronas de anchura haremos lo siguiente: Dentro de script.sh escribimos:
```
experimento 1 50 #se especifica 1 porque los modelos bases tienen 1 capa oculta por defecto
```
Ahora editamos el script pruebas.sh. Para ello escribimos en cada uno de los parámetros los experimentos que queramos realizar. Por ejemplo queremos pruebas con:
- 5, 6 y 7 bits
- cuantificación a nivel global y local
- usar la función de cuantificación ASYMM
Entonces tendremos que editar las opciones de la siguiente forma:

```
funciones=(0)
bits=(5 6 7)
global=(0 1)
```

El nombre del programa y el conjunto de datos no se tienen que modificar. 

<br/>
Finalmente, para mayor automatización y no tener que ejecutar a mano un script por algoritmo, en la carpeta [modelos](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/tree/main/codigo/modelos) encontramos dos script para lanzar todas las experimentaciones que hayamos especificado para cada uno de los algorimtos. 

Para ello ejecutamos el archivo script_cuantizacion.sh.

```
./script_cuantizacion.sh
```
**Importante**: Para ejecutar cada experimento sobre una cierta arquitectura, se debe de haber previamente creado la version sin cuantificar de dicha red. Tomando como muestra el ejemplo anterior: Si queremos hacer ése experimento, se tiene que haber ejecutado el algorimto a estudiar sobre una red con las mismas dimensiones del experimento. Esto es así ya que en la extracción de información se hace una comparción entre los modelos con y sin cuantificación.


### Gráficas

Para sacar las gráficas se deben de usar los scripts de python: [extractor.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/extractor.py) y [extractor_tablas.py](https://github.com/pacoapm/TFG-Francisco-Jose-Aparicio-Martos/blob/main/codigo/extractor_tablas.py). Estos archivos son sencillos de editar. Simplemente en cada uno de los argumentos se especifican los experimentos a graficar. Dentro de cada archivo hay ejemplos de uso. 










## Bibliografía

<a id="1">[1]</a> D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning
representations by back-propagating errors,” Nature (London), vol.
323, no. 6088, pp. 533–536, Oct 9, 1986.

<a id="2">[2]</a> W.-D. K. Ma, J. Lewis, and W. B. Kleijn, “The hsic bottleneck: Deep
learning without back-propagation,” in Proceedings of the AAAI Con-
ference on Artificial Intelligence, vol. 34, no. 04, 2020, pp. 5085–5092.

<a id="3">[3]</a> M. Jaderberg, W. M. Czarnecki, S. Osindero, O. Vinyals, A. Graves,
D. Silver, and K. Kavukcuoglu, “Decoupled neural interfaces using
synthetic gradients,” in International conference on machine learning.
PMLR, 2017, pp. 1627–1635

<a id="4">[4]</a> T. P. Lillicrap, D. Cownden, D. B. Tweed, and C. J. Akerman, “Random
synaptic feedback weights support error backpropagation for deep
learning,” Nature Communications, vol. 7, no. 1, pp. 1–10, -11-08 2016.

<a id="5">[5]</a> P. Nayak, D. Zhang, and S. Chai, “Bit efficient quantization for deep
neural networks,” 2019.


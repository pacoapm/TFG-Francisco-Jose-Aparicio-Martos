#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:19:57 2022

@author: francisco
"""

from __future__ import print_function
import argparse

import numpy as np
import matplotlib.pyplot as plt

def substring(string, ini, fin):
    return string[string.find(ini)+len(ini):string.find(fin)]

#chequea que los pesos de los archivos esten entre -1 y 1
def comprobarPesos(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    
    
    datos = []
    
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T

    
    
    
    return np.all(abs(datos) <= 1)

#grafica la evolucion de los pesos en los experimentos
def graficarPesos(archivo):
    fig = plt.figure(figsize=(8,6))
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    
    datos = []
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])+2
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T
    
    
    batch_size = 64
    dataset_len = 60000
    
    iteraciones = int(dataset_len/batch_size)
    for i in range(n_layers):
        
        plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
    
    plt.legend()
    plt.show()
    
    
    
#grafica la evolucion de los pesos del algorimto synthetic gradient
def graficarPesosDNI(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    
    datos = []
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])*4+5
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T
    
    
    batch_size = 64
    dataset_len = 60000
    
    iteraciones = int(dataset_len/batch_size)
    
    
    for i in range(n_layers):
        plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
    
    plt.legend()
    plt.show()
    
    
#extrea valores maximos, minimos, media y varianza de los pesos por capa
def datosPesos(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    datos = []
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])+2
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T
    
    dataset = substring(archivo,"sinq_","_nbits")  
    epocas = substring(archivo,"epochs","_global")
    batch_size = 64
    dataset_len = 60000
    
    iteraciones = int(dataset_len/batch_size)
    for i in range(n_layers):
        
        #plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
        info = datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,:]
        
        print("Capa: ", i)
        print("Peso maximo: %.4f Peso minimo: %.4f" % (np.max(info[:,0]), np.min(info[:,1])))
        print("Peso maximo medio: %.4f Peso minimo medio: %.4f" % (np.mean(info[:,0]), np.mean(info[:,1])))
        print("Peso maximo varianza: %.4f Peso minimo varianza: %.4f" % (np.var(info[:,0]),np.var(info[:,1])))
        
#extrea valores maximos, minimos, media y varianza de los pesos a nivel global 
def datosPesosJuntos(archivo, tabla):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    datos = []
    
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T
    

    info = datos

    print("Peso maximo: %.4f Peso minimo: %.4f" % (np.max(info[:,0]), np.min(info[:,1])))
    print("Peso maximo medio: %.4f Peso minimo medio: %.4f" % (np.mean(info[:,0]), np.mean(info[:,1])))
    print("Peso maximo varianza: %.4f Peso minimo varianza: %.4f" % (np.var(info[:,0]),np.var(info[:,1])))
    
    tabla[-1] = tabla[-1]+str(np.var(info[:,1]))+","+str(np.var(info[:,]))+","+str(np.min(info[:,1]))+","+str(np.max(info[:,0]))
    plt.boxplot(datos[:,0])
    plt.show()
    
    
#igual que datosPesos pero para el algoritmo Feedback Alignment
def datosPesosFA(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    datos = []
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])+2
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1),np.max(datos[:,[4,6]],axis=1),np.min(datos[:,[5,7]],axis=1)]).T
    
    dataset = substring(archivo,"sinq_","_nbits")  
    epocas = substring(archivo,"epochs","_global")
    batch_size = 64
    dataset_len = 60000
    
    iteraciones = int(dataset_len/batch_size)
    for i in range(n_layers):
        
        #plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
        info = datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,:]
        
        print("Capa: ", i)
        print("Peso maximo: %.4f Peso minimo: %.4f Peso back maximo %.4f Peso back minimo %.4f" % (np.max(info[:,0]), np.min(info[:,1]), np.max(info[:,2]), np.min(info[:,2])))
        print("Peso maximo medio: %.4f Peso minimo medio: %.4f Peso back maximo medio %.4f Peso back minimo medio %.4f" % (np.mean(info[:,0]), np.mean(info[:,1]),np.mean(info[:,2]), np.mean(info[:,2])))
        print("Peso maximo varianza: %.4f Peso minimo varianza: %.4f Peso back maximo varianza %.4f Peso back minimo varianza %.4f" % (np.var(info[:,0]),np.var(info[:,1]),np.var(info[:,2]), np.var(info[:,2])))
        
    
    #print("Los pesos son correctos: ", np.all(abs(datos) <= 1.1))
    
    
#igual que datosPesos pero para el algoritmo Feedback synthetic gradients
def datosPesosDNI(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    f.close()
    datos = []
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])*4+5
    
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
    datos = np.array(datos)
    datos = np.array([np.max(datos[:,[0,2]],axis=1),np.min(datos[:,[1,3]],axis=1)]).T
    
    dataset = substring(archivo,"sinq_","_nbits")  
    epocas = substring(archivo,"epochs","_global")
    batch_size = 64
    dataset_len = 60000
    
    iteraciones = int(dataset_len/batch_size)
    for i in range(n_layers):
        
        #plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
        info = datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,:]
        
        print("Capa: ", i)
        print("Peso maximo: %.4f Peso minimo: %.4f" % (np.max(info[:,0]), np.min(info[:,1])))
        print("Peso maximo medio: %.4f Peso minimo medio: %.4f" % (np.mean(info[:,0]), np.mean(info[:,1])))
        print("Peso maximo varianza: %.4f Peso minimo varianza: %.4f" % (np.var(info[:,0]),np.var(info[:,1])))
        
    
    
    
#funciones para mostrar la informacion de los experimentos especificados   
    
def mostrarInfoPesos(ruta,var):
    dicc = {0:"ASYMM",1:"SYMM"}
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    print("Dataset: ", dataset, " n_bits: ", bits, " funcion: ", dicc[func], " global: ", glbl)
                    datosPesos(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4")
                    print("\n\n")

def mostrarInfoPesosv2(ruta,var):
    dicc = {0:"ASYMM",1:"SYMM"}
    info = []
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    print("Dataset: ", dataset, " n_bits: ", bits, " funcion: ", dicc[func], " global: ", glbl)
                    info.append(str(bits)+",")
                    datosPesosJuntos(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4", info)
                    print("\n\n")
                    
        print("nbits,vmin,vmax,pmin,pmax")
        for i in info:
            print(i)
        
        

def graficarInfoPesos(ruta,var):
    dicc = {0:"ASYMM",1:"SYMM"}
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    print("Dataset: ", dataset, " n_bits: ", bits, " funcion: ", dicc[func], " global: ", glbl)
                    graficarPesos(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4")
                    print("\n\n")
    
    
#extrae la informacion de la precision y la funcion de perdida de cada experimento
def extraerLossAcc(archivo):
    print(archivo)
    datos = []
    with open(archivo,'r') as f:
        lines = f.readlines()
        for line in lines:
            datos.append(list(map(float,line.split(" "))))
    datos = np.array(datos)
    
    return datos[:,0],datos[:,1]

#grafica la evolucion de la funcion de perdida y precision del experimento especificado
def graficarEvaluacion(ruta,var):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    
    
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    loss, acc = extraerLossAcc(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4")
                    x = np.arange(1,len(loss)+1)
                    ax[0].plot(x,loss,'*-',label="n_bits"+str(bits))
                    ax[1].plot(x,acc,'*-', label="n_bits"+str(bits))
                    ax[0].grid()
                    ax[1].grid()
    
    
    ax[0].legend(bbox_to_anchor=(1.04,1),borderaxespad=0)
    ax[0].set_title("Test loss")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("Loss")
    #ax[0].set_ylim(0,max(loss)+1)
    
    ax[1].legend(bbox_to_anchor=(1.04,1),borderaxespad=0)
    ax[1].set_title("Test acc")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0,100)

    fig.tight_layout()
    plt.show()

#grafica la evolucion de los experimentos especificados
def graficarACC(ruta,var):
    fig = plt.figure()
    x = np.arange(1,var["epochs"][0]+1)
    dicc = {0:"global",1:"local"}
    colores = ["r","b","g","y","c","m","y"]
    iterador = 0
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    
                    loss, acc = extraerLossAcc(ruta+"sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers"+str(var["nlayers"])+"_hidden_width"+str(var["nunits"]))
                    print(len(acc))
                    x = np.arange(1,len(acc)+1)
                    #plt.plot(x,loss,'*-',label="n_bits"+str(bits))
                    if len(var["n_bits"]) > 1:
                        plt.plot(x,acc,'*-', label=str(bits)+" bits")
                    if len(var["globl"]) > 1:
                        plt.plot(x,acc,'*-', label=dicc[glbl])
                        
                    """if glbl == 0:
                        plt.plot(x,acc,colores[iterador]+'*--', label="n_bits"+str(bits))
                        iterador += 1
                    else:
                        plt.plot(x,acc,colores[iterador]+'*-', label="n_bits"+str(bits))
                        iterador += 1"""
                    
    
    
    
    plt.legend(bbox_to_anchor=(1.04,1))
    plt.title("Precisi??n en el Test")
    plt.xlabel("??pocas")
    plt.ylabel("Precisi??n")
    plt.ylim(0,100)
    plt.yticks(range(0,100,10))
    plt.grid()
    
    fig.tight_layout()

    plt.show()
                    
                


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='hola')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    
    """for i in glob(args.ruta+"/*_global1*"):
        #print(i)
        if comprobarPesos(i) == False:
            print("NO es correcto")"""
        #hol = input()
    #graficarACC("modelos/feedbackAlignment/historial/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[2,3,4,5,6,7,8],"func":[1],"globl":[0]})
    #graficarInfoPesos("modelos/HSIC/infoPesos/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[2,3,4,5,6,7,8],"func":[1],"globl":[1]})
    #graficarEvaluacion("modelos/HSIC/historial/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[2,3,4,5,6,7,8],"func":[0],"globl":[0]})
    graficarACC("modelos/feedbackAlignment/historial/",{"epochs":[30],"dataset":["FMNIST"],"n_bits":[6,7,8],"func":[1],"globl":[1], "nlayers":5,"nunits":20})
    #mostrarInfoPesosv2("modelos/HSIC/infoPesos/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[6,7,8],"func":[0],"globl":[1]})
    #mostrarInfoPesosv2("modelos/HSIC/infoPesos/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[6,7,8],"func":[0],"globl":[1]})
    
    
if __name__=="__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:06:41 2022

@author: francisco
"""
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

def substring(string, ini, fin):
        
    return string[string.find(ini)+len(ini):string.find(fin)]

def leerTabla(ruta):
    f = open(ruta,"r")
    Lines = f.readlines()
    f.close()
    
    datos = []
    for line in Lines:
        datos.append(line.replace("+ADs-+AC0",";").replace("+ADs-",";").replace("+AC0",";").split(";"))
        
    
    return datos

def extraerDatosArquitectura(ruta, num_capas, num_unidades):
    datos = leerTabla(ruta)
    contador = 0
    lineas = []
    num_linea = -1
    for i in datos:
        if i[0].find("Arquitectura") != -1:
            ncapas = int(substring(i[0],"Arquitectura "," capas"))
            nunidades = int(substring(i[0],"capas "," unidades"))
            if ncapas == num_capas and num_unidades == nunidades:
                num_linea = len(lineas)
            lineas.append([ncapas,nunidades,contador])
        contador += 1
    lineas.append(["","",len(datos)])
    
    return np.array(datos[lineas[num_linea][2]+1:lineas[num_linea+1][2]])

def graficarPrec(datos,titulo,ruta):
    fig = plt.figure()
    x_local = datos[np.where(datos[:,1] == "local")[0],0]
    x_global = datos[np.where(datos[:,1] == "global")[0],0]
    datos = np.array(datos)
    y_local = np.array(datos[np.where(datos[:,1] == "local")[0],5],dtype=float)
    y_global = np.array(datos[np.where(datos[:,1] == "global")[0],5],dtype=float)
    plt.plot(x_local,y_local,"*-",label="local")
    plt.plot(x_global,y_global,"*-",label="global")
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0,100,10))
    plt.savefig(ruta+titulo)
    plt.show()
    
    
def graficarPrec2(datos_mnist,datos_fmnist,funcion,titulo,ruta):
    fig = plt.figure()
    for datos, dataset, linea in zip([datos_mnist,datos_fmnist],["MNIST","FMNIST"],["*-","*--"]):
        x_local = datos[np.where(datos[:,1] == "local" & datos[:,2] == funcion)[0],0]
        x_global = datos[np.where(datos[:,1] == "global" & datos[:,2] == funcion)[0],0]
        datos = np.array(datos)
        y_local = np.array(datos[np.where(datos[:,1] == "local" & datos[:,2] == funcion)[0],5],dtype=float)
        y_global = np.array(datos[np.where(datos[:,1] == "global" & datos[:,2] == funcion)[0],5],dtype=float)
        plt.plot(x_local,y_local,linea+"b",label="local "+dataset)
        plt.plot(x_global,y_global,linea+"r",label="global "+dataset)
        
    plt.title(titulo)
    plt.xlabel("Número bits")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0,100,10))
    plt.savefig(ruta+titulo)
    #plt.show()
    
def graficarPrec3(datos_mnist,datos_fmnist,titulo,ruta):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    
    for datos, dataset, linea in zip([datos_mnist,datos_fmnist],["MNIST","FMNIST"],["*-","*--"]):
        x_local = datos[np.where(datos[:,1] == "local")[0],0]
        x_global = datos[np.where(datos[:,1] == "global")[0],0]
        datos = np.array(datos)
        y_local = np.array(datos[np.where(datos[:,1] == "local")[0],5],dtype=float)
        y_global = np.array(datos[np.where(datos[:,1] == "global")[0],5],dtype=float)
        plt.plot(x_local,y_local,linea+"b",label="local "+dataset)
        plt.plot(x_global,y_global,linea+"r",label="global "+dataset)
        
    plt.title(titulo)
    plt.xlabel("Número bits")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid()
    plt.yticks(np.arange(0,100,10))
    
    fig.tight_layout()
    plt.savefig(ruta+titulo)
    #plt.show()
    
def graficarFQuant(datos_mnist,datos_fmnist,titulo,ruta):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    for grafica,nombre in zip([0,1],["ASYMM","SYMM"]):
        for datos, dataset, linea in zip([datos_mnist,datos_fmnist],["MNIST","FMNIST"],["*-","*--"]):
            
            x_local = datos[np.where((datos[:,1] == "local") & (datos[:,2] == nombre))[0],0]
            x_global = datos[np.where((datos[:,1] == "global") & (datos[:,2] == nombre))[0],0]
            datos = np.array(datos)
            y_local = np.array(datos[np.where((datos[:,1] == "local") & (datos[:,2] == nombre))[0],5],dtype=float)
            y_global = np.array(datos[np.where((datos[:,1] == "global") & (datos[:,2] == nombre))[0],5],dtype=float)
            ax[grafica].plot(x_local,y_local,linea+"b",label="local "+dataset)
            ax[grafica].plot(x_global,y_global,linea+"r",label="global "+dataset)
            
        ax[grafica].set_title(nombre)
        ax[grafica].set_xlabel("Número bits")
        ax[grafica].set_ylabel("Precisión")
        ax[grafica].legend()
        ax[grafica].grid()
        ax[grafica].set_yticks(np.arange(0,100,10))
        
    fig.suptitle(titulo)
    plt.savefig(ruta+titulo)
    #plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='hola')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    
    #capas = [5,2,1,1,2]
    #unidades = [20,100,100,50,50]
    capas = [1]
    unidades = [4]
    
    """for i,j in zip(capas,unidades):
        datos = extraerDatosArquitectura("modelos/HSIC/datos/FMNIST.csv",i,j)
        print(datos)
        graficarPrec(datos,"{} capa {} unidades".format(i,j),"modelos/HSIC/graficas/FMNIST/")"""
        
    """for dataset in ["feedbackAlignment","backprop","dni","HSIC"]:
        for i,j in zip(capas,unidades):
            datos_fmnist = extraerDatosArquitectura("modelos/"+dataset+"/datos/FMNIST.csv",i,j)
            datos_mnist = extraerDatosArquitectura("modelos/"+dataset+"/datos/MNIST.csv",i,j)
            
            #print(datos)
            graficarFQuant(datos_mnist,datos_fmnist,"Precision con arquitectura: {} capa {} unidades".format(i,j),"modelos/"+dataset+"/graficas/")
"""
    capas = [5,2,1,1,2]
    unidades = [20,100,100,50,50]
    for alg in ["feedbackAlignment"]:
        for i,j in zip(capas,unidades):
            datos_fmnist_s = extraerDatosArquitectura("modelos/"+alg+"/datos/FMNIST_v2.csv",i,j)
            datos_mnist_s = extraerDatosArquitectura("modelos/"+alg+"/datos/MNIST_v2.csv",i,j)
            datos_fmnist_a = extraerDatosArquitectura("modelos/"+alg+"/datos/FMNIST.csv",i,j)
            datos_mnist_a = extraerDatosArquitectura("modelos/"+alg+"/datos/MNIST.csv",i,j)
            
            #print(datos)
            graficarPrec2(datos_mnist_s,datos_fmnist_s,"ASYMM","Precision con arquitectura: {} capas {} unidades. Función {}".format(i,j,"ASYMM"),"modelos/"+alg+"/graficas/")

    """capas = [5,2,1,1,2]
    unidades = [20,100,100,50,50]
    for dataset in ["feedbackAlignment","backprop","dni","HSIC"]:
        for i,j in zip(capas,unidades):
            datos_fmnist = extraerDatosArquitectura("modelos/"+dataset+"/datos/FMNIST.csv",i,j)
            datos_mnist = extraerDatosArquitectura("modelos/"+dataset+"/datos/MNIST.csv",i,j)
            
            #print(datos)
            graficarPrec2(datos_mnist,datos_fmnist,"Precision con arquitectura: {} capa {} unidades".format(i,j),"modelos/"+dataset+"/graficas/")
"""
    
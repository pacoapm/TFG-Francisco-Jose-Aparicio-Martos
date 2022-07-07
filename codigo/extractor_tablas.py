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

#lectura de los archivos dentro de la carpeta de datos
def leerTabla(ruta):
    f = open(ruta,"r")
    Lines = f.readlines()
    f.close()
    
    datos = []
    for line in Lines:
        datos.append(line.replace("+ADs-+AC0",";").replace("+ADs-",";").replace("+AC0",";").split(";"))
        
    
    return datos

#de la tabla leida extrae los datos de la arquitectura especificada
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

#grafica la precision que se les pasa como argumento
def graficarPrec(datos_mnist,datos_fmnist,funcion,titulo,ruta):
    fig = plt.figure()
    for datos, dataset, linea in zip([datos_mnist,datos_fmnist],["MNIST","FMNIST"],["*-","*--"]):
        
        x_local = datos[np.where((datos[:,1] == "local") & (datos[:,2] == funcion))[0],0]
        x_global = datos[np.where((datos[:,1] == "global") & (datos[:,2] == funcion))[0],0]
        #datos = np.array(datos)
        
        y_local = np.array(datos[np.where((datos[:,1] == "local") & (datos[:,2] == funcion))[0],5],dtype=float)
        y_global = np.array(datos[np.where((datos[:,1] == "global") & (datos[:,2] == funcion))[0],5],dtype=float)
        
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
    
#genera una tabla con las precisiones de los modelos sin cuantificar
def precSinCuantificacion():
    tabla_fmnist = tabla_mnist = "Arquitectura,BP,HSIC,FA,SG\n"
    
    capas = [1,5,2,1,1,2]
    unidades = [4,20,100,100,50,50]
    
    for i,j in zip(capas,unidades):
        tabla_fmnist += "{} capas {} unidades".format(i+1,j)+","
        tabla_mnist += "{} capas {} unidades".format(i+1,j)+","
        for alg in ["backprop","HSIC","feedbackAlignment","dni"]:
            datos_fmnist= extraerDatosArquitectura("modelos/"+alg+"/datos/FMNIST.csv",i,j)
            datos_mnist= extraerDatosArquitectura("modelos/"+alg+"/datos/MNIST.csv",i,j)
            
            tabla_fmnist += str(datos_fmnist[0,3])+","
            tabla_mnist += str(datos_mnist[0,3])+","
            
            
        tabla_fmnist += "\n"
        tabla_mnist += "\n"
        
    return tabla_mnist, tabla_fmnist
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='hola')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    
    
    
    a,b = precSinCuantificacion()
    print(a,b)

    capas = [1,5,2,1,1,2]
    unidades = [4,20,100,100,50,50]
    for alg in ["feedbackAlignment","backprop","HSIC","dni"]:
        for i,j in zip(capas,unidades):
            datos_fmnist= extraerDatosArquitectura("modelos/"+alg+"/datos/FMNIST.csv",i,j)
            datos_mnist= extraerDatosArquitectura("modelos/"+alg+"/datos/MNIST.csv",i,j)
            
            graficarPrec(datos_mnist,datos_fmnist,"SYMM","Precision con arquitectura: {} capas {} unidades Funcion {}".format(i+1,j,"SYMM"),"modelos/"+alg+"/graficas/")
            graficarPrec(datos_mnist,datos_fmnist,"ASYMM","Precision con arquitectura: {} capas {} unidades Funcion {}".format(i+1,j,"ASYMM"),"modelos/"+alg+"/graficas/")

    
    
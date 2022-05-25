#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:19:57 2022

@author: francisco
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sys
from custom_funcs import extraerInfo
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from custom_funcs import load_dataset

def substring(string, ini, fin):
    return string[string.find(ini)+len(ini):string.find(fin)]

def comprobarPesos(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
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
    
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=epocas, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--dataset', type=str, default=dataset, metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST", choices={"MNIST","FMNIST"})
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    args = parser.parse_args()
            
    train_loader,test_loader = load_dataset(args.dataset, args, torch.device("cpu"), False)
    for i in range(n_layers):
        iteraciones = int(len(train_loader.dataset)/args.batch_size)
        plt.plot(list(range(len(datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0]))),datos[iteraciones*n_layers-n_layers-i::iteraciones*n_layers,0], label = "capa "+str(i+1))
        #plt.legend()
        #plt.show()
        
    plt.legend()
    plt.show()
    print("Los pesos son correctos: ", np.all(abs(datos) <= 1.1))
    f.close()
    
def extraerLossAcc(archivo):
    print(archivo)
    datos = []
    with open(archivo,'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line.split(" "))
            datos.append(list(map(float,line.split(" "))))
    datos = np.array(datos)
    print(datos)
    return datos[:,0],datos[:,1]

def graficarEvaluacion(ruta,var):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    x = np.arange(0,var["epochs"][0])
    
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    loss, acc = extraerLossAcc(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4")
                    ax[0].plot(x,loss,'*-',label="n_bits"+str(bits))
                    ax[1].plot(x,acc,'*-', label="n_bits"+str(bits))
    
    
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
    #plt.savefig("images/"+nombre)
    plt.show()
    
def graficarACC(ruta,var):
    fig = plt.figure(figsize=(8,6))
    x = np.arange(0,var["epochs"][0])
    dicc = {0:"global",1:"local"}
    colores = ["r","b","g","y","c","m","y"]
    iterador = 0
    for dataset in var["dataset"]:
        for bits in var["n_bits"]:
            for func in var["func"]:
                for glbl in var["globl"]:
                    loss, acc = extraerLossAcc(ruta+"/sinq_"+dataset+"_nbits"+str(bits)+"_epochs"+str(var["epochs"][0])+"_global"+str(glbl)+"_modo"+str(func)+"_n_layers0_hidden_width4")
                    #plt.plot(x,loss,'*-',label="n_bits"+str(bits))
                    if len(var["n_bits"]) > 1:
                        plt.plot(x,acc,'*-', label="n_bits"+str(bits))
                    if len(var["globl"]) > 1:
                        plt.plot(x,acc,'*-', label=dicc[glbl])
                        
                    """if glbl == 0:
                        plt.plot(x,acc,colores[iterador]+'*--', label="n_bits"+str(bits))
                        iterador += 1
                    else:
                        plt.plot(x,acc,colores[iterador]+'*-', label="n_bits"+str(bits))
                        iterador += 1"""
                    
    
    
    
    plt.legend(bbox_to_anchor=(1.04,1))
    plt.title("Test acc")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0,100)
    fig.tight_layout()

    #plt.savefig("images/"+nombre)
    plt.show()
                    
                
    #var es un diccionario


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='hola')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    print(type(args))
    
    """for i in glob(args.ruta+"/*_global1*"):
        print(i)
        comprobarPesos(i)"""
        #hol = input()
    #graficarACC("modelos/feedbackAlignment/historial/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[2,3,4,5,6,7,8],"func":[1],"globl":[0]})
    graficarACC("modelos/feedbackAlignment/historial/",{"epochs":[30],"dataset":["MNIST"],"n_bits":[5,6,7,8],"func":[1],"globl":[0]})
    
    
if __name__=="__main__":
    """tensor = torch.Tensor([[0,-1.1],[0,1]])
    resultado = torch.any(torch.abs(tensor)>1).numpy()
    print(tensor)
    print(resultado == True)"""
    main()
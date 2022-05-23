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
def comprobarPesos(archivo):
    f = open(archivo,"r")
    Lines = f.readlines()
    datos = []
    
    pos1 = archivo.find("n_layers")
    pos2 = archivo.find("_hidden")
    
    n_layers = int(archivo[pos1+len("n_layers"):pos2])+3
    for line in Lines:
        datos.append(list(map(float,line.split(" ")[:-1])))
        
        
    datos = np.array(datos)
    salir = False
    """for i in datos:
        for j in i:
            if abs(j) -1 >= 0.01:
                print("diferencia ", abs(j) -1)
                print(j)
                print(i)
                salir = True
                break
        if salir:
            break"""
            
        
            
    print(np.all(abs(datos)-1 <= 0.01))
    f.close()
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    
    for i in glob(args.ruta+"/*_global1*"):
        print(i)
        comprobarPesos(i)
        #hol = input()
    
    
if __name__=="__main__":
    tensor = torch.Tensor([[0,-1.1],[0,1]])
    resultado = torch.any(torch.abs(tensor)>1).numpy()
    print(tensor)
    print(resultado == True)
    #main()
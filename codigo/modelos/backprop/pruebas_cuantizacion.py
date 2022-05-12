#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:40:45 2022

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


from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


import sys
import numpy as np
sys.path.insert(1, '../../')
from custom_funcs import extraerInfo

"""import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train,test,create_backward_hooks,ASYMM,dASYMM,ASYMMf,minmax"""

"""from sklearn.cluster import KMeans, MeanShift

def clustering(tensor, n_bits):
    if 2**n_bits <= torch.numel(tensor):
        modelo = KMeans(n_clusters=2**n_bits,random_state=0).fit(tensor.reshape(-1,1))
        return torch.tensor(modelo.cluster_centers_[modelo.predict(tensor.reshape(-1,1))].flatten())
    else:
        return tensor
    
def clustering2(tensor, n_bits):
    modelo = MeanShift().fit(tensor.reshape(-1,1))
    if modelo.cluster_centers_.size <= 2**n_bits:
        return torch.tensor(modelo.cluster_centers_[modelo.predict(tensor.reshape(-1,1))].flatten())
    else:
        return clustering(tensor,n_bits)

tensor = torch.Tensor([0.3,0.6,0.9,1,0,0.4,0.7,0.2])"""

"""res1 = ASYMM(tensor, 0, 1, 2)
print(tensor)
res = ASYMMf(tensor, 0, 1, 3)
print(res)
print("Diferencia: ", torch.mean(torch.abs(tensor-res)))"""

"""tensor = torch.Tensor([1,1,1])
print(tensor - 1)"""

"""print(clustering2(tensor,2))
"""
def f():
    return [1,2],[3,4],[5,6]

def y(variable):
    print(len(variable))

variable = f()
y(variable)
for i in range(len(variable)):
    print(variable[i][0])
    
extraerInfo("datosMaxMin_n_layers3.dat")
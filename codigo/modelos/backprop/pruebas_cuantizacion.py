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

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import numpy as np


import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train,test,create_backward_hooks,ASYMM,dASYMM,ASYMMf,minmax

tensor = torch.Tensor([0.3,0.6,0.9,1,0,0.4,0.7,0.2])

res1 = ASYMM(tensor, 0, 1, 2)
print(tensor)
res = ASYMMf(tensor, 0, 1, 3)
print(res)
print("Diferencia: ", torch.mean(torch.abs(tensor-res)))

tensor = torch.Tensor([1,1,1])
print(tensor - 1)
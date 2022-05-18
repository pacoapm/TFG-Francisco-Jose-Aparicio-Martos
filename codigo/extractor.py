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




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--ruta',type=str, default=None, metavar="archivo")
    args = parser.parse_args()
    
    extraerInfo(args.ruta)
    
if __name__=="__main__":
    main()
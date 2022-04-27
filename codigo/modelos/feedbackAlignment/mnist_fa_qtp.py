#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:27:41 2022

@author: francisco
Script para crear las custom layers con cuantización
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from biotorch.benchmark.run import Benchmark
from biotorch.module.biomodule import BioModule

from mnist_fa import Net

import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,create_backward_hooks, train_loop, minmax, actualizar_pesos, visualizar_caracteristicas, load_dataset, dibujar_loss_acc
import custom_funcs



class QuantNet(nn.Module):
    def __init__(self):
        super(QuantNet, self).__init__()
        #self.round = my_round_func.apply
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28,4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4,10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.precision = 15
        

    def forward(self,x):
        x = self.flatten(x)
        #print(x)
        x = my_round_func.apply(x)
        #print(x)
        x = self.l1(x)
        #print(x)
        x = my_round_func.apply(x)
        #print(x)
        x = self.l2(self.relu(x))
        #print(x)
        x = my_round_func.apply(x)
        #print(x)
        x = self.softmax(x)
        #print(x)
        x = my_round_func.apply(x)
        #print(x)
        return x





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--global-quantization', type=int, default=1, metavar='G',
                        help="indica si se realiza la cuantizacion a nivel global (1) o local (0)")
    parser.add_argument('--n-bits', type=int, default=8, metavar='N',
                        help="numero de bits usados para la cuantizacion")
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.global_quantization == 1:
        global_quantization = True
    else:
        global_quantization = False
        
    custom_funcs.n_bits = args.n_bits

    torch.manual_seed(args.seed)

    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    train_loader,test_loader = load_dataset(args.dataset, args, device, use_cuda)

    images, labels = next(iter(train_loader))
    imagen = images[0]


    #entreno el modelo normal 
    model = Net()
    model = create_backward_hooks(model)
    
    model = BioModule(model,mode="fa")
    model = model.to(device)

    loss, acc = train_loop(model,args,device,train_loader,test_loader)


    #version cuantizada
    
    #cogemos los valores minimos y maximos de la red anterior
    minimo, maximo = minmax(model, global_quantization)
    print("minimo: ", minimo,"\nmaximo: " ,maximo)
    #creamos el modelo
    modelq = QuantNet()
    
    modelq = BioModule(modelq,mode="fa")
    modelq = create_backward_hooks(modelq)
    
    modelq = modelq.to(device)
    #cuantizamos los pesos
    actualizar_pesos(modelq,args.n_bits,minimo,maximo, global_quantization)
    #entrenamiento 
    lossq, accq = train_loop(modelq, args, device, train_loader, test_loader, True, minimo, maximo, global_quantization)
    
    #visualizar_caracteristicas(model, imagen)
    #visualizar_caracteristicas(modelq, imagen)

    nombre = "sinq_"+args.dataset+"_nbits"+str(args.n_bits)+"_epochs"+str(args.epochs)+"_global"+str(args.global_quantization)
    dibujar_loss_acc(loss,acc,args.epochs, nombre)

    nombreq = "q_"+args.dataset+"_nbits"+str(args.n_bits)+"_epochs"+str(args.epochs)+"_global"+str(args.global_quantization)
    dibujar_loss_acc(lossq,accq,args.epochs,nombreq)

    """if args.save_model:
        torch.save(model.state_dict(), "../pesosModelos/mnist_backprop.pt")"""


if __name__ == '__main__':
    main()
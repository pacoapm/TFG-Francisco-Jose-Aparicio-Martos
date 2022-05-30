#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:07:05 2022

@author: francisco
"""

from __future__ import print_function
import dni
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

from mnist_dni import Net
import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train_DNI,test,create_backward_hooks, load_dataset, train_loop_dni, one_hot, actualizar_pesos,minmax
from custom_funcs import generarNombre, dibujar_loss_acc, train_loop_dni, maximof, guardarDatos, generarInformacion, QuantLayer, linearStack, guardarHistorial
import custom_funcs
from mnist_dni import Net

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--dni', action='store_true', default=True,
                    help='enable DNI')
    parser.add_argument('--context', action='store_true', default=True,
                        help='enable context (label conditioning) in DNI')
    parser.add_argument('--global-quantization', type=int, default=1, metavar='G',
                        help="indica si se realiza la cuantizacion a nivel global (1) o local (0)")
    parser.add_argument('--n-bits', type=int, default=8, metavar='N',
                        help="numero de bits usados para la cuantizacion")
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")
    parser.add_argument('--modo', type=int, default=0, metavar='n',
                        help="indica la cuantizacion a usar: ASYMM(0) o SYMM(1)")
    parser.add_argument('--n-layers',type=int, default= 0, metavar = 'n', help = "indica la cantidad de capas ocultas de la red (sin contar la de salida)")
    parser.add_argument('--hidden-width', type=int, default = 4, metavar = 'n', help = "numero de unidades de las capas ocultas ")
    parser.add_argument('--input-width',type=int, default = 784, metavar = 'n', help = "numero de unidades de la capa de entrada")
    parser.add_argument('--output-width',type=int, default = 10, metavar = 'n', help = "numero de unidades de la capa de salida")


    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.global_quantization == 1:
        global_quantization = True
    else:
        global_quantization = False
        
    custom_funcs.n_bits = args.n_bits
    custom_funcs.modo = args.modo
    
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader,test_loader = load_dataset(args.dataset, args, device, use_cuda)

    model = Net(args)
    model = model.to(device)
    model.load_state_dict(torch.load("../../pesosModelos/"+args.dataset+"_dni.pt"))
    
    loss, acc = test(model,device, test_loader)
    modelq = Net(args)
    modelq = modelq.to(device)
    
        
    minimo = -1
    maximo = 1
        
        
    #cuantizamos los pesos
    actualizar_pesos(modelq,args.n_bits,minimo,maximo, global_quantization)
    #entrenamiento 
    lossq, accq = train_loop_dni(modelq, args, device, train_loader, test_loader, True, minimo, maximo, global_quantization, "infoPesos/"+generarNombre(args,True))
    


    nombreq = generarNombre(args,True)
    dibujar_loss_acc(lossq,accq,args.epochs,nombreq)
    
        
    guardarDatos("datos/"+args.dataset+".csv",generarInformacion(args,acc,loss,accq[-1],lossq[-1]))
    guardarHistorial("historial/"+generarNombre(args,True),lossq,accq)

if __name__ == '__main__':
    main()

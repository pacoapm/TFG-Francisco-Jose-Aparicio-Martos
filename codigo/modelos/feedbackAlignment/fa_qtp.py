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
from custom_funcs import my_round_func,create_backward_hooks, train_loop, minmax, actualizar_pesos, generarNombre, generarInformacion
from custom_funcs import  visualizar_caracteristicas, load_dataset, dibujar_loss_acc, maximof, actualizar_pesos_fa, guardarDatos, QuantNet, test, train_loop_fa, guardarHistorial
import custom_funcs



def main():
    #hola = input()
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
    parser.add_argument('--global-quantization', type=int, default=1, metavar='G',
                        help="indica si se realiza la cuantizacion a nivel global (1) o local (0)")
    parser.add_argument('--n-bits', type=int, default=8, metavar='N',
                        help="numero de bits usados para la cuantizacion")
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST o FMNIST")
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
    
    """with open("datos/"+args.dataset+".csv",'r') as f:
        dicc = {'ASYMM':0,'SYMM':1}
        info = generarInformacion(args,0,0,0,0)
        info = info.split(';')
        lines = f.readlines()
        
        for line in lines:
            linea = line.split(';')
            if linea[0:3] == info[0:3]:
                print("ya existe")
                return 0"""
        

    device = torch.device("cpu")

    train_loader,test_loader = load_dataset(args.dataset, args, device, use_cuda)

    images, labels = next(iter(train_loader))
    imagen = images[0]


    #cargamos el modelo preentrenado
    model = Net(args.n_layers,args.hidden_width,args.input_width,args.output_width)
    model = BioModule(model,mode="fa")
    model = model.to(device)
    
    model.load_state_dict(torch.load("../../pesosModelos/"+args.dataset+"_fa_n_layers"+str(args.n_layers)+"_hidden_width"+str(args.hidden_width)+".pt"))
    loss,acc = test(model,device,test_loader)

    #version cuantizada
    #creamos el modelo
    modelq = Net(args.n_layers,args.hidden_width,args.input_width,args.output_width)
    
    modelq = BioModule(modelq,mode="fa")
    
    modelq = modelq.to(device)
    
        
    minimo = -1
    maximo = 1
        
        
    #cuantizamos los pesos
    actualizar_pesos_fa(modelq,args.n_bits,minimo,maximo, global_quantization)
    with open("infoPesos/"+generarNombre(args,True),"w") as f:
        f.write("\n")
    #entrenamiento cuantizado
    lossq, accq, lossq_train, accq_train = train_loop_fa(modelq, args, device, train_loader, test_loader, True, minimo, maximo, global_quantization,"infoPesos/"+generarNombre(args,True))
    
    #visualizar_caracteristicas(model, imagen)
    #visualizar_caracteristicas(modelq, imagen)


    nombreq = generarNombre(args,True)
    dibujar_loss_acc(lossq,accq,args.epochs,nombreq)
    
    guardarDatos("datos/"+args.dataset+".csv",generarInformacion(args,acc,loss,accq[-1],lossq[-1]))
    guardarHistorial("historial/"+generarNombre(args,True),lossq,accq)
    guardarHistorial("historial_train/"+generarNombre(args,True),lossq_train,accq_train)


if __name__ == '__main__':
    main()
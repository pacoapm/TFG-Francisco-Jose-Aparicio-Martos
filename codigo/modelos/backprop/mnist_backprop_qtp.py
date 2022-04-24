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
import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train,test,create_backward_hooks, train_loop, minmax, actualizar_pesos
from mnist_backprop_visualizacion import Net


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        #self.round = my_round_func.apply
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28,4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        

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
    
class QuantNet(nn.Module):
    def __init__(self, mini, maxi, n_bits):
        super(QuantNet, self).__init__()
        #self.round = my_round_func.apply
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(28*28,4)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4,10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.mini = mini
        self.maxi = maxi
        self.nbits = n_bits
        
        

    def forward(self,x):
        x = self.flatten(x)
        
        x = my_round_func.apply(x)
        
        x = self.l1(x)
        
        x = my_round_func.apply(x)
        
        x = self.l2(self.relu(x))
       
        x = my_round_func.apply(x)
        
        x = self.softmax(x)
        
        x = my_round_func.apply(x)
        
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
    parser.add_argument('--global-quantization', type=bool, default=True, metavar='G',help="indica si se realiza la cuantizacion a nivel global")
    parser.add_argument('--n-bits', type=int, default=8, metavar='N',help="numero de bits usados para la cuantizacion")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        #media y desviación típica de la base de datos MNIST
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('../../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../../data', train=False,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    

    #version con redondeo
    
    """model = CustomNet()
    model = create_backward_hooks(model,4)
    
    model = model.to(device)"""
    model = Net()
    model = model.to(device)
    
    #falta el redondeo de los pesos
        
    train_loop(model,args,device,train_loader,test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "../pesosModelos/mnist_backprop.pt")
        
    #version cuantizada
    
    #cogemos los valores minimos y maximos de la red anterior
    if args.global_quantization:
        minimo, maximo = minmax(model)
        
        #creamos el modelo
        modelq = QuantNet(minimo, maximo, args.n_bits)
        model = create_backward_hooks(model, 0)
        modelq = modelq.to(device)
        #cuantizamos los pesos
        actualizar_pesos(modelq,args.n_bits,minimo,maximo)
        #entrenamiento 
        train_loop(modelq, args, device, train_loader, test_loader, True, minimo, maximo)
    else:
        #creamos el modelo
        modelq = QuantNet(minimo, maximo, args.n_bits)
        
        #cuantizamos los pesos
        actualizar_pesos(modelq,args.n_bits)
        #entrenamiento 
        train_loop(modelq, args, device, train_loader, test_loader, True)
        
        
    
    
    

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:41:51 2022

@author: francisco
"""
from cmath import nan
import torch
import torch.nn.functional as F
import torch.nn as nn
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
n_bits = 8

#funcion sacada de https://discuss.pytorch.org/t/torch-round-gradient/28628/5
"""class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input=input,decimals=3)
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input"""
    
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        minimo = torch.min(input)
        maximo = torch.max(input)
        return ASYMMf(input, minimo, maximo, n_bits)
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def load_dataset(dataset, args, device, use_cuda):
    if dataset == "MNIST":
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
    else:
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform=transforms.Compose([
            transforms.ToTensor()
            ])
        dataset1 = datasets.FashionMNIST('../../data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.FashionMNIST('../../data', train=False,
                           transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
    return train_loader,test_loader
    
def train(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True):
    model.train()
    output = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = torch.round(input=data,decimals=3)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if cuantizacion:
            actualizar_pesos(model, args.n_bits, minimo, maximo, glob)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    #print(output)
    
def train_DNI(args, model, device, train_loader, optimizer, epoch):
    model.train()
    output = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = torch.round(input=data,decimals=3)
        optimizer.zero_grad()
        output = model(data,target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def train_loop(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob)
        test(model, device, test_loader)
        scheduler.step()
    
    

"""def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
    return model"""
def hook(grad):
    minimo = torch.min(grad)
    maximo = torch.max(grad)
    return ASYMMf(grad,minimo,maximo,n_bits)


def create_backward_hooks( model :nn.Module) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(hook)
    return model

#funcion de cuantizacion flotante -> entero
def ASYMM(t, mini, maxi, n):
    if mini == maxi:
        print("son iguales AAHHHHHH")
        hol = input()
    return torch.round((t-mini)*((2**(n)-1)/(maxi-mini)))

#funcion de decuantizacion entero -> flotante
def dASYMM(t,mini,maxi,n):
    return t/((2**(n)-1)/(maxi-mini))+mini
#funcion de cuantizacion flotante -> flotante
def ASYMMf(t,mini,maxi,n):
    res = ASYMM(t,mini,maxi,n)
    return dASYMM(res,mini,maxi,n)

def minmax(modelo,glob = True):
    minimo = 100
    maximo = 0
    minimos = []
    maximos = []
                
    for i in modelo.parameters():
        min_capa = torch.min(i)
        max_capa = torch.max(i)
        
        minimos.append(min_capa)
        maximos.append(max_capa)
        if min_capa < minimo:
            minimo = min_capa
        if max_capa > maximo:
            maximo = max_capa
            
    if glob:
        return minimo,maximo
    else:
        return minimos,maximos

def actualizar_pesos(modelo,n_bits,minimo=None,maximo=None, glob = True):
    i = 0
    for layer in modelo.children():
        if type(layer) == nn.Linear:
            if glob:
                layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
            else:
                layer.bias.data = ASYMMf(layer.bias.data,minimo[i],maximo[i],n_bits)
                layer.weight.data = ASYMMf(layer.weight.data,minimo[i+1],maximo[i+1],n_bits)
                i+=2

def visualizar_caracteristicas(model, imagen):
    model.to(torch.device("cpu"))
    integrated_gradients = IntegratedGradients(model)
    pred_score, pred_label = torch.topk(model(imagen),1)
    pred_label.squeeze_()
    
    
    attributions_ig = integrated_gradients.attribute(imagen.unsqueeze(0), target=pred_label, n_steps = 200)
    
    
    
    # Show the original image for comparison
    plt.imshow(imagen.reshape(28,28), cmap = "gray")
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                     [(0, '#ffffff'),
                                                      (0.25, '#0000ff'),
                                                      (1, '#0000ff')], N=256)
    
    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze(0).cpu().detach().numpy(),(1,2,0)),
                                 np.transpose(imagen.cpu().detach().numpy(),(1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 title='Integrated Gradients')

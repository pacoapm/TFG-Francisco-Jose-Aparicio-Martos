#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:41:51 2022

@author: francisco
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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
        return ASYMMf(input, minimo, maximo, 8)
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
def train(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None):
    model.train()
    output = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = torch.round(input=data,decimals=3)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if cuantizacion:
            actualizar_pesos(model, args.n_bits, minimo, maximo)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    print(output)
    
def train_DNI(args, model, device, train_loader, optimizer, epoch):
    model.train()
    output = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = torch.round(input=data,decimals=3)
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
    
def train_loop(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo)
        test(model, device, test_loader)
        scheduler.step()
    
    

"""def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
    return model"""
def hook(grad):
    minimo = torch.min(grad)
    maximo = torch.max(grad)
    return ASYMMf(grad,minimo,maximo,8)


def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(hook)
    return model

def ASYMM(t, mini, maxi, n):
    return torch.round((t-mini)*((2**(n)-1)/(maxi-mini)))

def dASYMM(t,mini,maxi,n):
    return t/((2**(n)-1)/(maxi-mini))+mini

def ASYMMf(t,mini,maxi,n):
    res = ASYMM(t,mini,maxi,n)
    return dASYMM(res,mini,maxi,n)

def minmax(modelo,glob = True):
    minimo = 100
    maximo = 0
    minimos = []
    maximos = []
    for i in modelo.modules():
        print(type(i))
        if type(i) == nn.Linear:
            capa = i.weight.data
            bias = i.bias.data
            
            min_weight = torch.min(capa)
            max_weight = torch.max(capa)
            min_bias = torch.min(bias)
            max_bias = torch.max(bias)
            
            
            if min_weight <= min_bias:
                min_capa = min_weight
            else:
                min_capa = min_bias
                
            if max_weight >= max_bias:
                max_capa = max_weight
            else:
                max_capa = max_bias
                
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
            if glob == True:
                layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
            else:
                layer.weight.data = ASYMMf(layer.weight.data,minimo[i],maximo[i],n_bits)
                i+=1
    
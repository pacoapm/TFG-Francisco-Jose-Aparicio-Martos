#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:41:51 2022

@author: francisco
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


#funcion sacada de https://discuss.pytorch.org/t/torch-round-gradient/28628/5
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input=input,decimals=3)
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
def train(args, model, device, train_loader, optimizer, epoch):
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
        """print(model.linear_relu_stack[0].weight.grad)
        print(model.linear_relu_stack[2].weight.grad)"""
        #hola = input()
        optimizer.step()
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
    
    
    
def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
    return model
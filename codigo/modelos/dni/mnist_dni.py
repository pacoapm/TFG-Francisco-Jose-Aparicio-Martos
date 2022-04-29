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
import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train_DNI,test,create_backward_hooks, load_dataset, train_loop_dni, one_hot
import custom_funcs



class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28,4)
        self.layer1_f = nn.ReLU()
        self.layer2 = nn.Linear(4,10)
        self.args = args
        if self.args.dni:
            if self.args.context:
                context_dim = 10
            else:
                context_dim = None
            self.backward_interface = dni.BackwardInterface(
                dni.BasicSynthesizer(
                    output_dim=4, n_hidden=1, context_dim=context_dim
                )
            )
        

    def forward(self, x, y = None):
        x = x.view(x.size()[0], -1)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer1_f(x)
        if self.args.dni and self.training:
            if self.args.context:
                context = one_hot(y, 10, self.args)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
        x = self.layer2(x)
        
        output = F.log_softmax(x, dim=1)
        return output



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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--dni', action='store_true', default=True,
                    help='enable DNI')
    parser.add_argument('--context', action='store_true', default=True,
                        help='enable context (label conditioning) in DNI')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")



    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader,test_loader = load_dataset(args.dataset, args, device, use_cuda)

    model = Net(args)
    model = create_backward_hooks(model)
    model = model.to(device)
    
    loss, acc = train_loop_dni(model,args,device,train_loader,test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "../../pesosModelos/"+args.dataset+"_dni.pt")


if __name__ == '__main__':
    main()
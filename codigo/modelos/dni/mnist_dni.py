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
from custom_funcs import train_DNI,test
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=6, metavar='N',
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




args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    if args.no_cuda == False:
        result = result.cuda()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank,
        index=indexes.data.unsqueeze(dim=indexes_rank),
        value=1
    )
    return Variable(result)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28,4)
        self.layer1_f = nn.ReLU()
        self.layer2 = nn.Linear(4,10)
        if args.dni:
            if args.context:
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
        if args.dni and self.training:
            if args.context:
                context = one_hot(y, 10)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
        x = self.layer2(x)
        
        output = F.log_softmax(x, dim=1)
        return output



def main():
    

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

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train_DNI(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "../pesosModelos/mnist_dni.pt")


if __name__ == '__main__':
    main()
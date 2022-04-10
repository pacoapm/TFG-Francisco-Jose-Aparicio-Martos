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
import dni
from torch.autograd import Variable

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
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--dni', action='store_true', default=True,
                help='enable DNI')
parser.add_argument('--context', action='store_true', default=False,
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
    
#funcion sacada de https://discuss.pytorch.org/t/torch-round-gradient/28628/5

class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input=input,decimals=3)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
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
    
class ConvSynthesizer(nn.Module):
    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.input_context = nn.Linear(10, 20)
        self.hidden = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.output = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.output.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                                           .unsqueeze(3)
                                           .expand_as(x)
            )
        x = self.hidden(F.relu(x))
        return self.output(F.relu(x))
    
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
        x = my_round_func.apply(x)
        x = self.layer1(x)
        x = my_round_func.apply(x)
        x = self.layer1_f(x)
        x = my_round_func.apply(x)
        if args.dni and self.training:
            if args.context:
                context = one_hot(y, 10)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = self.backward_interface(x)
                x = my_round_func.apply(x)
        x = self.layer2(x)
        x = my_round_func.apply(x)
        
        x = F.log_softmax(x, dim=1)
        x = my_round_func.apply(x)
        return x


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
    
    def printgradnorm(self, grad_input, grad_output):
        if type(grad_input[0]) == torch.Tensor:
            input_gradient = (torch.round(input=grad_input[0],decimals=10),)
            return input_gradient
        
    def gradient_clipper(model: nn.Module, val: int) -> nn.Module:
        for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=val))
        
        return model
    
    def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
        for parameter in model.parameters():
                parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
        return model
    
    def forward_hook(module, inputs, outputs):
        print(outputs)
        
    
    def create_forward_hooks(model :nn.Module, decimals: int) -> nn.Module:
        for layer in model.children():
            layer.register_forward_hook(forward_hook)
            print(layer)
        return model


    model = Net()
    #model = create_backward_hooks(model,4)
    #model = create_forward_hooks(model,2)
    """model.linear_relu_stack[0].register_full_backward_hook(printgradnorm)
    model.linear_relu_stack[2].register_full_backward_hook(printgradnorm)"""
    
    model = create_backward_hooks(model, 3)
    
    model = model.to(device)
    
    for layer in model.children():
        if type(layer) == nn.Linear:
            #print(layer.weight.data)
            layer.weight.data = torch.round(input=layer.weight.data,decimals =3)
            #layer.weight = torch.round(input=layer.weight,decimals =3)"""
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "../pesosModelos/mnist_backprop.pt")


if __name__ == '__main__':
    main()
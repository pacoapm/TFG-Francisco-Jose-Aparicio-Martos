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

class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input=input,decimals=3)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

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

#copiado del propio dni para crear la version cuantizada

class BasicSynthesizerQ(torch.nn.Module):
    """Basic `Synthesizer` based on an MLP with ReLU activation.

    Args:
        output_dim: Dimensionality of the synthesized `messages`.
        n_hidden (optional): Number of hidden layers. Defaults to 0.
        hidden_dim (optional): Dimensionality of the hidden layers. Defaults to
            `output_dim`.
        trigger_dim (optional): Dimensionality of the trigger. Defaults to
            `output_dim`.
        context_dim (optional): Dimensionality of the context. If `None`, do
            not use context. Defaults to `None`.
    """

    def __init__(self, output_dim, n_hidden=0, hidden_dim=None,
                 trigger_dim=None, context_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        if trigger_dim is None:
            trigger_dim = output_dim

        top_layer_dim = output_dim if n_hidden == 0 else hidden_dim

        self.input_trigger = torch.nn.Linear(
            in_features=trigger_dim, out_features=top_layer_dim
        )

        if context_dim is not None:
            self.input_context = torch.nn.Linear(
                in_features=context_dim, out_features=top_layer_dim
            )
        else:
            self.input_context = None

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=(
                    hidden_dim if layer_index < n_hidden - 1 else output_dim
                )
            )
            for layer_index in range(n_hidden)
        ])

        # zero-initialize the last layer, as in the paper
        if n_hidden > 0:
            nn.init.constant(self.layers[-1].weight, 0)
        else:
            nn.init.constant(self.input_trigger.weight, 0)
            if context_dim is not None:
                nn.init.constant(self.input_context.weight, 0)

    def forward(self, trigger, context):
        """Synthesizes a `message` based on `trigger` and `context`.

        Args:
            trigger: `trigger` to synthesize the `message` based on. Size:
                (`batch_size`, `trigger_dim`).
            context: `context` to condition the synthesizer. Ignored if
                `context_dim` has not been specified in the constructor. Size:
                (`batch_size`, `context_dim`).

        Returns:
            The synthesized `message`.
        """
        last = my_round_func.apply(self.input_trigger(trigger))

        if self.input_context is not None:
            last += my_round_func.apply(self.input_context(context))
            
        last = my_round_func.apply(last)

        for layer in self.layers:
            last = my_round_func.apply(layer(my_round_func.apply(F.relu(last))))

        return last

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
                BasicSynthesizerQ(
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
            #print(parameter)
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
    return model


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

    model = Net()
    """for layer in model.children():
        print(layer)
    hol = input()"""
    model = create_backward_hooks(model, 3)
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "../pesosModelos/mnist_dni.pt")


if __name__ == '__main__':
    main()
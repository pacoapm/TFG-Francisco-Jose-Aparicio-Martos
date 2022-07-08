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



def linearStackDNI(input_width,output_width, args):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    
    if args.dni:
        if args.context:
            context_dim = 10
        else:
            context_dim = None
        backward_interface = dni.BackwardInterface(
            dni.BasicSynthesizer(
                output_dim=args.hidden_width, n_hidden=1, context_dim=context_dim
            )
        )
        
    if args.dni:
        return nn.Sequential(*[linear,relu,backward_interface]) 
    else:
        return nn.Sequential(*[linear,relu]) 
    
def aplicarStack(modelo,args,stack, entrada, y):
    cont = 0
    x = torch.clone(entrada)
    for layer in stack:
        cont+=1
        
        if cont == 3 and args.dni and modelo.training:
            if args.context:
                context = one_hot(y, 10, args)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = layer(x)
        else:
            x = layer(x)
                    
    return x
            

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.flatten = nn.Flatten()
        self.input_layer = linearStackDNI(args.input_width,args.hidden_width, args)
        
        blocks = []
        for i in range(args.n_layers):
            blocks.append(linearStackDNI(args.hidden_width, args.hidden_width, args))
            
        self.hidden_layers = nn.Sequential(*blocks)
        
        self.output_layer = nn.Linear(args.hidden_width,args.output_width)
        
        self.args = args
        
        

    def forward(self, x, y = None):
        x = x.view(x.size()[0], -1)
        
        x = aplicarStack(self,self.args,self.input_layer,x,y)
        for i in self.hidden_layers:
            x = aplicarStack(self,self.args,i,x,y)
            
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        #x = my_round_func.apply(x)
        return x



def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='Synthetic Gradients')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='tamaño del batch de entrenamiento (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='tamaño del batch de test (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='numero de epocas (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='inhabilita CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='frecuencia de iteraciones con las que mostrar info de entrenamiento')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")
    parser.add_argument('--n-layers',type=int, default= 0, metavar = 'n', help = "indica la cantidad de capas ocultas de la red (sin contar la de salida)")
    parser.add_argument('--hidden-width', type=int, default = 4, metavar = 'n', help = "numero de unidades de las capas ocultas ")
    parser.add_argument('--input-width',type=int, default = 784, metavar = 'n', help = "numero de unidades de la capa de entrada")
    parser.add_argument('--output-width',type=int, default = 10, metavar = 'n', help = "numero de unidades de la capa de salida")
    


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
        torch.save(model.state_dict(), "../../pesosModelos/"+args.dataset+"_dni_n_layers"+str(args.n_layers)+"_hidden_width"+str(args.hidden_width)+".pt")


if __name__ == '__main__':
    main()
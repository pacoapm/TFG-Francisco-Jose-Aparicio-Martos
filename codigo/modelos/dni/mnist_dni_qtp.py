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

from mnist_dni import Net
import sys
sys.path.insert(1, '../../')
from custom_funcs import my_round_func,train_DNI,test,create_backward_hooks, load_dataset, train_loop_dni, one_hot, actualizar_pesos,minmax
from custom_funcs import generarNombre, dibujar_loss_acc, train_loop_dni, maximof, guardarDatos, generarInformacion, QuantLayer, linearStack, guardarHistorial
import custom_funcs
from mnist_dni import Net

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
    
    
def quantLinearStackDNI(input_width,output_width, args):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    quant = QuantLayer()
    
    if args.dni:
        if args.context:
            context_dim = 10
        else:
            context_dim = None
        backward_interface = dni.BackwardInterface(
            BasicSynthesizerQ(
                output_dim=4, n_hidden=1, context_dim=context_dim
            )
        )
        
    if args.dni:
        return nn.Sequential(*[linear,quant,relu,quant,backward_interface, quant]) 
    else:
        return nn.Sequential(*[linear,quant,relu,quant]) 
    
def aplicarStack(modelo,args,stack, entrada, y):
    cont = 0
    x = torch.clone(entrada)
    for layer in stack:
        cont+=1
        
        if cont == 5 and args.dni and modelo.training:
            if args.context:
                context = one_hot(y, 10, args)
            else:
                context = None
            with dni.synthesizer_context(context):
                x = layer(x)
        else:
            x = layer(x)
                    
    return x
            

class QuantNet(nn.Module):
    def __init__(self, args):
        super(QuantNet, self).__init__()
        
        self.flatten = nn.Flatten()
        self.input_layer = quantLinearStackDNI(args.input_width,args.hidden_width, args)
        
        blocks = []
        for i in range(args.n_layers):
            blocks.append(quantLinearStackDNI(args.hidden_width, args.hidden_width, args))
            
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
        x = my_round_func.apply(x)
        return x



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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dni', action='store_true', default=True,
                    help='enable DNI')
    parser.add_argument('--context', action='store_true', default=True,
                        help='enable context (label conditioning) in DNI')
    parser.add_argument('--global-quantization', type=int, default=1, metavar='G',
                        help="indica si se realiza la cuantizacion a nivel global (1) o local (0)")
    parser.add_argument('--n-bits', type=int, default=8, metavar='N',
                        help="numero de bits usados para la cuantizacion")
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")
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

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader,test_loader = load_dataset(args.dataset, args, device, use_cuda)

    model = Net(args)
    model = model.to(device)
    model.load_state_dict(torch.load("../../pesosModelos/"+args.dataset+"_dni.pt"))
    
    loss, acc = test(model,device, test_loader)
    modelq = Net(args)
    #modelq = create_backward_hooks(modelq)
    modelq = modelq.to(device)
    #cogemos los valores minimos y maximos de la red anterior
    if custom_funcs.modo == 0:
        """minimo, maximo = minmax(model, global_quantization)
        print(minimo,maximo)
        hol =input()"""
        minimo = -1
        maximo = 1
    else:
        maximo = maximof(model, global_quantization)
        maximo = 1
        minimo = 0
        
        
    #cuantizamos los pesos
    actualizar_pesos(modelq,args.n_bits,minimo,maximo, global_quantization)
    #entrenamiento 
    lossq, accq = train_loop_dni(modelq, args, device, train_loader, test_loader, True, minimo, maximo, global_quantization, "infoPesos/"+generarNombre(args,True))
    
    
    """nombre = generarNombre(args,False)
    dibujar_loss_acc(loss,acc,args.epochs, nombre)"""

    nombreq = generarNombre(args,True)
    dibujar_loss_acc(lossq,accq,args.epochs,nombreq)
    
        
    guardarDatos("datos/"+args.dataset+".csv",generarInformacion(args,acc,loss,accq[-1],lossq[-1]))
    guardarHistorial("historial/"+generarNombre(args,True),lossq,accq)

if __name__ == '__main__':
    main()

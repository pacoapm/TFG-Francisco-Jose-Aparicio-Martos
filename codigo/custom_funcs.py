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
from torch.autograd import Variable
from biotorch.layers.fa_constructor.linear import Linear

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import dni
import numpy as np

import csv
n_bits = 8
modo = 0

#formatea la entrada para synthetic gradient
def one_hot(indexes, n_classes, args):
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

#aplica la función de cuantificacion seleccionada
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        minimo = torch.min(input)
        maximo = torch.max(input)
        if modo == 0:
            return ASYMMf(input, minimo, maximo, n_bits)
        else:
            if maximo == 0:
                return SYMMf(input,maximo,n_bits)
            else:
                return input
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
#capa lineal que aplica cuantificacion (no se usa finalmente)
class QuantLayer(nn.Module):
    def __init__(self):
        super(QuantLayer,self).__init__()
    
    def forward(self,input):
        return my_round_func.apply(input)


#crea un conjunto capa lineal capa de salida (relu)
def linearStack(input_width,output_width):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    return nn.Sequential(*[linear,relu])

def quantLinearStack(input_width,output_width):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    quant = QuantLayer()
    return nn.Sequential(*[linear,quant,relu,quant])


#clase para crear la redes neuronales
class Net(nn.Module):
    def __init__(self, n_layers, hidden_width, input_width, output_width):
        super(Net, self).__init__()
        
        self.flatten = nn.Flatten()
        self.input_layer = linearStack(input_width,hidden_width)
        blocks = []
        for i in range(n_layers):
            blocks.append(linearStack(hidden_width,hidden_width))
        self.hidden_layers = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(hidden_width,output_width)
            
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
    
#clase para crear las redes neuronales con cuantificacin en la inferencia y en los gradientes (no se aplica finalmente)
class QuantNet(nn.Module):
    def __init__(self, n_layers, hidden_width, input_width, output_width):
        super(QuantNet, self).__init__()
        
        self.flatten = nn.Flatten()
        self.input_layer = quantLinearStack(input_width,hidden_width)
        blocks = []
        for i in range(n_layers):
            blocks.append(quantLinearStack(hidden_width,hidden_width))
        self.hidden_layers = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(hidden_width,output_width)
        self.quant_output = QuantLayer()
            
        

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.quant_output(x)
        x = F.log_softmax(x, dim=1)
        x = my_round_func.apply(x)
        return x

#carga la base de datos especificada MNIST o FMNIST
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
        return train_loader,test_loader
    if dataset=="FMNIST":
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
    
#funcion que realiza el entrenamiento de la red con backprop
def train(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True, archivo = None):
    model.train()
    output = 0
    info = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if cuantizacion:
            info.append(actualizar_pesos(model, args.n_bits, minimo, maximo, glob))
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    if archivo != None:
        guardarMaxMin(archivo, info)
        
#funcion que realiza el entrenamiento de la red con synthetic gradient
def train_DNI(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True, archivo = None):
    model.train()
    output = 0
    info = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data,target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if cuantizacion:
            info.append(actualizar_pesos(model,args.n_bits,minimo,maximo,glob))
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            
    if archivo != None:
        guardarMaxMin(archivo, info)
            
#funcion que realiza el entrenamiento de la red con feedback alignment  
def train_fa(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True, archivo = None):
    model.train()
    output = 0
    info = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if cuantizacion:
            info.append(actualizar_pesos_fa(model, args.n_bits, minimo, maximo, glob))
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    if archivo != None:
        guardarMaxMin(archivo, info)

#funcion para medir la precision de los modelos en el conjunto que se pase como argumento
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)
    
#funcion que realiza el bucle de entrenamiento para backprop
def train_loop(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True, archivo = None):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    loss_list_train = []
    acc_list_train = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob, archivo)
        loss, acc = test(model, device, test_loader)
        loss_train, acc_train = test(model, device, train_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        loss_list_train.append(loss_train)
        acc_list_train.append(acc_train)
        scheduler.step()
    
    return loss_list, acc_list, loss_list_train, acc_list_train

#funcion que realiza el bucle de entrenamiento para feedback alignment
def train_loop_fa(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True, archivo = None):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    loss_list_train = []
    acc_list_train = []
    for epoch in range(1, args.epochs + 1):
        train_fa(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob, archivo)
        loss, acc = test(model, device, test_loader)
        loss_train, acc_train = test(model, device, train_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        loss_list_train.append(loss_train)
        acc_list_train.append(acc_train)
        scheduler.step()
    
    return loss_list, acc_list, loss_list_train, acc_list_train

#funcion que realiza el bucle de entrenamiento para synthetic gradient
def train_loop_dni(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True, archivo = None):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    loss_list_train = []
    acc_list_train = []
    for epoch in range(1, args.epochs + 1):
        train_DNI(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob, archivo)
        loss, acc = test(model, device, test_loader)
        loss_train, acc_train = test(model, device, train_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        loss_list_train.append(loss_train)
        acc_list_train.append(acc_train)
        scheduler.step()
    
    return loss_list, acc_list, loss_list_train, acc_list_train

#hook usado para cuantificar los gradientes (no se usa)
def hook(grad):
    if modo == 0:
        minimo = torch.min(grad)
        maximo = torch.max(grad)
        return ASYMMf(grad,minimo,maximo,n_bits)
    elif modo == 1:
        maximo = torch.max(torch.abs(grad))
        return SYMMf(grad,maximo,n_bits)
    else:
        return my_round_func.apply(grad)
    
#hook para imprimir los gradientes (se uso en su momento para depurar)
def hook_print(grad):
    if modo == 0:
        minimo = torch.min(grad)
        maximo = torch.max(grad)
        return ASYMMf(grad,minimo,maximo,n_bits)
    elif modo == 1:
        maximo = torch.max(torch.abs(grad))
        return SYMMf(grad,maximo,n_bits)
    else:
        print("ey")
        return my_round_func.apply(grad)
        

#funcion para aplicar los hooks(en desuso)
def create_backward_hooks( model :nn.Module) -> nn.Module:
    for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(hook)
    return model

#funcion para aplicar los hooks(en desuso)
def create_backward_hooks_print( model :nn.Module) -> nn.Module:
    for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(hook_print)
    return model

#funcion de cuantizacion flotante -> entero
def ASYMM(t, mini, maxi, n):
    return torch.round((t-mini)*((2**(n)-1)/(maxi-mini)))

#funcion de decuantizacion entero -> flotante
def dASYMM(t,mini,maxi,n):
    return t/((2**(n)-1)/(maxi-mini))+mini
#funcion de cuantizacion flotante -> flotante
def ASYMMf(t,mini,maxi,n):
    t_mod = torch.clamp(t,min=mini,max=maxi)
    
    if maxi == mini:
        return t_mod
    res = ASYMM(t_mod,mini,maxi,n)
           
    return dASYMM(res,mini,maxi,n)

#comprobacion de que los pesos del tensor se encuentran entre -1 y 1 (se uso en fase de depuracion)
def correcto(tensor):
    return torch.all(torch.abs(tensor)<=1).numpy()


def SYMM(t,maxi,n):
    return torch.round(t*((2**(n-1)-1)/maxi))

def dSYMM(t,maxi,n):
    return t*(maxi/(2**(n-1)-1))

def SYMMf(t,maxi,n):
    if maxi == 0:
        return t
    t = torch.clamp(t,min=-maxi,max=maxi)
    res = SYMM(t,maxi,n)
    return dSYMM(res,maxi,n)


#encuentra minimos y maximos de los pesos de los modelos (en desuso)
def minmax(modelo,glob = True):
    minimo = 100
    maximo = 0
    minimos = []
    maximos = []
    contador = 0       
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
    
#encuentra el maximo de los pesos de un modelo (en desuso)
def maximof(modelo,glob = True):
    maxi = 0
    maximos = []
    
    for i in modelo.parameters():
        
        max_actual = torch.max(torch.abs(i))
        
        maximos.append(max_actual)
        
        if max_actual > maxi:
            maxi = max_actual
            
    
            
    if glob:
        return maxi
    else:
        return maximos

#funcion que aplica la funcion de cuantificacion a la matriz de pesos, backprop
def actualizar_pesos(modelo,n_bits,minimo=None,maximo=None, glob = True):
    
    max_bias = []
    min_bias = []
    max_weight = []
    min_weight = []
    for layer in modelo.modules():
        if type(layer) == nn.Linear or isinstance(layer,Linear): 
            
            if glob:
                if modo == 0:
                    
                    layer.bias.data = ASYMMf(layer.bias.data,minimo,maximo,n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    
                    
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    
                    
                elif modo == 1:
                    
                    
                    layer.bias.data = SYMMf(layer.bias.data,maximo,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo,n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                
                    
                    
            else:
                if modo == 0:
                    
                    
                    layer.bias.data = ASYMMf(layer.bias.data,torch.min(layer.bias.data),torch.max(layer.bias.data),n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,torch.min(layer.weight.data),torch.max(layer.weight.data),n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    
                elif modo == 1:
                    
                    max_bias_abs = torch.max(torch.abs(layer.bias.data))
                    
                    max_weight_abs = torch.max(torch.abs(layer.weight.data))
                    
                    layer.bias.data = SYMMf(layer.bias.data,max_bias_abs,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,max_weight_abs,n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
              
    
    return max_bias, min_bias, max_weight, min_weight
                    
#funcion que aplica la funcion de cuantificacion a la matriz de pesos, feedback alignment
def actualizar_pesos_fa(modelo,n_bits,minimo=None,maximo=None, glob = True):
    max_bias = []
    min_bias = []
    max_weight = []
    min_weight = []
    max_bias_back = []
    min_bias_back = []
    min_weight_back = []
    max_weight_back = []
    for layer in modelo.modules():
        if type(layer) == nn.Linear or isinstance(layer,Linear): 
            if glob:
                if modo == 0:
                    
                    
                    layer.bias.data = ASYMMf(layer.bias.data,minimo,maximo,n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
                    layer.weight_backward.data = ASYMMf(layer.weight_backward.data,minimo,maximo,n_bits)
                    layer.bias_backward.data = ASYMMf(layer.bias_backward.data,minimo,maximo,n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    max_bias_back.append(torch.max(layer.bias_backward.data))
                    min_bias_back.append(torch.min(layer.bias_backward.data))
                    max_weight_back.append(torch.max(layer.weight_backward.data))
                    min_weight_back.append(torch.min(layer.weight_backward.data))
                else:
                    
                    
                    layer.bias.data = SYMMf(layer.bias.data,maximo,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo,n_bits)
                    layer.weight_backward.data = SYMMf(layer.weight_backward.data,maximo,n_bits)
                    layer.bias_backward.data = SYMMf(layer.bias_backward.data,maximo,n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    
                    max_bias_back.append(torch.max(layer.bias_backward.data))
                    min_bias_back.append(torch.min(layer.bias_backward.data))
                    
                    max_weight_back.append(torch.max(layer.weight_backward.data))
                    min_weight_back.append(torch.min(layer.weight_backward.data))
                    
            else:
                if modo == 0:
                    
                    
                    layer.bias.data = ASYMMf(layer.bias.data,torch.min(layer.bias.data),torch.max(layer.bias.data),n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,torch.min(layer.weight.data),torch.max(layer.weight.data),n_bits)
                    layer.bias_backward.data = ASYMMf(layer.bias_backward.data,torch.min(layer.bias_backward.data),torch.max(layer.bias_backward.data),n_bits)
                    layer.weight_backward.data = ASYMMf(layer.weight_backward.data,torch.min(layer.weight_backward.data),torch.max(layer.weight_backward.data),n_bits)
                    
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    max_bias_back.append(torch.max(layer.bias_backward.data))
                    min_bias_back.append(torch.min(layer.bias_backward.data))
                    max_weight_back.append(torch.max(layer.weight_backward.data))
                    min_weight_back.append(torch.min(layer.weight_backward.data))
                    
                else:
                    
                    max_bias_abs = torch.max(torch.abs(layer.bias.data))
                    
                    max_weight_abs = torch.max(torch.abs(layer.weight.data))
                    
                    max_bias_back_abs = torch.max(torch.abs(layer.bias_backward.data))
                    
                    max_weight_back_abs = torch.max(torch.abs(layer.weight_backward.data))
                    
                    layer.bias.data = SYMMf(layer.bias.data,max_bias_abs,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,max_weight_abs,n_bits)
                    layer.bias_backward.data = SYMMf(layer.bias_backward.data,max_bias_back_abs,n_bits)
                    layer.weight_backward.data = SYMMf(layer.weight_backward.data,max_weight_back_abs,n_bits)
                    
                    max_weight.append(torch.max(layer.weight.data))
                    min_weight.append(torch.min(layer.weight.data))
                    max_bias.append(torch.max(layer.bias.data))
                    min_bias.append(torch.min(layer.bias.data))
                    max_bias_back.append(torch.max(layer.bias_backward.data))
                    min_bias_back.append(torch.min(layer.bias_backward.data))
                    max_weight_back.append(torch.max(layer.weight_backward.data))
                    min_weight_back.append(torch.min(layer.weight_backward.data))
                    
    return max_bias, min_bias, max_weight, min_weight, max_bias_back, min_bias_back, max_weight_back, min_weight_back        
        
#funcion que dada una imagen muestra en que zonas se fija nuestro modelo (no se ha usado finalmente, pero es funcional)
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

#grafica la evolucion de la precison y de la funcion de perdida del modelo entrenado
def dibujar_loss_acc(loss,acc,epochs,nombre):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    x = np.arange(0,epochs)
    if max(loss) == nan:
        ax[0].plot(x,loss,'*-')
        ax[0].set_title("Test loss")
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("Loss")
        ax[0].set_ylim(0,max(loss)+1)
        #ax[0].set_xlim(0,epochs-1)

    ax[1].plot(x,acc,'*-')
    ax[1].set_title("Test acc")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0,100)
    #ax[1].set_xlim(0,epochs-1)


    plt.savefig("images/"+nombre)
    #plt.show()
    
#genera el nombre del archivo segun los parametros de la experimentacion
def generarNombre(args, quantize):
    
    if quantize:
        nombre =  "sinq_"
    else:
        nombre =  "q_"
    
    return nombre + args.dataset+"_nbits"+str(args.n_bits)+"_epochs"+str(args.epochs)+"_global"+str(args.global_quantization)+"_modo"+str(args.modo)+"_n_layers"+str(args.n_layers)+"_hidden_width"+str(args.hidden_width)

#genera la linea a insertar en el directorio datos
def generarInformacion(args, acc, loss, accq, lossq):
    if args.modo == 0:
        modo = "ASYMM"
    else:
        modo = "SYMM"
        
    if args.global_quantization:
        globalq = "global"
    else:
        globalq = "local"
    
    return str(args.n_bits)+";"+globalq+";"+modo+";"+str(acc)+";"+str(loss)+";"+str(accq)+";"+str(lossq)+";"+str(accq-acc)+"\n"
            
#guarda los resultados en el archivo especificado
def guardarDatos(archivo, informacion):
    with open(archivo,'a') as f:
        f.write(informacion)
        #writer.writerow(informacion)
  
#guarda la evolucion del entrenamiento en archivo
def guardarHistorial(archivo,loss,acc):
    with open(archivo,'w') as f:
        for i,j in zip(loss,acc):
            f.write(str(i)+" "+str(j)+"\n")
            
#guarda la informacion de los pesos en archivo
def guardarMaxMin(archivo,informacion):
    with open(archivo,"a") as f:
        for info in informacion:
            for i in range(0,len(info[0])):
                for j in range(0,len(info)):
                    f.write(str(info[j][i].item()) + " ")
                f.write("\n")
 


        
    
    
    
    


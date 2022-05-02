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
    
class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        minimo = torch.min(input)
        maximo = torch.max(input)
        if modo == 0:
            return ASYMMf(input, minimo, maximo, n_bits)
        else:
            return SYMMf(input,maximo,n_bits)
        #return int_quant(scale, zero_point, bit_width, input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class QuantLayer(nn.Module):
    def __init__(self):
        super(QuantLayer,self).__init__()
    
    def forward(self,input):
        return my_round_func.apply(input)

    
def linearStack(input_width,output_width):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    return nn.Sequential(*[linear,relu])

def quantLinearStack(input_width,output_width):
    linear = nn.Linear(input_width,output_width)
    relu = nn.ReLU()
    quant = QuantLayer()
    return nn.Sequential(*[linear,quant,relu,quant])



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
        x = my_round_func.apply(x)
        return x
    
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
    
def train_DNI(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True):
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
        if cuantizacion:
            actualizar_pesos(model,args.n_bits,minimo,maximo,glob)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            
            
def train_fa(args, model, device, train_loader, optimizer, epoch, cuantizacion = False, minimo = None, maximo = None, glob = True):
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
            actualizar_pesos_fa(model, args.n_bits, minimo, maximo, glob)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    #print(output)


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

    return test_loss, 100. * correct / len(test_loader.dataset)
    
def train_loop(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob)
        loss, acc = test(model, device, test_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        scheduler.step()
    
    return loss_list, acc_list

def train_loop_fa(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        train_fa(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob)
        loss, acc = test(model, device, test_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        scheduler.step()
    
    return loss_list, acc_list

def train_loop_dni(model, args, device, train_loader, test_loader, cuantizacion = False, minimo = None, maximo = None,  glob = True):
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss_list = []
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        train_DNI(args, model, device, train_loader, optimizer, epoch, cuantizacion, minimo, maximo, glob)
        loss, acc = test(model, device, test_loader)
        loss_list.append(loss)
        acc_list.append(acc)
        scheduler.step()
    
    return loss_list, acc_list


"""def create_backward_hooks( model :nn.Module, decimals: int) -> nn.Module:
    for parameter in model.parameters():
            parameter.register_hook(lambda grad: torch.round(input=grad,decimals=decimals))
    return model"""
def hook(grad):
    
    if modo == 0:
        minimo = torch.min(grad)
        maximo = torch.max(grad)
        return ASYMMf(grad,minimo,maximo,n_bits)
    else:
        maximo = torch.max(torch.abs(grad))
        return SYMMf(grad,maximo,n_bits)
        


def create_backward_hooks( model :nn.Module) -> nn.Module:
    for parameter in model.parameters():
            if parameter.requires_grad:
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
    if maxi == mini:
        return t
    res = ASYMM(t,mini,maxi,n)
    return dASYMM(res,mini,maxi,n)

def SYMM(t,maxi,n):
    return torch.round(t*((2**(n-1)-1)/maxi))

def dSYMM(t,maxi,n):
    return t*(maxi/(2**(n-1)-1))

def SYMMf(t,maxi,n):
    if maxi == 0:
        """print("t ",t)
        print("maxi ", maxi)
        hol = input()"""
        return t
    res = SYMM(t,maxi,n)
    return dSYMM(res,maxi,n)

def minmax(modelo,glob = True):
    minimo = 100
    maximo = 0
    minimos = []
    maximos = []
    contador = 0       
    for i in modelo.parameters():
        """contador += 1
        print(i)
        hol = input()"""
        min_capa = torch.min(i)
        max_capa = torch.max(i)
        
        minimos.append(min_capa)
        maximos.append(max_capa)
        if min_capa < minimo:
            minimo = min_capa
        if max_capa > maximo:
            maximo = max_capa
            
    #print("en total hay ", contador, " parametros")
    if glob:
        return minimo,maximo
    else:
        return minimos,maximos
    
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

def actualizar_pesos(modelo,n_bits,minimo=None,maximo=None, glob = True):
    i = 0
    for layer in modelo.modules():
        """print(layer)
        hol = input()"""
        if type(layer) == nn.Linear or isinstance(layer,Linear): 
            """print("cuantizado")
            hol = input()"""
            if glob:
                if modo == 0:
                    layer.bias.bias = ASYMMf(layer.bias.data,minimo,maximo,n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
                    
                elif modo == 1:
                    layer.bias.data = SYMMf(layer.bias.data,maximo,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo,n_bits)
                else:
                    layer.bias.data = my_round_func.apply(layer.bias.data)
                    layer.weight.data = my_round_func.apply(layer.weight.data)
                    
                    
            else:
                if modo == 0:
                    layer.bias.data = ASYMMf(layer.bias.data,minimo[i+1],maximo[i+1],n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo[i],maximo[i],n_bits)
                    i+=2
                elif modo == 1:
                    layer.bias.data = SYMMf(layer.bias.data,maximo[i+1],n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo[i],n_bits)
                    i+= 2
                else:
                    layer.bias.data = my_round_func.apply(layer.bias.data)
                    layer.weight.data = my_round_func.apply(layer.weight.data)
                    
def actualizar_pesos_fa(modelo,n_bits,minimo=None,maximo=None, glob = True):
    i = 0
    for layer in modelo.modules():
        """print(layer)
        hol = input()"""
        if type(layer) == nn.Linear or isinstance(layer,Linear): 
            """print("cuantizado")
            hol = input()"""
            if glob:
                if modo == 0:
                    layer.bias.bias = ASYMMf(layer.bias.data,minimo,maximo,n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo,maximo,n_bits)
                    layer.weight_backward.data = ASYMMf(layer.weight_backward.data,minimo,maximo,n_bits)
                    layer.bias_backward.data = ASYMMf(layer.bias_backward.data,minimo,maximo,n_bits)
                else:
                    layer.bias.data = SYMMf(layer.bias.data,maximo,n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo,n_bits)
                    layer.weight_backward.data = SYMMf(layer.weight_backward.data,maximo,n_bits)
                    layer.bias_backward.data = SYMMf(layer.bias_backward.data,maximo,n_bits)
            else:
                if modo == 0:
                    layer.bias.data = ASYMMf(layer.bias.data,minimo[i+1],maximo[i+1],n_bits)
                    layer.weight.data = ASYMMf(layer.weight.data,minimo[i],maximo[i],n_bits)
                    layer.weight_backward.data = ASYMMf(layer.weight_backward.data,minimo[i+2],maximo[i+2],n_bits)
                    layer.bias_backward.data = ASYMMf(layer.bias_backward.data,minimo[i+3],maximo[i+3],n_bits)
                    i+=4
                else:
                    layer.bias.data = SYMMf(layer.bias.data,maximo[i+1],n_bits)
                    layer.weight.data = SYMMf(layer.weight.data,maximo[i],n_bits)
                    layer.weight_backward.data = SYMMf(layer.weight_backward.data,maximo[i+2],n_bits)
                    layer.bias_backward.data = SYMMf(layer.bias_backward.data,maximo[i+3],n_bits)
                    i+= 4

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

def dibujar_loss_acc(loss,acc,epochs,nombre):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    
    x = np.arange(0,epochs)
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


    #plt.savefig("images/"+nombre)
    plt.show()
    
def generarNombre(args, quantize):
    
    if quantize:
        nombre =  "sinq_"
    else:
        nombre =  "q_"
    
    return nombre + args.dataset+"_nbits"+str(args.n_bits)+"_epochs"+str(args.epochs)+"_global"+str(args.global_quantization)+"_modo"+str(args.modo)

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
            
    
def guardarDatos(archivo, informacion):
    with open(archivo,'a') as f:
        f.write(informacion)
        #writer.writerow(informacion)
    


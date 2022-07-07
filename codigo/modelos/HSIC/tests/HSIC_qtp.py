from hsicbt.core.train_hsic import hsic_train, quant_hsic_train
from hsicbt.core.train_standard import standard_train, quant_standard_train
from hsicbt.model.mhlinear import ModelLinear
from hsicbt.utils.dataset import get_dataset_from_code
from hsicbt.utils.misc import get_accuracy_epoch
from hsicbt.utils.misc import get_accuracy_hsic
from hsicbt.model.mvanilla import ModelVanilla, QuantModelVanilla
from hsicbt.model.mensemble import ModelEnsemble
import torch
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt
import numpy as np

from source.hsicbt.model.mhlinear import ModelQuantLinear
import sys
sys.path.insert(1, '/home/francisco/Documentos/ingenieria_informatica/cuarto_informatica/segundo_cuatri/TFG/TFG-Francisco-Jose-Aparicio-Martos/codigo')
from custom_funcs import load_dataset,create_backward_hooks, create_backward_hooks_print, minmax, maximof, actualizar_pesos, dibujar_loss_acc, generarNombre, guardarDatos, generarInformacion, guardarHistorial
import custom_funcs
from torch.optim.lr_scheduler import StepLR


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hidden = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc = 100. * correct / len(test_loader.dataset)
    loss = test_loss

    return acc, loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if args.global_quantization == 1:
        global_quantization = True
    else:
        global_quantization = False


    custom_funcs.n_bits = args.n_bits
    custom_funcs.modo = args.modo


    # # # configuration
    config_dict = {}
    config_dict['batch_size'] = args.batch_size
    config_dict['learning_rate'] = args.lr#0.001
    config_dict['lambda_y'] = 500#100
    config_dict['sigma'] = 5#2
    config_dict['task'] = 'hsic-train'
    config_dict['device'] = 'cuda'
    config_dict['log_batch_interval'] = 10
    config_dict['epochs'] = 5

    

    # # # data prepreation
    train_loader, test_loader = get_dataset_from_code(args.dataset.lower(), args.batch_size)

    #cargamos el modelo sin cuantizar


    #creamos la arquitectura del modelo y cargamos los pesos del modelo sin cuantizar
    model = ModelLinear(hidden_width=args.hidden_width,
                        n_layers=args.n_layers,
                        atype='relu',
                        last_hidden_width=args.output_width,
                        model_type='simple-dense',
                        data_code=args.dataset.lower())

    final_layer = ModelVanilla(args.output_width)
    final_model = ModelEnsemble(model,final_layer)
    final_model = final_model.to(device)

    final_model.load_state_dict(torch.load('/home/francisco/Documentos/ingenieria_informatica/cuarto_informatica/segundo_cuatri/TFG/TFG-Francisco-Jose-Aparicio-Martos/codigo/pesosModelos/'+args.dataset+"_HSIC_n_layers"+str(args.n_layers)+"_hidden_width"+str(args.hidden_width)+".pt"))
    
    acc,loss = test(final_model,device,test_loader)


    #creamos el modelo cuantizado
    modelq = ModelLinear(hidden_width=args.hidden_width,
                        n_layers=args.n_layers,
                        atype='relu',
                        last_hidden_width=args.output_width,
                        model_type='simple-dense',
                        data_code=args.dataset.lower())
    
    
    modelq = modelq.to(device)

        
    maximo = 1
    minimo = -1

    #cuantizamos los pesos del modelo
    actualizar_pesos(modelq,args.n_bits,minimo,maximo, global_quantization)
    #UNFORMATED TRAINING: entrenamiento de la red con HSIC
    epochs = 30
    with open("infoPesos/"+generarNombre(args,True),"w") as f:
        f.write("\n")
    for cepoch in range(epochs):
        quant_hsic_train(cepoch, modelq, train_loader, config_dict, args, minimo, maximo, global_quantization, "infoPesos/"+generarNombre(args,True))

    #FORMATED TRAINING: entrenamiento de la ultima capa con sgd
    final_layerq = ModelVanilla(args.output_width)#QuantModelVanilla(args.output_width)

    modelq.eval() 
    final_modelq = ModelEnsemble(modelq,final_layerq)

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, final_layerq.parameters()),
                lr = config_dict['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


    epochs = 20
    vacc = []
    vloss = []
    
    for cepoch in range(epochs):
        quant_standard_train(cepoch, final_modelq, train_loader, optimizer,config_dict, args, minimo, maximo, global_quantization, "infoPesos/"+generarNombre(args,True))
        accq, lossq = test(final_modelq,device,test_loader)
        vacc.append(accq)
        vloss.append(lossq)
        scheduler.step()

    
    nombreq = generarNombre(args,True)
    dibujar_loss_acc(vloss,vacc,20,nombreq)
    
        
    guardarDatos("datos/"+args.dataset+".csv",generarInformacion(args,acc,loss,vacc[-1],vloss[-1]))
    guardarHistorial("historial/"+generarNombre(args,True),vloss,vacc)
if __name__ == '__main__':
    main()










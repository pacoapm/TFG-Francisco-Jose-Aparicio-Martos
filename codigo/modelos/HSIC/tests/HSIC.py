from hsicbt.core.train_hsic import hsic_train
from hsicbt.core.train_standard import standard_train
from hsicbt.model.mhlinear import ModelLinear
from hsicbt.utils.dataset import get_dataset_from_code
from hsicbt.utils.misc import get_accuracy_epoch
from hsicbt.utils.misc import get_accuracy_hsic
from hsicbt.model.mvanilla import ModelVanilla
from hsicbt.model.mensemble import ModelEnsemble
import torch
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import StepLR




def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    output = 0
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = torch.round(input=data,decimals=3)
        optimizer.zero_grad()
        output, hidden = model(data)
        loss = cross_entropy_loss(output,target)#F.nll_loss(output, target)
        loss.backward()
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
    acc = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, hidden = model(data)
            """print(type(output))
            print("tamaño: ", len(output))
            print("output[0]: ", output[0])
            print("output[1]: ", output[1])
            hola = input()"""
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

def dibujar_loss_acc(loss,acc,epochs):
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
    parser.add_argument('--dataset', type=str, default='FMNIST', metavar='d',
                        help="indica la base de datos a usar: MNIST O FMNIST")
    parser.add_argument('--modo', type=int, default=0, metavar='n',
                        help="indica la cuantizacion a usar: ASYMM(0) o SYMM(1)")

    parser.add_argument('--n-layers',type=int, default= 0, metavar = 'n', help = "indica la cantidad de capas ocultas de la red (sin contar la de salida)")
    parser.add_argument('--hidden-width', type=int, default = 4, metavar = 'n', help = "numero de unidades de las capas ocultas ")
    parser.add_argument('--input-width',type=int, default = 784, metavar = 'n', help = "numero de unidades de la capa de entrada")
    parser.add_argument('--output-width',type=int, default = 10, metavar = 'n', help = "numero de unidades de la capa de salida")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    # # # configuration
    config_dict = {}
    config_dict['batch_size'] = 128
    config_dict['learning_rate'] = args.lr#0.001
    config_dict['lambda_y'] = 500#100
    config_dict['sigma'] = 5#2
    config_dict['task'] = 'hsic-train'
    config_dict['device'] = 'cuda'
    config_dict['log_batch_interval'] = 10
    config_dict['epochs'] = 5

    # # # data prepreation
    train_loader, test_loader = get_dataset_from_code(args.dataset.lower(), 128)

    # # # simple fully-connected model
    model = ModelLinear(hidden_width=args.hidden_width,
                        n_layers=args.n_layers,
                        atype='relu',
                        last_hidden_width=args.output_width,
                        model_type='simple-dense',
                        data_code=args.dataset.lower())

    """model = ModelLinear(hidden_width=256,
                        n_layers=5,
                        atype='relu',
                        last_hidden_width=10,
                        model_type='simple-dense',
                        data_code='mnist')"""

    final_layer = ModelVanilla(args.output_width)
    final_layer = final_layer.to(torch.device("cuda"))
    


    #UNFORMATED TRAINING: entrenamiento de la red con HSIC
    epochs = 30
    for cepoch in range(epochs):
        hsic_train(cepoch, model, train_loader, config_dict)

    


    #FORMATED TRAINING: entrenamiento de la ultima capa con sgd
    model.eval()
    final_model = ModelEnsemble(model,final_layer)
    final_model = final_model.to(torch.device("cuda"))

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, final_layer.parameters()),
                lr = config_dict['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    epochs = 20
    vacc = []
    vloss = []
    for cepoch in range(epochs):
        #train(args, model, torch.device('cuda'), train_loader, optimizer, cepoch)
        standard_train(cepoch, final_model, train_loader, optimizer, config_dict)
        acc, loss = test(final_model,torch.device("cuda"),test_loader)
        vacc.append(acc)
        vloss.append(loss)
        scheduler.step()

    if args.save_model:
        torch.save(final_model.state_dict(),"/home/francisco/Documentos/ingenieria_informatica/cuarto_informatica/segundo_cuatri/TFG/TFG-Francisco-Jose-Aparicio-Martos/codigo/pesosModelos/"+args.dataset+"_HSIC.pt")

    dibujar_loss_acc(vloss,vacc,epochs)

if __name__ == '__main__':
    main()










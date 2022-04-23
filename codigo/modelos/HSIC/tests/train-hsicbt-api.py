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
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    output = 0
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = torch.round(input=data,decimals=3)
        optimizer.zero_grad()
        output, hidden = model(data)
        #print(output)
        loss = cross_entropy_loss(output,target)#F.nll_loss(output, target)
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

# # # configuration
config_dict = {}
config_dict['batch_size'] = 128
config_dict['learning_rate'] = 0.001
config_dict['lambda_y'] = 100
config_dict['sigma'] = 2
config_dict['task'] = 'hsic-train'
config_dict['device'] = 'cuda'
config_dict['log_batch_interval'] = 10
config_dict['epochs'] = 5

# # # data prepreation
train_loader, test_loader = get_dataset_from_code('mnist', 128)

# # # simple fully-connected model
"""model = ModelLinear(hidden_width=4,
                    n_layers=1,
                    atype='relu',
                    last_hidden_width=10,
                    model_type='simple-dense',
                    data_code='mnist')"""

model = ModelLinear(hidden_width=256,
                    n_layers=5,
                    atype='relu',
                    last_hidden_width=10,
                    model_type='simple-dense',
                    data_code='mnist')

final_layer = ModelVanilla(hidden_width=10)
final_layer = final_layer.to(torch.device("cuda"))

"""# # # start to train
epochs = 5
for cepoch in range(epochs):
    # you can also re-write hsic_train function
    hsic_train(cepoch, model, train_loader, config_dict)
    #test_acc, reord = get_accuracy_hsic(model,test_loader) falla
    #print("Precision: ", test_acc)"""

#UNFORMATED TRAINING: entrenamiento de la red con HSIC
epochs = 10
for cepoch in range(epochs):
    # you can also re-write hsic_train function
    hsic_train(cepoch, model, train_loader, config_dict)
    """train_acc, reordered = get_accuracy_hsic(model=model, dataloader=train_loader)
    test_acc, reordered = get_accuracy_hsic(model=model, dataloader=test_loader)"""
    #test_acc, reord = get_accuracy_hsic(model,test_loader) falla
    #print("Precision: ", test_acc)

model.eval()
final_model = ModelEnsemble(model,final_layer)
final_model = final_model.to(torch.device("cuda"))

optimizer = torch.optim.SGD( filter(lambda p: p.requires_grad, final_layer.parameters()),
            lr = config_dict['learning_rate'], weight_decay=0.001)

train_loader, test_loader = get_dataset_from_code('mnist', 128)

epochs = 5
for cepoch in range(epochs):
    #train(args, model, torch.device('cuda'), train_loader, optimizer, cepoch)
    standard_train(cepoch, final_model, train_loader, optimizer, config_dict)
    test(final_model,torch.device("cuda"),test_loader)




#FORMATED TRAINING: entrenamiento de la ultima capa con sgd





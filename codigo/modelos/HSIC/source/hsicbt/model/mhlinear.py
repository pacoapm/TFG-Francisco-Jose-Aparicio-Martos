from .. import *
from .block import *
from ..utils.misc import *

import sys
sys.path.insert(1, '/home/francisco/Documentos/ingenieria_informatica/cuarto_informatica/segundo_cuatri/TFG/TFG-Francisco-Jose-Aparicio-Martos/codigo')
from custom_funcs import my_round_func,train_DNI,test,create_backward_hooks, load_dataset, train_loop_dni, one_hot, actualizar_pesos,minmax
from custom_funcs import generarNombre, dibujar_loss_acc, train_loop_dni, maximof, guardarDatos, generarInformacion
import custom_funcs


class ModelLinear(nn.Module):

    def __init__(self, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, model_type='simple-dense', data_code='mnist', **kwargs):
        super(ModelLinear, self).__init__()
    
        block_list = []
        is_conv = False
        
        last_hw = hidden_width
        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            block = get_primative_block('simple-dense', hidden_width, hidden_width, atype)
            block_list.append(block)

        in_width = get_in_dimensions(data_code)
        in_ch = get_in_channels(data_code)

        self.input_layer    = makeblock_dense(in_width*in_ch, hidden_width, atype)
        self.sequence_layer = nn.Sequential(*block_list)
        self.output_layer   = makeblock_dense(hidden_width, last_hw, atype)

        self.is_conv = is_conv
        self.in_width = in_width*in_ch

    def forward(self, x):

        output_list = []
        x = x.view(-1, self.in_width)
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list

class ModelQuantLinear(nn.Module):
    
    def __init__(self, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, model_type='simple-dense', data_code='mnist', **kwargs):
        super(ModelQuantLinear, self).__init__()
    
        block_list = []
        is_conv = False
        
        last_hw = hidden_width
        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            block = get_primative_quant_block('simple-dense', hidden_width, hidden_width, atype)
            block_list.append(block)

        in_width = get_in_dimensions(data_code)
        in_ch = get_in_channels(data_code)

        self.input_layer    = makeQuantblock_dense(in_width*in_ch, hidden_width, atype)
        self.sequence_layer = nn.Sequential(*block_list)
        self.output_layer   = makeQuantblock_dense(hidden_width, last_hw, atype)

        self.is_conv = is_conv
        self.in_width = in_width*in_ch

    def forward(self, x):

        output_list = []
        x = x.view(-1, self.in_width)
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list

"""
class ModelLinear(nn.Module):
    
    def __init__(self, hidden_width=64, n_layers=5, atype='relu', 
        last_hidden_width=None, model_type='simple-dense', data_code='mnist', **kwargs):
        super(ModelLinear, self).__init__()
    
        block_list = []
        is_conv = False
        
        last_hw = hidden_width
        if last_hidden_width:
            last_hw = last_hidden_width
        
        for i in range(n_layers):
            block = get_primative_block('simple-dense', hidden_width, hidden_width, atype)
            block_list.append(block)

        in_width = get_in_dimensions(data_code)
        in_ch = get_in_channels(data_code)

        self.flatten = nn.Flatten()
        self.input_layer    = makeblock_dense(in_width*in_ch, hidden_width, atype)
        self.sequence_layer = nn.Sequential(*block_list)
        self.output_layer   = makeblock_dense(hidden_width, last_hw, "logsoftmax")

        self.is_conv = is_conv
        self.in_width = in_width*in_ch

    def forward(self, x):

        output_list = []
        #x = x.view(-1, self.in_width)
        x = self.flatten(x)
        x = self.input_layer(x)
        output_list.append(x)
        
        for block in self.sequence_layer:
            x = block(x)
            output_list.append(x)
        x = self.output_layer(x)
        output_list.append(x)

        return x, output_list
"""
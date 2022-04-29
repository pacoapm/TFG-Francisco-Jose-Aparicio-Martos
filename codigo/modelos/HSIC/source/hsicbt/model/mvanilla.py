from .. import *
import sys
sys.path.insert(1, '/home/francisco/Documentos/ingenieria_informatica/cuarto_informatica/segundo_cuatri/TFG/TFG-Francisco-Jose-Aparicio-Martos/codigo')
from custom_funcs import QuantLayer

class ModelVanilla(nn.Module):

    def __init__(self, hidden_width=64, last_hidden_width=None, **kwargs):
        super(ModelVanilla, self).__init__()
        last_dim = hidden_width
        if last_hidden_width:
        	last_dim = last_hidden_width
        self.output = nn.Linear(last_dim, 10)

    def forward(self, x):
        x = self.output(x)
        return F.log_softmax(x, dim=1)

class QuantModelVanilla(nn.Module):
    
    def __init__(self, hidden_width=64, last_hidden_width=None, **kwargs):
        super(QuantModelVanilla, self).__init__()
        last_dim = hidden_width
        if last_hidden_width:
        	last_dim = last_hidden_width

        self.output = nn.Linear(last_dim, 10)
        self.quant = QuantLayer()

    def forward(self, x):
        x = self.output(x)
        x = self.quant(x)
        return F.log_softmax(x, dim=1)

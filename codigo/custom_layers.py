#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:27:41 2022

@author: francisco
Script para crear las custom layers con cuantización
"""


from brevitas.core.scaling import ConstScaling
from brevitas.core.quant.int_base import IntQuant
import torch
int_quant = IntQuant(narrow_range=True, signed=False)
print(torch.tensor(4.))
scale, zero_point, bit_width = torch.tensor(0.01), torch.tensor(0.), torch.tensor(2.)
inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
out = int_quant(scale, zero_point, bit_width, inp)
print(out)

"""
Creacion de mi custom linear layer
"""


from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union


import torch.nn as nn
import torch
from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import LSTM
from torch.nn.functional import relu

from pfhedge.nn import MultiLayerPerceptron


class TransformerModel(nn.Module):
    def __init__(self,d_model,hidden_size=512, num_layers=4, num_heads=8):
        super(TransformerModel, self).__init__()
        self.l1 = nn.Linear(d_model, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size,
                                       batch_first=True),
            num_layers
        )
        self.fc = MultiLayerPerceptron(in_features=hidden_size, n_layers=3)


    def forward(self, x):
        x = self.l1(x)
        # activation
        x = relu(x)

        # transformer
        encoded = self.encoder(x)





        output = self.fc(encoded)
        return output



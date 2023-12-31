import sys

sys.path.append("..")

from math import sqrt

import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption

from pfhedge.nn import BlackScholes
from pfhedge.features import list_feature_names
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import LSTM

from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn.modules.LSTM import LSTM_FC
from pfhedge.nn.modules.transformer import TransformerModel
from pfhedge.nn import BlackScholes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#小雪球即AmericanBinaryOption call + tail EuropeanOption put

torch.manual_seed(42)

strike = 1.0
maturity = 1.0
stock = BrownianStock()
european = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)

#   AmericanBinaryOption call - tail EuropeanOption put


def tail_clause(derivative, payoff):
    barrier = 0.92
    ending_spot = derivative.ul().spot[..., -1]
    tail_payoff = torch.full_like(payoff, strike - barrier)
    return torch.where(ending_spot > barrier, -payoff, -tail_payoff)


def up_and_out_clause(derivative, payoff, couple=0.15):
    #   add cap clause for AmericanBinaryOption call
    barrier = 1.1
    max_spot = derivative.ul().spot.max(-1).values
    capped_payoff = torch.full_like(payoff, couple)

    return torch.where(max_spot < barrier, payoff, capped_payoff)


xiaoxueqiu = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)
xiaoxueqiu.add_clause("tail_clause", tail_clause)
xiaoxueqiu.add_clause("up_and_out_clause", up_and_out_clause)

xiaoxueqiu.to(device)

'''
model = LSTM_FC(in_features=3,num_layers=2)
hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"]).to(device)

# Fit and price
batch_size = 1024
n_epochs = 100
price_list = []
for _ in range(n_epochs):
    hedger.fit(xiaoxueqiu, n_paths=batch_size, n_epochs=1)
    price = hedger.price(xiaoxueqiu, n_paths=batch_size)
    price_list.append(price)


price = torch.tensor(price_list).to('cpu').detach().numpy()
'''

#  use the transformer
model = TransformerModel(d_model=3)
hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"]).to(device)

# Fit and price
batch_size = 64
n_epochs = 200
price_list = []

for _ in range(n_epochs):
    hedger.fit(xiaoxueqiu, n_paths=batch_size, n_epochs=1)
    price = hedger.price(xiaoxueqiu, n_paths=512)
    price_list.append(price)

price = torch.tensor(price_list).to('cpu').detach().numpy()

import matplotlib.pyplot as plt
plt.plot(price)

plt.show()














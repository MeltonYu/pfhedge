import sys

sys.path.append("..")

from math import sqrt

import torch
import numpy as np
import matplotlib.pyplot as plt

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption

from pfhedge.nn import BlackScholes
from pfhedge.features import list_feature_names
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd
import numpy as np

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
maturity = 20 / 250

# set the volatility and the couple rate

volatility_list = [0.1, 0.15, 0.2]
couple_list = [0.15, 0.2, 0.25]

# construct the dataframe to store the results,set the index as the volatility, the columns as the couple rate
df = pd.DataFrame(index=volatility_list, columns=couple_list)

# set the name of the index
df.index.name = 'volatility/couple rate'

df_ = df.copy()

for vol in volatility_list:

    for couple in couple_list:

        print('--------------------------------------------------------------------')

        # Prepare a derivative to hedge
        stock = BrownianStock(sigma=vol, cost=2.7e-4)


        #   AmericanBinaryOption call - tail EuropeanOption put

        def tail_clause(derivative, payoff):
            barrier = 0.95
            ending_spot = derivative.ul().spot[..., -1]
            tail_payoff = torch.full_like(payoff, strike - barrier)
            return torch.where(ending_spot > barrier, -payoff, -tail_payoff)


        def up_and_out_clause(derivative, payoff, couple=couple):
            #   add cap clause for AmericanBinaryOption call
            barrier = 1.03
            max_spot = derivative.ul().spot.max(-1).values
            capped_payoff = torch.full_like(payoff, couple)

            return torch.where(max_spot < barrier, payoff, capped_payoff)


        xiaoxueqiu = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)
        xiaoxueqiu.add_clause("tail_clause", tail_clause)
        xiaoxueqiu.add_clause("up_and_out_clause", up_and_out_clause)

        xiaoxueqiu.to(device)

        #

        #  use the transformer
        model = TransformerModel(d_model=3)
        hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"]).to(device)

        # Fit and price
        batch_size = 64
        n_epochs = 1000
        price_list = []

        hedger.fit(xiaoxueqiu, n_paths=batch_size, n_epochs=n_epochs)

        for _ in range(10):
            price = hedger.price(xiaoxueqiu, n_paths=2048)
            price_list.append(price.item())

        price = np.mean(price_list)
        df.loc[vol, couple] = price

        #  use the BlackScholes
        model_BS = BlackScholes(xiaoxueqiu)
        hedger_BS = Hedger(model_BS, model_BS.inputs()).to(device)

        price_list_BS = []

        # Fit and price
        for _ in range(10):
            price_BS = hedger_BS.price(xiaoxueqiu, n_paths=2048)
            price_list_BS.append(price_BS.item())

        price_BS = np.mean(price_list_BS)
        df_.loc[vol, couple] = price_BS



# save the dataframe
df.to_excel('output/xiaoxueqiu_transformer.xlsx')

df_.to_excel('output/xiaoxueqiu_BS.xlsx')



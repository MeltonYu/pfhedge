import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn.modules.LSTM import LSTM_FC
from pfhedge.nn import BlackScholes

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the volatility and the cost
volatility_list = [0.1, 0.15,0.2]
cost_list = [1e-4, 2e-4, 3e-4]

# construct the dataframe to store the results,set the index as the volatility, the columns as the cost
df = pd.DataFrame(index=volatility_list, columns=cost_list)

# set the name of the columns and the index
df.index.name = 'volatility/cost'


df_ = df.copy()




for vol in volatility_list:
    for cost in cost_list:

        print('--------------------------------------------------------------------')

        # Prepare a derivative to hedge
        derivative = EuropeanOption(BrownianStock(sigma=vol, cost=cost),
                                    maturity=20/250).to(device)

        # Create your hedger
        model_BS = BlackScholes(derivative)
        hedger_BS = Hedger(model_BS, model_BS.inputs()).to(device)

        # Fit and price
        price = hedger_BS.price(derivative, n_paths=10000)
        print(f"BS Price={price:.5e}")
        # to cpu
        price = price.to('cpu').detach().numpy()
        # store the price
        df.loc[vol, cost] = price


        derivative.to(device)
        #  use the LSTM
        model_LSTM = LSTM_FC(in_features=3, num_layers=2)

        hedger_LSTM = Hedger(model_LSTM, ["log_moneyness", "expiry_time", "volatility"]).to(device)

        # Fit and price
        hedger_LSTM.fit(derivative, n_paths=64, n_epochs=300)
        price = hedger_LSTM.price(derivative, n_paths=10000)
        print(f"LSTM Price={price:.5e}")

        # to cpu
        price = price.to('cpu').detach().numpy()
        # store the price
        df_.loc[vol, cost] = price

        print('--------------------------------------------------------------------')


# save the dataframe








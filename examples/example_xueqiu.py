import sys

sys.path.append("..")

from math import sqrt

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.features import list_feature_names

from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn.modules.LSTM import LSTM_FC
from pfhedge.nn import BlackScholes
from pfhedge.nn.modules.transformer import TransformerModel


#  经典雪球可以看作是Two AmericanBinaryOption calls - down and in EuropeanOption put



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

strike = 1.0
maturity = float((252/250)*2)  # approximately 2 years, here we use 21 days as a month
stock = BrownianStock()
european = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)

# two  AmericanBinaryOption call - tail EuropeanOption put





def xueqiu_payoff(derivative, payoff,up_barrier=1.03, down_barrier= 0.80,couple=0.24,suoding_months= 3):
    '''
    Args:
        up_barrier(float,default = 1.03): up and out barrier
        down_barrier(float,default = 0.75): down and in barrier
        couple(float,default = 0.12):couple yearly
        guancha_days(int,default = 0): the days to observe the barrier
        suoding_days(int,default = 0): the days to lock the barrier

    '''

    n_paths = derivative.ul().spot.shape[0]
    num_days = derivative.ul().spot.shape[-1]
    #  find the guancha_days
    guancha_days = [21*i for i in range(suoding_months,num_days//21+1)]
    guancha_days[-1] = num_days-1  #  the last day is always observed

    #  get the underlying spot in the guancha_days


    #  find the first guancha_day where the spot is larger than the barrier
    for n in range(n_paths):
        out_signal = 0
        in_signal = 0

        for day in range(guancha_days[0],num_days):
            if day in guancha_days:
                if derivative.ul().spot[n,day] >= up_barrier:
                    out_signal = 1
                    payoff[n] = couple * day/252
                    break

            if derivative.ul().spot[n,day] <= down_barrier:
                in_signal = 1
                continue

            if out_signal == 0:
                if in_signal == 1:
                    payoff[n] = -payoff[n]  #  敲入未敲出，short put

                else:
                    payoff[n] = (num_days-1)/252 * couple  #  未敲入未敲出，红利票息









    return payoff

honglixueqiu = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)
honglixueqiu.add_clause("xueqiu_payoff", xueqiu_payoff)
honglixueqiu.to(device)

#  use the transformer
model = TransformerModel(d_model=3)
hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"]).to(device)

# Fit and price
batch_size = 64
n_epochs = 30

price_list = []
for _ in range(n_epochs):
    hedger.fit(honglixueqiu, n_paths=batch_size, n_epochs=1)
    price = hedger.price(honglixueqiu, n_paths=512)
    price_list.append(price)

price = torch.tensor(price_list).to('cpu').detach().numpy()

plt.plot(price)
plt.show()











import sys

sys.path.append("..")

from math import sqrt

import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.features import list_feature_names


#小雪球即AmericanBinaryOption call + tail EuropeanOption put




torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

strike = 1.0
maturity = 1.0
stock = BrownianStock()
european = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)

#   AmericanBinaryOption call - tail EuropeanOption put


def tail_clause(derivative, payoff):
    barrier = 0.95
    ending_spot = derivative.ul().spot[..., -1]
    tail_payoff = torch.full_like(payoff, strike - barrier)
    return torch.where(ending_spot > barrier, -payoff, -tail_payoff)


def up_and_out_clause(derivative, payoff, couple=0.12):
    #   add cap clause for AmericanBinaryOption call
    barrier = 1.03
    max_spot = derivative.ul().spot.max(-1).values
    capped_payoff = torch.full_like(payoff, couple)

    return torch.where(max_spot < barrier, payoff, capped_payoff)


xiaoxueqiu = EuropeanOption(stock, call=False, strike=strike, maturity=maturity).to(device=device, dtype=dtype)
xiaoxueqiu.add_clause("tail_clause", tail_clause)
xiaoxueqiu.add_clause("up_and_out_clause", up_and_out_clause)

n_paths = 10000
xiaoxueqiu.simulate(n_paths=n_paths)

payoff_xiaoxueqiu = xiaoxueqiu.payoff()
mean_xiaoxueqiu = payoff_xiaoxueqiu.mean().item()

model = MultiLayerPerceptron().to(device=device, dtype=dtype)
hedger = Hedger(model,
                ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]).to(device=device, dtype=dtype)

hedger.fit(xiaoxueqiu, n_paths=10000, n_epochs=10)
price = hedger.price(xiaoxueqiu, n_paths=10000)
print(f"Price={price:.5e}")

ratio=hedger.compute_hedge(xiaoxueqiu).squeeze(1)







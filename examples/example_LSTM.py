# Example to use a multi-layer perceptron as a hedging model

import sys

import torch

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn.modules.LSTM import LSTM_FC
from pfhedge.nn import BlackScholes

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare a derivative to hedge
derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(device)

# Create your hedger

'''
#  use the MultiLayerPerceptron
model_1 = MultiLayerPerceptron()
hedger_1 = Hedger(model_1, ["log_moneyness", "expiry_time", "volatility"]).to(device)

# Fit and price
hedger_1.fit(derivative, n_paths=10000, n_epochs=200)
price_1 = hedger_1.price(derivative, n_paths=10000)
print(f"Multilayer Price={price_1:.5e}")
'''
#  use the BlackScholes
model_2 = BlackScholes(derivative).to(device)
hedger_2 = Hedger(model_2, model_2.inputs()).to(device)

# Fit and price
price_2 = hedger_2.price(derivative, n_paths=10000)

print(f"BlackScholes Price={price_2:.5e}")


#  use the LSTM
model_3 = LSTM_FC(in_features=3)
hedger_3 = Hedger(model_3, ["log_moneyness", "expiry_time", "volatility"]).to(device)

# Fit and price
hedger_3.fit(derivative, n_paths=10000, n_epochs=200)
price_3 = hedger_3.price(derivative, n_paths=10000)
print(f"LSTM Price={price_3:.5e}")

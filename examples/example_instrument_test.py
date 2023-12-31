import sys

sys.path.append("..")

from math import sqrt

import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes

#  test the atributte of EuropeanOption
_ = torch.manual_seed(42)
strike = 1.0
maturity = 1.0
stock = BrownianStock()
european = EuropeanOption(stock, call=False, strike=strike, maturity=maturity)

#  Note, every simulation is independent !
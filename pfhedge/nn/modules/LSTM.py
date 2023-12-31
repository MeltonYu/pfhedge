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


class LSTM_FC(Module):
    r"""Creates a LSTM structure.

    Number of input features is lazily determined.

    Args:
        in_features (int, optional): Size of each input sample.
            If ``None`` (default), the number of input features will be
            will be inferred from the ``input.shape[-1]`` after the first call to
            ``forward`` is done. Also, before the first ``forward`` parameters in the
            module are of :class:`torch.nn.UninitializedParameter` class.
        out_features (int, default=1): Size of each output sample.
        n_layers (int, default=4): The number of hidden layers.
        n_units (int or tuple[int], default=32): The number of units in
            each hidden layer.
            If ``tuple[int]``, it specifies different number of units for each layer.
        activation (torch.nn.Module, default=torch.nn.ReLU()):
            The activation module of the hidden layers.
            Default is a :class:`torch.nn.ReLU` instance.
        out_activation (torch.nn.Module, default=torch.nn.Identity()):
            The activation module of the output layer.
            Default is a :class:`torch.nn.Identity` instance.

    Shape:
        - Input: :math:`(N, *, H_{\text{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\text{in}}` is the number of input features.
        - Output: :math:`(N, *, H_{\text{out}})` where
          all but the last dimension are the same shape as the input and
          :math:`H_{\text{out}}` is the number of output features.

    Examples:

        By default, ``in_features`` is lazily determined:


        MultiLayerPerceptron(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )

        MultiLayerPerceptron(
          (0): Linear(in_features=2, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )

        Specify different number of layers for each layer:


        MultiLayerPerceptron(
          (0): Linear(in_features=1, out_features=16, bias=True)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=1, bias=True)
          (5): Identity()
        )
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        hidden_lstm_out: int = 64,
        num_layers: int = 1,
        out_features: int = 1,
        n_layers: int = 4,
        n_units: int = 64,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_lstm_out = hidden_lstm_out
        self.out_features = out_features
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.out_activation = out_activation

        self.lstm = LSTM(input_size=self.in_features, hidden_size=self.hidden_lstm_out,
                         num_layers=num_layers, batch_first=True)

        self.fc = Sequential()
        self.fc.append(Linear(in_features=self.hidden_lstm_out, out_features=self.n_units))
        self.fc.append(self.activation)

        for i in range(self.n_layers - 1):
            self.fc.append(Linear(in_features=self.n_units, out_features=self.n_units))
            self.fc.append(self.activation)

        self.fc.append(Linear(in_features=self.n_units, out_features=self.out_features))

        self.fc.append(self.out_activation)



    def forward(self, input):
        lstm_out, _ = self.lstm(input)

        #  activation
        lstm_out = self.activation(lstm_out)

        #  fc layers
        output = self.fc(lstm_out)

        return output



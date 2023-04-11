import torch
import torch.nn as nn
from typing import Callable


class NeuralNetworkFunctionWrapper1D:
    def __init__(
            self,
            appr_func: Callable = None,
            num_hidden: int = 50,
            num_layers: int = 1,
            act=nn.Tanh(),
    ):
        self.nn_model = NeuralNetworkFunction(1, num_hidden, num_layers, act)
        if appr_func is None:
            self.appr_model = self.nn_model
        else:
            self.appr_model = lambda x: appr_func(x, self.nn_model)

    def train(self) -> None:
        self.nn_model.train()

    def eval(self) -> None:
        self.nn_model.train()

    def __call__(self, x):
        return self.appr_model(x)

    def parameters(self) -> iter:
        return self.nn_model.parameters()


class NeuralNetworkFunctionWrapper2D(NeuralNetworkFunctionWrapper1D):
    def __init__(
            self,
            appr_func: Callable = None,
            num_hidden: int = 50,
            num_layers: int = 1,
            act=nn.Tanh(),
    ):
        super().__init__(appr_func, num_hidden, num_layers, act)
        self.nn_model = NeuralNetworkFunction(2, num_hidden, num_layers, act)
        if appr_func is None:
            self.appr_model = self.nn_model
        else:
            self.appr_model = lambda x, y: appr_func(x, y, self.nn_model)

    def __call__(self, x, y):
        return self.appr_model(x, y)


class NeuralNetworkFunction(nn.Module):
    def __init__(self, input_dim: int = 1, num_hidden: int = 50, num_layers: int = 1, act=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.inp_layer = nn.Linear(input_dim, num_hidden)
        self.middle_layers = [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]
        self.act = act
        self.layer_out = nn.Linear(num_hidden, 1)

    def forward(self, *inputs):
        united_inp = torch.stack(inputs, dim=self.input_dim)
        out = self.inp_layer(united_inp)
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        return torch.squeeze(out, dim=self.input_dim)


class ResNet(torch.nn.Module):
    def __init__(self, module,
                 act=nn.Tanh()):
        super().__init__()
        self.module = module
        self.act = act

    def forward(self, inputs):
        return self.act(self.module(inputs)) + inputs


# nn with residual layers
class NeuralNetworkFunctionResNet(nn.Module):
    def __init__(self,
                 inp_dim: int = 1,
                 num_hidden: int = 50,
                 num_layers: int = 1,
                 act=nn.Tanh(),
                 ):
        super().__init__()
        self.inp_dim = inp_dim
        self.layer_in = nn.Linear(self.inp_dim, num_hidden)
        self.middle_layers = [ResNet(
            nn.Linear(num_hidden, num_hidden), act) for _ in range(num_layers)]
        self.layer_out = nn.Linear(num_hidden, 1)
        self.act = act

    def forward(self, *inputs):
        united_inp = torch.stack(inputs, dim=self.inp_dim)
        out = self.layer_in(united_inp)
        for layer in self.middle_layers:
            out = layer(out)
        out = self.layer_out(out)
        out = torch.squeeze(out, dim=self.inp_dim)
        return out

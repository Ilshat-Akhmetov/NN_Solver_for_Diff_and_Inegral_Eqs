import torch
import torch.nn as nn
from typing import Callable


class NeuralNetworkFunction1D(nn.Module):
    def __init__(self, num_hidden: int = 50, num_layers: int = 1, act=nn.Tanh()):
        super().__init__()

        self.layer_in = nn.Linear(1, num_hidden)
        self.layer_out = nn.Linear(num_hidden, 1)
        self.middle_layers = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]
        )
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.tensor:
        x = x.view(-1, 1)
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        out = torch.squeeze(out, dim=1)
        return out


class NeuralNetworkFunctionWrapper1D:
    def __init__(
        self,
        appr_func: Callable = None,
        num_hidden: int = 50,
        num_layers: int = 1,
        act=nn.Tanh(),
    ):
        self.nn_model = NeuralNetworkFunction1D(num_hidden, num_layers, act)
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


class NeuralNetworkFunction2D(NeuralNetworkFunction1D):
    def __init__(self, num_hidden: int = 50, num_layers: int = 1, act=nn.Tanh()):
        super().__init__(num_hidden, num_layers, act)
        self.inp_layer = nn.Linear(2, num_hidden)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        united_inp = torch.stack((x, y), dim=2)
        out = self.inp_layer(united_inp)
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        return torch.squeeze(out, dim=2)


class NeuralNetworkFunctionWrapper2D(NeuralNetworkFunctionWrapper1D):
    def __init__(
        self,
        appr_func: Callable = None,
        num_hidden: int = 50,
        num_layers: int = 1,
        act=nn.Tanh(),
    ):
        super().__init__(appr_func, num_hidden, num_layers, act)
        self.nn_model = NeuralNetworkFunction2D(num_hidden, num_layers, act)
        if appr_func is None:
            self.appr_model = self.nn_model
        else:
            self.appr_model = lambda x, y: appr_func(x, y, self.nn_model)

    def __call__(self, x, y):
        return self.appr_model(x, y)


# class ResNet(torch.nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module
#
#     def forward(self, inputs):
#         return self.module(inputs) + inputs
#
# #nn with residual layers
# class NeuralNetworkFunction(nn.Module):
#     def __init__(self, num_hidden: int = 50,
#                  num_layers: int = 1,
#                  act=nn.Tanh()):
#         super().__init__()
#
#         self.layer_in = nn.Linear(1, num_hidden)
#         self.resnet1 = ResNet(
#             nn.Sequential(
#             nn.Linear(num_hidden, num_hidden),
#             nn.Tanh()
#             )
#         )
#         self.layer_out = nn.Linear(num_hidden, 1)
#         self.act = act
#
#     def forward(self, x: torch.Tensor):
#         out = self.act(self.layer_in(x))
#         out = self.act(out)
#         out = self.resnet1(out)
#         return self.layer_out(out)

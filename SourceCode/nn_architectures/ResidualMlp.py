import torch
from torch import nn


class Residual_block(torch.nn.Module):
    def __init__(self, module, act=nn.Tanh):
        super().__init__()
        self.module = module
        self.act = act()

    def forward(self, inputs):
        return self.act(self.module(inputs)) + inputs


# nn with residual layers
class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 50,
        num_hidden_layers: int = 1,
        act=nn.Tanh,
    ):
        super().__init__()
        self.inp_dim = input_dim
        self.layer_in = nn.Linear(self.inp_dim, hidden_dim)
        self.middle_layers = [
            Residual_block(nn.Linear(hidden_dim, hidden_dim), act)
            for _ in range(num_hidden_layers)
        ]
        self.layer_out = nn.Linear(hidden_dim, 1)
        self.act = act()

    def forward(self, *inputs):
        united_inp = torch.stack(inputs, dim=self.inp_dim)
        out = self.layer_in(united_inp)
        for layer in self.middle_layers:
            out = layer(out)
        out = self.layer_out(out)
        out = torch.squeeze(out, dim=self.inp_dim)
        return out

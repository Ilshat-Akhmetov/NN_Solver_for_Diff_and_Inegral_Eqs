import torch
from torch import nn


class BasisFuncMLP(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 50, num_hidden_layers: int = 1, act=nn.Tanh,
                 basis_funcs=[lambda x: torch.ones(x.shape), lambda x: x, lambda x: x * x]):
        super().__init__()
        self.input_dim = input_dim
        self.basic_funcs = basis_funcs
        self.inp_layer = nn.Linear(input_dim, hidden_dim)
        self.middle_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.output_layer = nn.Linear(hidden_dim, len(basis_funcs))
        self.act = act()

    def forward(self, *inputs):
        united_inp = torch.stack(inputs, dim=self.input_dim)
        x = self.inp_layer(united_inp)
        for layer in self.middle_layers:
            x = self.act(layer(x))
        x = self.output_layer(x)
        basis_func_outp = [func(*inputs) for func in self.basic_funcs]
        basic_funcs_tensor = torch.stack(basis_func_outp, dim=len(inputs))
        prod = x * basic_funcs_tensor
        return torch.sum(prod, dim=len(inputs))


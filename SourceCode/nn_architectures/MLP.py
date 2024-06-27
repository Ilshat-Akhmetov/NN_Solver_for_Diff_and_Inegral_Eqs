from torch import nn
from torch import stack, squeeze


class MLP(nn.Module):
    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 50, num_hidden_layers: int = 1, act=nn.Tanh
    ):
        super().__init__()
        self.inp_layer = nn.Linear(input_dim, hidden_dim)
        self.middle_layers = [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ]
        self.act = act()
        self.layer_out = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim

    def forward(self, *inputs):
        united_inp = stack(inputs, dim=self.input_dim)
        out = self.inp_layer(united_inp)
        for layer in self.middle_layers:
            out = self.act(layer(out))
        out = self.layer_out(out)
        out = squeeze(out, dim=self.input_dim)
        return out

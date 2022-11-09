import torch
import torch.nn as nn


class NeuralNetworkFunction(nn.Module):
    def __init__(self, num_inputs: int = 1,
                 num_hidden: int = 50,
                 num_outputs: int = 1,
                 num_layers: int = 1,
                 act=nn.Tanh()):
        super().__init__()

        self.layer_in = nn.Linear(num_inputs, num_hidden)
        self.layer_out = nn.Linear(num_hidden, num_outputs)

        self.middle_layers = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]
        )
        self.act = act

    def forward(self, x: torch.Tensor):
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

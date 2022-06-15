import torch
import torch.nn as nn


class NeuralNetworkFunction(nn.Module):
    def __init__(self, number_of_inputs: int, hidden_neurons_number: int, number_of_outputs: int):
        super().__init__()
        self.Sequence = nn.Sequential(
            nn.Linear(number_of_inputs, hidden_neurons_number),
            nn.Tanh(),
            nn.Linear(hidden_neurons_number, hidden_neurons_number),
            nn.Tanh(),
            nn.Linear(hidden_neurons_number, hidden_neurons_number),
            nn.Tanh(),
            nn.Linear(hidden_neurons_number, hidden_neurons_number),
            nn.Tanh(),
            nn.Linear(hidden_neurons_number, number_of_outputs),
        )

    def forward(self, x: torch.Tensor):
        return self.Sequence(x)


# class NeuralNetworkFunction(nn.Module):
#     def __init__(self, num_hidden: int = 4, dim_hidden: int = 100, act=nn.Tanh()):
#         super().__init__()
#
#         self.layer_in = nn.Linear(1, dim_hidden)
#         self.layer_out = nn.Linear(dim_hidden, 1)
#
#         num_middle = num_hidden - 1
#         self.middle_layers = nn.ModuleList(
#             [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
#         )
#         self.act = act
#
#     def forward(self, x):
#         out = self.act(self.layer_in(x))
#         for layer in self.middle_layers:
#             out = self.act(layer(out))
#         return self.layer_out(out)

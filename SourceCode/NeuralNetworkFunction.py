import torch
import torch.nn as nn


class NeuralNetworkFunction(nn.Module):
    def __init__(self, num_inputs: int = 1,
                 num_hidden: int = 50,
                 num_layers: int = 1,
                 act=nn.Tanh()):
        super().__init__()

        self.layer_in = nn.Linear(num_inputs, num_hidden)
        self.layer_out = nn.Linear(num_hidden, 1)
        self.middle_layers = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]
        )
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.tensor:
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)

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
#     def __init__(self, num_inputs: int = 1,
#                  num_hidden: int = 50,
#                  num_layers: int = 1,
#                  act=nn.Tanh()):
#         super().__init__()
#
#         self.layer_in = nn.Linear(num_inputs, num_hidden)
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

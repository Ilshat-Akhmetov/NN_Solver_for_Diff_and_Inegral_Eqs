import torch
import torch.nn as nn


class CustomNeuralNetworkClass(nn.Module):
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

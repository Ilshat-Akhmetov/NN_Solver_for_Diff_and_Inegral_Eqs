import torch.nn as nn

# class CustomResNet(nn.Module):
#     def __init__(self, hidden_neurons):
#         super().__init__()
#         self.Sequence = nn.Sequential(
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.Tanh()
#         )
#     def forward(self,x):
#         return self.Sequence(x)
#
# class CustomNeuralNetworkClass(nn.Module):
#     def __init__(self, NumberOfInputs, hidden_neurons, NumberOfOutputs):
#         super().__init__()
#         self.SeqStart = nn.Sequential(
#             nn.Linear(NumberOfInputs, hidden_neurons),
#             nn.Tanh()
#         )
#         self.RN1 = CustomResNet(hidden_neurons)
#         self.RN2 = CustomResNet(hidden_neurons)
#         self.RN3 = CustomResNet(hidden_neurons)
#         self.RN4 = CustomResNet(hidden_neurons)
#         self.LinearOutput = nn.Linear(hidden_neurons, NumberOfOutputs)
#     def forward(self,X):
#         X = self.SeqStart(X)
#         X1 = self.RN1(X) + X
#         X2 = self.RN2(X1) + X1
#         X3 = self.RN3(X2) + X2
#         X4 = self.RN4(X3) + X3
#         X5 = self.LinearOutput(X4)
#         return X5


class CustomNeuralNetworkClass(nn.Module):
    def __init__(self, NumberOfInputs, hidden_neurons, NumberOfOutputs):
        super().__init__()
        self.Sequence = nn.Sequential(
            nn.Linear(NumberOfInputs, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, NumberOfOutputs),
        )

    def forward(self, X):
        return self.Sequence(X)

#
# class CustomNeuralNetworkClass(nn.Module):
#     def __init__(self, NumberOfInputs,hidden_neurons,NumberOfOutputs):
#         super().__init__()
#         self.Sequence = nn.Sequential(
#             nn.Linear(NumberOfInputs,hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons,hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons, hidden_neurons),
#             nn.CELU(),
#             nn.Linear(hidden_neurons,NumberOfOutputs)
#         )
#     def forward(self,X):
#         return self.Sequence(X)

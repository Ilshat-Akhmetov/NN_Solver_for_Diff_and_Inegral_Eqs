import torch
from typing import Callable


class NeuralNetworkFunctionWrapper1D:
    def __init__(
        self, appr_func: Callable, model_params: dict, model_type: torch.nn.Module
    ):
        self.nn_model = model_type(**model_params)
        if appr_func is None:
            self.appr_model = self.nn_model
        else:
            self.appr_model = lambda x: appr_func(x, self.nn_model)

    def train(self) -> None:
        self.nn_model.train()

    def eval(self) -> None:
        self.nn_model.eval()

    def __call__(self, x: torch.Tensor):
        out = self.appr_model(x)
        return out

    def parameters(self) -> iter:
        return self.nn_model.parameters()


class NeuralNetworkFunctionWrapper2D(NeuralNetworkFunctionWrapper1D):
    def __init__(
        self, appr_func: Callable, model_params: dict, model_type: torch.nn.Module
    ):
        super().__init__(appr_func, model_params, model_type)
        self.nn_model = model_type(**model_params)
        if appr_func is None:
            self.appr_model = self.nn_model
        else:
            self.appr_model = lambda x, y: appr_func(x, y, self.nn_model)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.appr_model(x, y)
        return out

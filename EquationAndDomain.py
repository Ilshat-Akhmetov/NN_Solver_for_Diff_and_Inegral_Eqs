import abc
import torch
from typing import Callable


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_value(self):
        pass


class OnePointInitialCondition(AbstractEquation):
    def __init__(self, point: float, equation: Callable[[float, float], float]):
        self.point = torch.Tensor([point])
        self.point.requires_grad = True
        self.equation = equation

    def get_value(self, nn_model):
        nn_model_value = nn_model(self.point)
        return self.equation(self.point, nn_model_value)


class MainEquation(AbstractEquation):
    def __init__(
        self,
        left_point: float,
        right_point: float,
        n_points: int,
        equation: Callable[[float, float], float],
    ):
        self.n_points = n_points
        self.right_point = right_point
        self.left_point = left_point
        self.equation = equation
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()

    def get_value(self, point, nn_model_value):
        # nn_model_value = nn_model(point)
        residual = self.equation(point, nn_model_value)
        return residual

    def make_train_domain(self):
        return torch.linspace(self.left_point, self.right_point, self.n_points + 2)[
            1:-1
        ]

    def make_valid_domain(self):
        valid_domain = (self.train_domain[1:] + self.train_domain[:-1]) / 2
        return valid_domain

    def get_train_domain(self):
        return self.train_domain.unsqueeze(dim=1)

    def get_valid_domain(self):
        return self.valid_domain.unsqueeze(dim=1)

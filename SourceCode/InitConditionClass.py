import abc
import torch
from typing import Callable


class InitialCondition(abc.ABC):
    @abc.abstractmethod
    def get_boundary_residuals(self, nn_model: torch.nn) -> torch.tensor:
        raise NotImplementedError


class OnePointInitialCondition(InitialCondition):
    def __init__(
        self,
        point: float,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.point = torch.Tensor([point])
        self.point.requires_grad = True
        self.equation = equation

    def get_boundary_residuals(self, nn_model: torch.nn) -> torch.tensor:
        nn_model_value = nn_model(self.point)
        return self.equation(self.point, nn_model_value)

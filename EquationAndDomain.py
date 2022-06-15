import abc
import torch
from typing import Callable


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_residuals_train(self, nn_model: torch.nn) -> torch.tensor:
        pass

    @abc.abstractmethod
    def get_residuals_valid(self, nn_model: torch.nn) -> torch.tensor:
        pass

    @abc.abstractmethod
    def make_train_domain(self) -> torch.tensor:
        pass

    @abc.abstractmethod
    def make_valid_domain(self) -> torch.tensor:
        pass

    @abc.abstractmethod
    def get_train_domain(self) -> torch.tensor:
        pass

    @abc.abstractmethod
    def get_valid_domain(self) -> torch.tensor:
        pass


class InitialCondition(abc.ABC):
    @abc.abstractmethod
    def get_boundary_residuals(self, nn_model: torch.nn) -> torch.tensor:
        pass


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


class OneDimensionalMainEquation(AbstractEquation):
    def __init__(
        self,
        left_point: float,
        right_point: float,
        n_points: int,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.n_points = n_points
        self.right_point = right_point
        self.left_point = left_point
        self.equation = equation
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()

    def get_residuals_train(self, nn_model: torch.nn):
        domain = self.get_train_domain()
        nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model_value)
        return residual

    def get_residuals_valid(self, nn_model):
        domain = self.get_valid_domain()
        nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model_value)
        return residual

    def make_train_domain(self) -> torch.tensor:
        train_domain = torch.linspace(
            self.left_point, self.right_point, self.n_points + 2
        )[1:-1]
        train_domain.requires_grad = True
        return train_domain

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = (self.train_domain[1:] + self.train_domain[:-1]) / 2
        return valid_domain

    def get_train_domain(self) -> torch.tensor:
        return self.train_domain.unsqueeze(dim=1)

    def get_valid_domain(self) -> torch.tensor:
        return self.valid_domain.unsqueeze(dim=1)

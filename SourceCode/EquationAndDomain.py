import abc
import torch
from typing import Callable


class AbstractDomain(abc.ABC):
    @abc.abstractmethod
    def get_train_domain(self) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_domain(self) -> torch.tensor:
        raise NotImplementedError


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_residuals_train(self, nn_model: torch.nn) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_residuals_valid(self, nn_model: torch.nn) -> torch.tensor:
        raise NotImplementedError


class InitialCondition(abc.ABC):
    @abc.abstractmethod
    def get_boundary_residuals(self, nn_model: torch.nn) -> torch.tensor:
        raise NotImplementedError


class OneDimensionalSimpleDomain(AbstractDomain):
    def __init__(self, left_bound: int, right_bound: int, n_points: int):
        self.n_points = n_points
        self.left_point = left_bound
        self.right_point = right_bound
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()

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
        domain: AbstractDomain,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.domain = domain
        self.equation = equation

    def get_residuals_train(self, nn_model: torch.nn):
        domain = self.domain.get_train_domain()
        nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model_value)
        return residual

    def get_residuals_valid(self, nn_model):
        domain = self.domain.get_valid_domain()
        nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model_value)
        return residual

import abc
import torch
from typing import Callable, List, Union
from .DomainClass import AbstractDomain
from types import FunctionType


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_residuals(
        self, nn_models: torch.nn, domain: torch.tensor, equation: Callable
    ) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def count_equations(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_residuals_norm(
        self, nn_models: torch.nn, phase: str
    ) -> (torch.tensor, torch.tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def get_nn_model_type(self) -> object:
        raise NotImplementedError


class MainEquationClass(AbstractEquation):
    def __init__(
        self,
        domain: AbstractDomain,
        equations: Union[
            Callable[[torch.Tensor], torch.Tensor], List[Callable]
        ],
        boundary_conditions: list = None,
        domain_to_bound_ratio: Union[float, int] = 10,
    ):
        self.domain = domain
        assert type(equations) in [FunctionType, list]
        if type(equations) is FunctionType:
            self.equations = [equations]
        else:
            self.equations = equations
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            self.boundary_conditions = []
        self.boundary_coefficient = self.get_boundary_coefficient(domain_to_bound_ratio)
        self.norm = lambda x: torch.pow(x, 2)

    def get_nn_model_type(self) -> Callable:
        return self.domain.get_nn_type()

    def get_boundary_coefficient(
        self, desired_ratio: Union[float, int]
    ) -> Union[int, float, None]:
        n_bound_conditions = 0
        for bound_condition in self.boundary_conditions:
            n_bound_conditions += bound_condition.get_domain_size()
        if n_bound_conditions == 0:
            return None
        domain_size = self.domain.get_domain_size() * len(self.equations)
        return domain_size / (n_bound_conditions * desired_ratio)

    def count_equations(self) -> int:
        return len(self.equations)

    def get_residuals(
        self, nn_models: List[Callable], domain: List[torch.tensor], equation: Callable
    ) -> torch.tensor:
        residual = equation(*domain, *nn_models)
        return residual

    def get_residuals_norm(
        self, nn_models: torch.nn, phase: str
    ) -> (torch.tensor, torch.tensor):
        assert phase in ["train", "valid"]
        domain = self.domain.get_domain(phase)
        total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        max_res_norm = torch.tensor(0.0, requires_grad=False)
        total_n_points = self.domain.get_domain_size()
        with torch.set_grad_enabled(True):
            for equation in self.equations:
                main_domain_residuals = self.get_residuals(nn_models, domain, equation)
                loss_val = self.norm(main_domain_residuals)
                curr_max_res_norm = torch.max(torch.abs(main_domain_residuals))
                max_res_norm = torch.max(curr_max_res_norm, max_res_norm)
                total_loss += torch.sum(loss_val)
            for boundary_condition in self.boundary_conditions:
                boundary_residuals = boundary_condition.get_boundary_residuals(
                    nn_models
                )
                boundary_loss = torch.sum(
                    self.boundary_coefficient * self.norm(boundary_residuals)
                )
                total_n_points += boundary_condition.get_domain_size()
                boundary_res_norm = torch.max(torch.abs(boundary_residuals))
                max_res_norm = torch.max(max_res_norm, boundary_res_norm)
                total_loss += boundary_loss
        mean_loss = total_loss / total_n_points
        return mean_loss, max_res_norm

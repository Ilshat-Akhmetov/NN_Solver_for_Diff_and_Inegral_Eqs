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


class MainEquationClass(AbstractEquation):
    def __init__(
        self,
        domain: AbstractDomain,
        equations: Union[Callable[[torch.Tensor], torch.Tensor], List[Callable]],
        boundary_conditions: list = None,
        bound_cond_coef: Union[float, int] = 0.1,
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
        self.boundary_coefficient = self.get_boundary_coefficient(bound_cond_coef)
        abs_norm = lambda x: torch.abs(x)
        self.max_norm = lambda x: torch.max(abs_norm(x))
        square_norm = lambda x: torch.pow(x, 2)
        self.norm_to_minimize = square_norm

    def get_boundary_coefficient(
        self, bound_coefficient: Union[float, int]
    ) -> Union[int, float, None]:
        bound_cond_size = 0
        for bound_condition in self.boundary_conditions:
            bound_cond_size += bound_condition.get_domain_size()
        if bound_cond_size == 0:
            return None
        domain_size = self.domain.get_domain_size() * len(self.equations)
        coefficient = bound_coefficient * domain_size / bound_cond_size
        return coefficient

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
                loss_val = self.norm_to_minimize(main_domain_residuals)
                curr_max_res_norm = self.max_norm(main_domain_residuals)
                max_res_norm = torch.max(curr_max_res_norm, max_res_norm)
                total_loss += torch.sum(loss_val)
            for boundary_condition in self.boundary_conditions:
                boundary_residuals = boundary_condition.get_boundary_residuals(
                    nn_models
                )
                boundary_loss = torch.sum(
                    self.boundary_coefficient
                    * self.norm_to_minimize(boundary_residuals)
                )
                total_n_points += boundary_condition.get_domain_size()
                boundary_res_norm = self.max_norm(main_domain_residuals)
                max_res_norm = torch.max(max_res_norm, boundary_res_norm)
                total_loss += boundary_loss
        mean_square_loss = total_loss / total_n_points
        mean_square_loss = torch.sqrt(mean_square_loss)
        return mean_square_loss, max_res_norm

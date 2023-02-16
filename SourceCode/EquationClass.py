import abc
import torch
from typing import Callable
from .DomainClass import AbstractDomain


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_residuals(self, nn_model: torch.nn, phase: str) -> torch.tensor:
        raise NotImplementedError
    @abc.abstractmethod
    def get_residuals_norm(self) -> (torch.tensor, torch.tensor):
        raise NotImplementedError


class OneDimensionalMainEquation(AbstractEquation):
    def __init__(
        self,
        domain: AbstractDomain,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        boundary_conditions: list = None
    ):
        self.domain = domain
        self.equation = equation
        self.boundary_coefficient = 1
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            self.boundary_conditions = []
        self.norm = lambda x: torch.pow(x, 2)
    def get_residuals(self, nn_model: torch.nn, domain: torch.tensor) -> torch.tensor:
        #nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model)
        return residual

    def get_residuals_norm(self, nn_model: torch.nn, phase: str) -> (torch.tensor, torch.tensor):
        assert phase in ['train', 'valid']
        zero_val = torch.tensor(0.0, dtype=torch.float32)
        if phase == 'train':
            domain = self.domain.get_train_domain()
        else:
            domain = self.domain.get_valid_domain()
        #max_loss = torch.tensor(0.0)
        with torch.set_grad_enabled(True):
            main_domain_residuals = self.get_residuals(nn_model, domain)
            max_res_norm = torch.max(torch.abs(main_domain_residuals))
            loss_val = self.norm(main_domain_residuals)
            total_loss = torch.sum(loss_val)
            for boundary_condition in self.boundary_conditions:
                boundary_residuals = boundary_condition.get_boundary_residuals(
                    nn_model
                )
                boundary_loss = torch.sum(self.boundary_coefficient * self.norm(boundary_residuals))
                boundary_res_norm = torch.max(torch.abs(boundary_residuals))
                max_res_norm = torch.max(max_res_norm, boundary_res_norm)
                total_loss += boundary_loss
        return total_loss, max_res_norm

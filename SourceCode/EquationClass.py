import abc
import torch
from typing import Callable
from .DomainClass import AbstractDomain


class AbstractEquation(abc.ABC):
    @abc.abstractmethod
    def get_residuals(self, nn_model: torch.nn, phase: str) -> torch.tensor:
        raise NotImplementedError


class OneDimensionalMainEquation(AbstractEquation):
    def __init__(
        self,
        domain: AbstractDomain,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.domain = domain
        self.equation = equation

    def get_residuals(self, nn_model: torch.nn, phase: str):
        assert phase in ['train', 'valid']
        if phase == 'train':
            domain = self.domain.get_train_domain()
        else:
            domain = self.domain.get_valid_domain()
        nn_model_value = nn_model(domain)
        residual = self.equation(domain, nn_model_value)
        return residual

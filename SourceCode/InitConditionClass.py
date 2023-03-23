import abc
import torch
from typing import Callable, Union, List


class InitialCondition(abc.ABC):
    @abc.abstractmethod
    def get_boundary_residuals(self, nn_model: torch.nn) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain_size(self):
        raise NotImplementedError


class OnePointInitialCondition(InitialCondition):
    def __init__(
        self,
        point: Union[int, float],
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.point = torch.Tensor([point])
        self.point.requires_grad = True
        self.equation = equation

    def get_boundary_residuals(self, nn_models: torch.nn) -> torch.tensor:
        return self.equation(self.point, *nn_models)

    def get_domain_size(self):
        return 1


class TwoDimInitialCondition(InitialCondition):
    def __init__(
        self,
        const_var: Union[int, float],
        non_const_var_left: Union[int, float],
        non_const_var_right: Union[int, float],
        non_const_var_size: int,
        equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        const_var_ind: int = 1,
    ):
        assert const_var_ind in (1, 2)
        self.const_var = torch.Tensor([const_var])
        self.const_var.requires_grad = True
        self.non_const_domain = torch.linspace(
            non_const_var_left, non_const_var_right, non_const_var_size
        )
        self.non_const_domain.requires_grad = True
        self.size = non_const_var_size
        self.equation = equation
        if const_var_ind == 1:
            self.xm, self.ym = torch.meshgrid((self.const_var, self.non_const_domain))
        else:
            self.xm, self.ym = torch.meshgrid((self.non_const_domain, self.const_var))

    def get_boundary_residuals(self, nn_models: List[torch.nn]) -> torch.tensor:
        return self.equation(self.xm, self.ym, *nn_models)

    def get_domain_size(self):
        return self.size

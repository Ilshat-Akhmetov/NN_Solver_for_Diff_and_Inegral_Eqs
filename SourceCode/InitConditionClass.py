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
        non_const_var_left_value: Union[int, float],
        non_const_var_right_value: Union[int, float],
        non_const_var_size: int,
        equation: Callable[[torch.tensor, torch.tensor], torch.tensor],
        const_var_value: Union[int, float],
        const_var_ind: int = 1,
    ):
        assert const_var_ind in (1, 2)
        self.equation = equation
        self.size = non_const_var_size
        self.xm, self.ym = self.generate_boundary_points(
            non_const_var_left_value,
            non_const_var_right_value,
            non_const_var_size,
            const_var_value,
            const_var_ind,
        )

    def generate_boundary_points(
        self,
        non_const_var_left_value: Union[int, float],
        non_const_var_right_value: Union[int, float],
        non_const_var_size: int,
        const_var_value: Union[int, float],
        const_var_ind: int = 1,
    ):
        const_var = torch.Tensor([const_var_value])
        non_const_domain = torch.linspace(
            non_const_var_left_value, non_const_var_right_value, non_const_var_size
        )

        if const_var_ind == 1:
            xm, ym = torch.meshgrid((const_var, non_const_domain), indexing='ij')
        else:
            xm, ym = torch.meshgrid((non_const_domain, const_var), indexing='ij')
        xm.requires_grad = True
        ym.requires_grad = True
        return xm, ym

    def get_boundary_residuals(
        self, nn_models: List[Callable[[torch.tensor, torch.tensor], torch.tensor]]
    ) -> torch.tensor:
        return self.equation(self.xm, self.ym, *nn_models)

    def get_domain_size(self):
        return self.size

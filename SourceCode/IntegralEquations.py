from typing import Callable, Union
import torch


class IntegralEquations:
    @staticmethod
    def get1DcentralrectangularNodes(
        left_bound: Union[int, float], right_bound: Union[int, float], n_points: int
    ) -> (torch.tensor, float):
        domain_arr = torch.linspace(left_bound, right_bound, n_points)
        int_nodes = (domain_arr[1:] + domain_arr[:-1]) / 2
        dx = (right_bound - left_bound) / (n_points - 1)
        return int_nodes, dx

    @staticmethod
    def calculateFredholmEquation1D(
        integral_function: Callable,
        nn_model: torch.nn,
        curr_val: torch.tensor,
        left_bound: Union[int, float],
        right_bound: Union[int, float],
        n_points: int,
    ) -> torch.tensor:
        int_nodes, dx = IntegralEquations.get1DcentralrectangularNodes(
            left_bound, right_bound, n_points
        )
        total_val = torch.zeros(curr_val.shape)
        n = len(curr_val)
        for i in range(n):
            total_val[i] = (
                torch.sum(integral_function(curr_val[i], int_nodes, nn_model)) * dx
            )
        return total_val

    @staticmethod
    def calculateVolterraEquation1D(
        integral_function: Callable,
        nn_model: torch.nn,
        curr_val: torch.tensor,
        left_bound: Union[int, float],
        right_bound: Union[int, float],
        n_points: int,
    ) -> torch.tensor:
        int_nodes, dx = IntegralEquations.get1DcentralrectangularNodes(
            left_bound, right_bound, n_points
        )
        total_val = torch.zeros(curr_val.shape)
        n = len(curr_val)
        for i in range(n):
            total_val[i] = (
                torch.sum(integral_function(curr_val[i], int_nodes[: i + 1], nn_model))
                * dx
            )
        return total_val

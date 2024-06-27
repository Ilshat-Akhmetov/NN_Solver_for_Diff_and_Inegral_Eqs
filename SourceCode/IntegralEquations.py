from typing import Callable, Union
import torch


class IntegralEquations:
    @staticmethod
    def get1d_central_rectangular_nodes(
        left_bound: Union[int, float], right_bound: Union[int, float], n_points: int
    ) -> (torch.tensor, float):
        domain_arr = torch.linspace(
            left_bound, right_bound, n_points, requires_grad=True
        )
        int_nodes = (domain_arr[1:] + domain_arr[:-1]) / 2
        dx = (right_bound - left_bound) / (n_points - 1)
        return int_nodes, dx

    @staticmethod
    def get_gauss_weights(f_map) -> (torch.tensor, float):
        weights = [
            0.3607615730481386,
            0.3607615730481386,
            0.4679139345726910,
            0.4679139345726910,
            0.1713244923791704,
            0.1713244923791704,
        ]
        points = [
            0.6612093864662645,
            -0.6612093864662645,
            -0.2386191860831969,
            0.2386191860831969,
            -0.9324695142031521,
            0.9324695142031521,
        ]
        mapped_points = [f_map(point) for point in points]
        return torch.tensor(mapped_points, requires_grad=True), torch.tensor(weights)

    @staticmethod
    def calculate_fredholm_equation_1d_gauss_quadratures(
        integral_function: Callable,
        nn_model: torch.nn,
        curr_val: torch.tensor,
        left_bound: Union[int, float],
        right_bound: Union[int, float],
    ) -> torch.tensor:
        coefficient = (right_bound - left_bound) / 2

        def f_map(x):
            return coefficient * x + (left_bound + right_bound) / 2

        nodes, weights = IntegralEquations.get_gauss_weights(f_map)
        total_val = torch.zeros(curr_val.shape)
        n = len(curr_val)
        for i in range(n):
            total_val[i] = coefficient * torch.sum(
                integral_function(curr_val[i], nodes, nn_model) * weights
            )
        return total_val

    @staticmethod
    def calculate_fredholm_equation_1d(
        integral_function: Callable,
        nn_model: torch.nn,
        curr_val: torch.tensor,
        left_bound: Union[int, float],
        right_bound: Union[int, float],
        n_points: int,
    ) -> torch.tensor:
        int_nodes, dx = IntegralEquations.get1d_central_rectangular_nodes(
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
    def calculate_volterra_equation1d(
        integral_function: Callable,
        nn_model: torch.nn,
        curr_val: torch.tensor,
        left_bound: Union[int, float],
        right_bound: Union[int, float],
        n_points: int,
    ) -> torch.tensor:
        int_nodes, dx = IntegralEquations.get1d_central_rectangular_nodes(
            left_bound, right_bound, n_points
        )
        total_val = torch.zeros(curr_val.shape)
        n = len(curr_val)
        for i in range(n):
            subarray = int_nodes[int_nodes <= curr_val[i]]
            if len(subarray) > 0:
                total_val[i] = (
                    torch.sum(integral_function(curr_val[i], subarray, nn_model)) * dx
                )
        return total_val

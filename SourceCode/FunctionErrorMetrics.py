import torch


class FunctionErrorMetrics:
    @classmethod
    def calculate_absolute_error(
        cls, analytical_solution: torch.Tensor, approximation: torch.Tensor
    ):
        return torch.abs(analytical_solution - approximation)

    @classmethod
    def calculate_max_absolute_error(
        cls, analytical_solution: torch.Tensor, approximation: torch.Tensor
    ):
        return torch.max(
            cls.calculate_absolute_error(analytical_solution, approximation)
        )

    @staticmethod
    def calculate_mean_average_precision_error(
        analytical_solution: torch.Tensor, approximation: torch.Tensor
    ):
        return torch.sum(torch.abs(analytical_solution - approximation)) / torch.sum(
            torch.abs(analytical_solution)
        )

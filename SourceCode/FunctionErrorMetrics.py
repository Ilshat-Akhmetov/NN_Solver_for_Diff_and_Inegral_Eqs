import numpy as np


class FunctionErrorMetrics:
    @classmethod
    def calculate_absolute_error(
        cls, analytical_solution: np.array, approximation: np.array
    ):
        return np.abs(analytical_solution - approximation)

    @classmethod
    def calculate_max_absolute_error(
        cls, analytical_solution: np.array, approximation: np.array
    ):
        return np.max(
            cls.calculate_absolute_error(analytical_solution, approximation)
        )

    @staticmethod
    def calculate_mean_average_precision_error(
        analytical_solution: np.array, approximation: np.array
    ):
        return np.sum(np.abs(analytical_solution - approximation)) / np.sum(
            np.abs(analytical_solution)
        )

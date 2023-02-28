import numpy as np


class FunctionErrorMetrics:
    @classmethod
    def calculate_absolute_error(
            cls, analytical_solution: np.array, approximation: np.array
    ) -> np.array:
        ans = np.linalg.norm(analytical_solution - approximation, axis=0)
        ans = ans.astype(np.float32)
        return ans

    @classmethod
    def calculate_max_absolute_error(
            cls, analytical_solution: np.array, approximation: np.array
    ) -> np.array:
        return np.max(
            cls.calculate_absolute_error(analytical_solution, approximation)
        )

    @classmethod
    def calculate_mean_average_precision_error(cls,
                                               analytical_solution: np.array, approximation: np.array
                                               ) -> np.array:
        return np.sum(cls.calculate_absolute_error(analytical_solution, approximation)) / np.sum(
            np.abs(analytical_solution)
        )

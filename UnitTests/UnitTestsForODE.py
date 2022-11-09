import unittest
import sys
from SourceCode.FunctionErrorMetrics import FunctionErrorMetrics
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import *
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
sys.path.append("..")


class NNSolverForODETest(unittest.TestCase):
    def test_ode1(self):
        left_bound = 0
        right_bound = 1
        main_eq_residual = (
            lambda x, nn_model_value: nth_derivative(nn_model_value, x, 2)
                                      + 0.2 * nth_derivative(nn_model_value, x, 1)
                                      + nn_model_value
                                      + 0.2 * torch.exp(-x / 5) * torch.cos(x)
        )
        n_points = 100
        main_domain = OneDimensionalSimpleDomain(left_bound, right_bound, n_points)
        main_eq = OneDimensionalMainEquation(main_domain, main_eq_residual)

        first_init_cond_res = lambda x, nn_model_value: nn_model_value - 0
        first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

        second_init_cond_res = lambda x, nn_model_value: nn_model_value - torch.sin(
            torch.Tensor([1])
        ) * torch.exp(torch.Tensor([-0.2]))
        second_init_cond = OnePointInitialCondition(right_bound, second_init_cond_res)

        boundary_conditions = [first_init_cond, second_init_cond]

        true_solution = lambda x: torch.exp(-x / 5) * torch.sin(x)
        nn_ode_solver = TrainerForNNEquationSolver(main_eq, boundary_conditions)
        __, _, nn_model = nn_ode_solver.fit(verbose=False)
        valid_domain = main_domain.get_valid_domain()
        true_function_value = true_solution(valid_domain)
        approximation = nn_model(valid_domain)
        max_abs_error = FunctionErrorMetrics.calculate_max_absolute_error(true_function_value, approximation)
        benchmark = 5e-5
        message = f'{max_abs_error} is not less than {benchmark}'
        self.assertLess(max_abs_error, benchmark, message)
        mape = 100 * FunctionErrorMetrics.calculate_mean_average_precision_error(true_function_value, approximation)
        benchmark = 1e-2
        message = f'{mape} is not less than {benchmark}'
        self.assertLess(mape, benchmark, message)

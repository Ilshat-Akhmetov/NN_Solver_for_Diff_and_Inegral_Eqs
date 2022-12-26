import unittest
import sys
import torch
from SourceCode.FunctionErrorMetrics import FunctionErrorMetrics
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import OneDimensionalMainEquation
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.InitConditionClass import OnePointInitialCondition
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
sys.path.append("..")


class NNSolverForODETest(unittest.TestCase):
    def setUp(self) -> None:
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
        self.true_function_value = true_solution(valid_domain)
        self.approximation = nn_model(valid_domain)

    def test_ode1_abs_error(self):
        max_abs_error = FunctionErrorMetrics.calculate_max_absolute_error(self.true_function_value, self.approximation)
        abs_err_benchmark = 2e-3
        message = f'{max_abs_error} is not less than {abs_err_benchmark}'
        self.assertLess(max_abs_error, abs_err_benchmark, message)

    def test_ode1_mape(self):
        mape = 100 * FunctionErrorMetrics.calculate_mean_average_precision_error(self.true_function_value,
                                                                                 self.approximation)
        mape_benchmark = 2e-3
        message = f'{mape} is not less than {mape_benchmark}'
        self.assertLess(mape, mape_benchmark, message)

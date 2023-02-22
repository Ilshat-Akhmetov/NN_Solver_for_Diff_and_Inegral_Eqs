import unittest
import sys
import torch
from SourceCode.FunctionErrorMetrics import FunctionErrorMetrics
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import OneDimensionalMainEquation
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.InitConditionClass import OnePointInitialCondition
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker

sys.path.append("..")


class NNSolverForODETest(unittest.TestCase):
    def setUp(self) -> None:
        left_bound = 0
        right_bound = 1
        main_eq_residual = (
            lambda x, nn_appr: nth_derivative(nn_appr, x, 2)
            + 0.2 * nth_derivative(nn_appr, x, 1)
            + nn_appr(x)
            + 0.2 * torch.exp(-x / 5) * torch.cos(x)
        )
        n_points = 20
        n_epochs = 20

        first_init_cond_res = lambda x, nn_appr: nn_appr(x) - 0
        first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

        second_init_cond_res = lambda x, nn_appr: nn_appr(x) - torch.sin(
            torch.Tensor([1])
        ) * torch.exp(torch.Tensor([-0.2]))
        second_init_cond = OnePointInitialCondition(right_bound, second_init_cond_res)

        boundary_conditions = [first_init_cond, second_init_cond]
        main_domain = OneDimensionalSimpleDomain(left_bound, right_bound, n_points)
        main_eq = OneDimensionalMainEquation(
            main_domain, main_eq_residual, boundary_conditions
        )

        true_solution = lambda x: torch.exp(-x / 5) * torch.sin(x)
        nn_ode_solver = TrainerForNNEquationSolver(main_eq)
        loss_train, loss_valid, nn_models = nn_ode_solver.fit(verbose=False)
        report = ReportMaker(
            true_solution,
            nn_models,
            main_eq,
            loss_train,
            loss_valid,
            main_domain,
            num_epochs=n_epochs,
        )
        (
            self.valid_domain,
            self.approximation,
            self.analytical_solution,
        ) = report.get_domain_target("valid")
        self.approximation = self.approximation[0]
        self.analytical_solution = self.analytical_solution[0]

    def test_ode1_abs_error(self) -> None:
        max_abs_error = FunctionErrorMetrics.calculate_max_absolute_error(
            self.analytical_solution, self.approximation
        )
        abs_err_benchmark = 1e-4
        message = f"{max_abs_error} is not less than {abs_err_benchmark}"
        self.assertLess(max_abs_error, abs_err_benchmark, message)

    def test_ode1_mape(self) -> None:
        mape = 100 * FunctionErrorMetrics.calculate_mean_average_precision_error(
            self.analytical_solution, self.approximation
        )
        mape_benchmark = 1e-3
        message = f"{mape} is not less than {mape_benchmark}"
        self.assertLess(mape, mape_benchmark, message)

import torch
from typing import Callable, List, Union
from .utilities import plot_two_1d_functions, plot_1d_function

from .EquationClass import AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics
import numpy as np
from pandas import DataFrame
from .utilities import torch_to_numpy, get_domain_target


class ReportMaker:
    def __init__(
        self,
        nn_models: List[Callable[[torch.tensor], torch.tensor]],
        domain: AbstractDomain,
        compare_to_functions: Callable = plot_two_1d_functions,
        analytical_solutions: Union[
            List[Callable[[torch.tensor], torch.tensor]],
            Callable[[torch.tensor], torch.tensor],
        ] = None,
        main_eq_residuals: Callable = None,
    ):
        if (
            not isinstance(analytical_solutions, list)
            and analytical_solutions is not None
        ):
            self.analytical_solutions = [analytical_solutions]
        else:
            self.analytical_solutions = analytical_solutions
        self.nn_models = nn_models
        self.domain = domain
        self.compare_two_functions = compare_to_functions
        self.main_eq_residuals = main_eq_residuals
        if not isinstance(self.main_eq_residuals, list):
            self.main_eq_residuals = [self.main_eq_residuals]


    def get_residuals_values(self, phase: str = "train", offset: float = 1e-2):
        title = "abs res value on {} distr".format(phase)
        domain = self.domain.get_domain_copy(phase, offset=offset)
        n = len(self.main_eq_residuals)
        residuals = torch.zeros((n, *domain[0].shape))
        zero_arr = np.zeros((n, *domain[0].shape))
        for i in range(n):
            residuals[i] = self.main_eq_residuals[i](*domain, *self.nn_models)
        residuals = torch_to_numpy(residuals)
        for i in range(len(domain)):
            domain[i] = torch_to_numpy(domain[i])
        abs_residuals = FunctionErrorMetrics.calculate_absolute_error(
            residuals, zero_arr
        )
        return domain, abs_residuals, title

    def plot_abs_residual_distr(
        self, phase: str = "train", figsize: tuple = (9, 8), offset: float = 1e-2
    ) -> None:
        if self.main_eq_residuals[0] is None:
            raise ValueError("You have to provide main equation in residual form")
        assert phase in ["train", "valid"], "data array may be only train or valid"
        domain, abs_residuals, title = self.get_residuals_values(phase, offset)
        self.domain.plot_error_distribution(
            domain, abs_residuals, title, figsize=figsize
        )

    def plot_error_history(
        self,
        error_history: np.ndarray,
        title: str = 'Max error on train',
        figsize: tuple = (9, 8),
    ) -> None:
        epochs = torch.arange(len(error_history))
        plot_1d_function(
            epochs,
            error_history,
            title,
            "epoch",
            "abs value",
            figsize=figsize,
        )
        print("Value at last epoch: {}".format(error_history[-1]))

    def compare_appr_with_analytical(
        self, figsize: tuple = (9, 8), phase: str = "train", offset=1e-2
    ) -> None:
        if self.analytical_solutions is None:
            raise ValueError(
                "You have to provide analytical solution to compare it with the approximation"
            )
        (
            train_domain,
            nn_approximation_train,
            analytical_solution_train,
        ) = get_domain_target(
                    self.domain,
                    self.nn_models,
                    'train',
                    self.domain.offset,
                    self.analytical_solutions
                )
        (
            valid_domain,
            nn_approximation_valid,
            analytical_solution_valid,
        ) = get_domain_target(
                    self.domain,
                    self.nn_models,
                    'valid',
                    self.domain.offset,
                    self.analytical_solutions
                )
        abs_error_train = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_train, nn_approximation_train
        )

        if self.main_eq_residuals[0] is not None:
            _, residuals, __ = self.get_residuals_values(phase, offset)
            max_res_val = np.max(np.abs(residuals))
            print("Max residual value |R[NN]| on {}: {}".format(phase, max_res_val))
        print("Comparison of approximation and analytical solution:")
        print(
            "Train max absolute error |Appr(x)-y(x)|: {}".format(
                FunctionErrorMetrics.calculate_max_absolute_error(
                    analytical_solution_train, nn_approximation_train
                )
            )
        )

        print(
            "Valid max absolute error |Appr(x)-y(x)|: {}".format(
                FunctionErrorMetrics.calculate_max_absolute_error(
                    analytical_solution_valid, nn_approximation_valid
                )
            )
        )

        print(
            "MAPE on train data: {} %".format(
                100
                * FunctionErrorMetrics.calculate_mean_average_precision_error(
                    analytical_solution_train, nn_approximation_train
                )
            )
        )

        print(
            "MAPE on validation data: {} %".format(
                100
                * FunctionErrorMetrics.calculate_mean_average_precision_error(
                    analytical_solution_valid, nn_approximation_valid
                )
            )
        )
        self.domain.plot_error_distribution(
            train_domain, abs_error_train, figsize=figsize
        )
        if self.compare_two_functions is not None:
            self.compare_two_functions(
                valid_domain,
                analytical_solution_valid,
                nn_approximation_valid,
                "Compare True func Vs Approximation",
                "Analytical sol",
                "Approximation",
                figsize=figsize,
            )


    def print_comparison_table(
        self, domain_data: str = "train", filename="comparison.csv"
    ) -> None:
        assert domain_data in ["valid", "train"]
        assert self.analytical_solutions is not None
        print("{} data".format(domain_data))
        domain, appr_val, analytical_val = get_domain_target(
                    self.domain,
                    self.nn_models,
                    domain_data,
                    self.domain.offset,
                    self.analytical_solutions
                )

        data = dict()
        dimensionality = len(domain)
        if dimensionality == 1:
            data["Input_X"] = np.ravel(domain[0])
        else:
            for i in range(dimensionality):
                new_name = "Input_X{}".format(i + 1)
                data[new_name] = np.ravel(domain[i])
        n_outputs = len(self.nn_models)
        if n_outputs == 1:
            if self.analytical_solutions is not None:
                data["Analytical_F"] = np.ravel(analytical_val)
            data["ANN_F"] = np.ravel(appr_val)
        else:
            for i in range(n_outputs):
                if self.analytical_solutions is not None:
                    data["Analytical_F{}".format(i + 1)] = np.ravel(analytical_val[i])
                data["ANN_F{}".format(i + 1)] = np.ravel(appr_val[i])
        if self.analytical_solutions is not None:
            error = FunctionErrorMetrics.calculate_absolute_error(
                appr_val, analytical_val
            )
            data["Abs error"] = np.ravel(error)
        df = DataFrame(data=data)
        print(df)
        df.to_csv(filename, index_label="obs")

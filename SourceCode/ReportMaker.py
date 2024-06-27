import torch
from typing import Callable, List, Union
from .utilities import plot_two_1d_functions, plot_1d_function

from .EquationClass import AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics
import numpy as np
from pandas import DataFrame
from .utilities import torch_to_numpy


class ReportMaker:
    def __init__(
        self,
        nn_models: List[Callable[[torch.tensor], torch.tensor]],
        loss_history_train: torch.Tensor,
        loss_history_valid: torch.Tensor,
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
        self.loss_history_train = torch_to_numpy(loss_history_train)
        self.loss_history_valid = torch_to_numpy(loss_history_valid)
        num_epochs = len(self.loss_history_train)
        self.epochs = torch.arange(num_epochs)
        self.compare_two_functions = compare_to_functions
        self.plot_1d_function = plot_1d_function
        self.main_eq_residuals = main_eq_residuals
        if not isinstance(self.main_eq_residuals, list):
            self.main_eq_residuals = [self.main_eq_residuals]

    def get_domain_target(
        self, domain_data: str = "train", offset: float = 1e-2
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        return self.domain.get_domain_and_target(
            domain_data, offset, self.nn_models, self.analytical_solutions
        )

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

    def print_loss_history(self, phase: str = "train", figsize: tuple = (9, 8)) -> None:
        """
        This method plots loss history of NN's fitting to the given equation
        :param phase: str: phase of loss history you want to plot. It can be either train or valid
        :param figsize: str: size of the plot. must be a tuple
        :return: None
        """
        assert phase in ["train", "valid"]
        if phase == "train":
            loss = self.loss_history_train
        else:
            loss = self.loss_history_valid
        self.plot_1d_function(
            self.epochs,
            loss,
            "Max abs residual value on {}".format(phase),
            "epoch",
            "abs value",
            figsize=figsize,
        )

    def plot_abs_error_history(
        self,
        abs_error: np.ndarray,
        phase: str = "train",
        figsize: tuple = (9, 8),
    ) -> None:
        self.plot_1d_function(
            self.epochs,
            abs_error,
            "Max abs error max|appr(x)-u(x)| on {}".format(phase),
            "epoch",
            "abs value",
            figsize=figsize,
        )

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
        ) = self.get_domain_target(offset=offset)
        (
            valid_domain,
            nn_approximation_valid,
            analytical_solution_valid,
        ) = self.get_domain_target("valid", offset=offset)
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

        print(
            "Max abs value of residual on train at last epoch: {} ".format(
                self.loss_history_train[-1]
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
        print("{} data".format(domain_data))
        domain, appr_val, analytical_val = self.get_domain_target(domain_data)

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

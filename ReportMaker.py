from typing import Callable
from utilities import plot_one_dimensional_function
from EquationAndDomain import AbstractEquation
from FunctionErrorMetrics import FunctionErrorMetrics
import torch


class ReportMaker:
    def __init__(
            self,
            true_solution: Callable[[torch.Tensor], torch.Tensor],
            nn_model: torch.nn,
            main_eq: AbstractEquation,
            mse_loss_train: torch.Tensor,
            mse_loss_valid: torch.Tensor,
            num_epochs: int = 200,
            plot_graph_function: Callable[
                [torch.Tensor, torch.Tensor, str, str, str], None
            ] = plot_one_dimensional_function,
    ):
        self.true_solution = true_solution
        self.nn_model = nn_model
        self.main_eq = main_eq
        self.mse_loss_train = mse_loss_train
        self.mse_loss_valid = mse_loss_valid
        self.num_epochs = num_epochs
        self.plot_graph_function = plot_graph_function

    def make_report(self) -> None:
        self.nn_model.eval()
        train_domain = self.main_eq.get_train_domain()
        valid_domain = self.main_eq.get_valid_domain()
        analytical_solution_valid = torch.Tensor(self.true_solution(valid_domain))
        analytical_solution_train = torch.Tensor(self.true_solution(train_domain))

        nn_approximation_valid = self.nn_model(valid_domain)
        nn_approximation_train = self.nn_model(train_domain)

        print(
            "Train max absolute error: {}".format(
                FunctionErrorMetrics.calculate_max_absolute_error(
                    analytical_solution_train, nn_approximation_train
                )
            )
        )

        print(
            "Valid max absolute error: {}".format(
                FunctionErrorMetrics.calculate_max_absolute_error(
                    analytical_solution_valid, nn_approximation_valid
                )
            )
        )

        print(
            "Mean average percentage error on train data: {} %".format(
                100
                * FunctionErrorMetrics.calculate_mean_average_precision_error(
                    analytical_solution_train, nn_approximation_train
                )
            )
        )

        print(
            "Mean average percentage error on validation data: {} %".format(
                100
                * FunctionErrorMetrics.calculate_mean_average_precision_error(
                    analytical_solution_valid, nn_approximation_valid
                )
            )
        )

        abs_error_valid = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_valid, nn_approximation_valid
        )
        plot_one_dimensional_function(
            valid_domain,
            abs_error_valid,
            "Absolute error on validation domain: true sol - Approximation",
            "X",
            "Error",
        )

        abs_error_train = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_train, nn_approximation_train
        )
        plot_one_dimensional_function(
            train_domain,
            abs_error_train,
            "Absolute error on train domain: true sol - Approximation",
            "X",
            "Error",
        )

        self.plot_graph_function(
            valid_domain,
            analytical_solution_valid,
            "True Solution",
            "domain_value",
            "Function_value",
        )
        self.plot_graph_function(
            valid_domain,
            nn_approximation_valid,
            "Approximation",
            "domain_value",
            "Function_value",
        )

        epochs = torch.arange(self.num_epochs)

        plot_one_dimensional_function(
            epochs, self.mse_loss_valid, "MSE loss validation", "epoch", "loss"
        )
        plot_one_dimensional_function(
            epochs, self.mse_loss_train, "MSE loss train", "epoch", "loss"
        )

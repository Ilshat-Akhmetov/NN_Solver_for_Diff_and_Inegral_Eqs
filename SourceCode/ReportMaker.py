import torch
from typing import Callable
from .utilities import plot_1d_function, plot_two_1d_functions
from .EquationClass import AbstractEquation, AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics

class ReportMaker:
    def __init__(
            self,
            true_solution: Callable[[torch.Tensor], torch.Tensor],
            nn_model: torch.nn,
            main_eq: AbstractEquation,
            mse_loss_train: torch.Tensor,
            mse_loss_valid: torch.Tensor,
            domain: AbstractDomain,
            num_epochs: int = 200,
            plot_graph_function: Callable[
                [torch.Tensor, torch.Tensor, str, str, str], None
            ] = plot_1d_function,
    ):
        self.true_solution = true_solution
        self.nn_model = nn_model
        self.main_eq = main_eq
        self.mse_loss_train = mse_loss_train
        self.mse_loss_valid = mse_loss_valid
        self.domain = domain
        self.num_epochs = num_epochs
        self.plot_graph_function = plot_graph_function

    def make_report(self) -> None:
        self.nn_model.eval()
        train_domain = self.domain.get_train_domain()
        valid_domain = self.domain.get_valid_domain()
        analytical_solution_valid = torch.Tensor(self.true_solution(valid_domain))
        analytical_solution_train = torch.Tensor(self.true_solution(train_domain))

        nn_approximation_valid = self.nn_model(valid_domain)
        nn_approximation_train = self.nn_model(train_domain)

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
            "Max residual square loss on train at last epoch: {} ".format(self.mse_loss_train[-1])
        )

        abs_error_train = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_train, nn_approximation_train
        )
        plot_1d_function(
            train_domain,
            abs_error_train,
            "Absolute error on train data: |Appr(x)-y(x)|",
            "X",
            "Error",
        )

        plot_two_1d_functions(valid_domain,
                              analytical_solution_valid,
                              nn_approximation_valid,
                              "Compare True func Vs Approximation",
                              "domain",
                              "True",
                              "Approximation")

        epochs = torch.arange(self.num_epochs)

        plot_1d_function(
            epochs, self.mse_loss_train, "Max abs residual value on train", "epoch", "loss"
        )

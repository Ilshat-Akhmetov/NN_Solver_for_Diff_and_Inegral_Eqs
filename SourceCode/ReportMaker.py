import torch
from typing import Callable
from .utilities import plot_1d_function, plot_two_1d_functions
from .EquationClass import AbstractEquation, AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics
from numpy import array as np_array
from numpy import abs
from numpy import set_printoptions, ravel
from pandas import DataFrame


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
        self.nn_model.eval()
        self.main_eq = main_eq
        self.mse_loss_train = self.torch_to_numpy(mse_loss_train)
        self.mse_loss_valid = self.torch_to_numpy(mse_loss_valid)
        self.domain = domain
        self.num_epochs = num_epochs
        self.plot_graph_function = plot_graph_function

    def torch_to_numpy(self, arr: torch) -> np_array:
        return arr.cpu().detach().numpy()

    def make_report(self) -> None:
        train_domain = self.domain.get_train_domain()
        valid_domain = self.domain.get_valid_domain()

        analytical_solution_valid = torch.Tensor(self.true_solution(valid_domain))
        analytical_solution_valid = self.torch_to_numpy(analytical_solution_valid)
        analytical_solution_train = torch.Tensor(self.true_solution(train_domain))
        analytical_solution_train = self.torch_to_numpy(analytical_solution_train)

        nn_approximation_valid = self.nn_model(valid_domain)
        nn_approximation_valid = self.torch_to_numpy(nn_approximation_valid)
        nn_approximation_train = self.nn_model(train_domain)
        nn_approximation_train = self.torch_to_numpy(nn_approximation_train)

        train_domain = self.torch_to_numpy(train_domain)
        valid_domain = self.torch_to_numpy(valid_domain)

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

    def print_comparison_table(self, domain_data: str = 'train'):
        assert domain_data in ['train', 'valid']
        if domain_data == 'train':
            domain = self.domain.get_train_domain()
        else:
            domain = self.domain.get_valid_domain()
        appr_val = self.nn_model(domain)
        analyt_val = self.true_solution(domain)
        domain = self.torch_to_numpy(domain)
        appr_val = self.torch_to_numpy(appr_val)
        analyt_val = self.torch_to_numpy(analyt_val)
        error = ravel(abs(appr_val - analyt_val))
        data = {"Input": ravel(domain), "Analytical": ravel(analyt_val), "ANN": ravel(appr_val), "Error": ravel(error)}
        df = DataFrame(data=data)
        print(df)

import torch
from typing import Callable
from .utilities import plot_two_1d_functions
from .EquationClass import AbstractEquation, AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


class ReportMaker:
    def __init__(
            self,
            true_solutions: Callable[[torch.Tensor], torch.Tensor],
            nn_models: torch.nn,
            main_eq: AbstractEquation,
            mse_loss_train: torch.Tensor,
            mse_loss_valid: torch.Tensor,
            domain: AbstractDomain,
            num_epochs: int = 20,
            plot2functions: Callable = plot_two_1d_functions,
            do_plot_func: bool = True
    ):
        if not isinstance(true_solutions, list):
            self.true_solutions = [true_solutions]
        else:
            self.true_solutions = true_solutions
        self.nn_models = nn_models
        self.nn_models = [nn_model.eval() for nn_model in self.nn_models]
        self.main_eq = main_eq
        self.mse_loss_train = self.torch_to_numpy(mse_loss_train)
        self.mse_loss_valid = self.torch_to_numpy(mse_loss_valid)
        self.domain = domain
        self.epochs = torch.arange(num_epochs)
        self.do_plot_func = do_plot_func
        self.plot_two_functions = plot2functions

    @staticmethod
    def torch_to_numpy(arr: torch) -> np.array:
        return arr.cpu().detach().numpy()

    @staticmethod
    def plot_1d_function(
            x_value: torch.Tensor, y_value: torch.Tensor, title: str, x_label: str, y_label: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.plot(x_value, y_value)
        plt.show()

    @staticmethod
    def get_func_value(funcs, domain: torch.tensor) -> torch.tensor:
        result = torch.zeros((len(funcs), *domain.shape))
        for i, func in enumerate(funcs):
            result[i] = func(domain)
        return result

    def get_domain_target(self, domain_data: str = 'train'):
        assert domain_data in ['train', 'valid']
        if domain_data == 'train':
            domain = self.domain.get_train_domain()
        else:
            domain = self.domain.get_valid_domain()
        appr_val = ReportMaker.get_func_value(self.nn_models, domain)
        analyt_val = ReportMaker.get_func_value(self.true_solutions, domain)
        domain = ReportMaker.torch_to_numpy(domain)
        appr_val = ReportMaker.torch_to_numpy(appr_val)
        analyt_val = ReportMaker.torch_to_numpy(analyt_val)
        return domain, appr_val, analyt_val

    def make_report(self) -> None:
        train_domain, nn_approximation_train, analytical_solution_train = self.get_domain_target()
        valid_domain, nn_approximation_valid, analytical_solution_valid = self.get_domain_target("valid")

        abs_error_train = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_train, nn_approximation_train
        )

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

        ReportMaker.plot_1d_function(
            self.epochs, self.mse_loss_train, "Max abs residual value on train", "epoch", "abs value"
        )

        ReportMaker.plot_1d_function(
            train_domain,
            abs_error_train,
            "Absolute error on train data: |Appr(x)-y(x)|",
            "X",
            "Error",
        )

        if self.do_plot_func:
            self.plot_two_functions(valid_domain,
                                    analytical_solution_valid,
                                    nn_approximation_valid,
                                    "Compare True func Vs Approximation",
                                    "domain",
                                    "True",
                                    "Approximation")

    def print_comparison_table(self, domain_data: str = 'train', filename='comparison.csv'):
        if domain_data == "train":
            print("train data")
            domain, appr_val, analyt_val = self.get_domain_target()
        else:
            print("valid data")
            domain, appr_val, analyt_val = self.get_domain_target("valid")
        error = FunctionErrorMetrics.calculate_absolute_error(appr_val, analyt_val)
        data = {}
        data["Input"] = np.ravel(domain)
        n_outputs = len(analyt_val)
        if n_outputs == 1:
            data["Analytical"] = np.ravel(analyt_val)
            data["ANN"] = np.ravel(appr_val)
        else:
            for i in range(n_outputs):
                data["Analytical_x{}".format(i + 1)] = np.ravel(analyt_val[i])
                data["ANN_x{}".format(i + 1)] = np.ravel(appr_val[i])
        data["Error"] = np.ravel(error)
        df = DataFrame(data=data)
        print(df)
        df.to_csv(filename, index_label='obs')

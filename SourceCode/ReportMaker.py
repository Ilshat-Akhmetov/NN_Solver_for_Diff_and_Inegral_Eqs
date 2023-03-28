import torch
from typing import Callable, List, Union
from .utilities import plot_two_1d_functions, plot_1d_function

from .EquationClass import AbstractDomain
from .FunctionErrorMetrics import FunctionErrorMetrics
import numpy as np
from pandas import DataFrame


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
        main_eq_residuals: Callable = None
    ):
        if not isinstance(analytical_solutions, list):
            self.analytical_solutions = [analytical_solutions]
        else:
            self.analytical_solutions = analytical_solutions
        self.nn_models = nn_models
        (nn_model.eval() for nn_model in self.nn_models)
        self.loss_history_train = self.torch_to_numpy(loss_history_train)
        self.loss_history_valid = self.torch_to_numpy(loss_history_valid)
        self.domain = domain
        num_epochs = len(self.loss_history_train)
        self.epochs = torch.arange(num_epochs)
        self.compare_two_functions = compare_to_functions
        self.plot_1d_function = plot_1d_function
        self.main_eq_residuals = main_eq_residuals
        if not isinstance(self.main_eq_residuals, list):
            self.main_eq_residuals = [self.main_eq_residuals]
    @staticmethod
    def torch_to_numpy(arr: torch.tensor) -> np.array:
        return arr.cpu().detach().numpy()

    @staticmethod
    def get_func_value(funcs, domain: List[torch.tensor]) -> torch.tensor:
        result = torch.zeros((len(funcs), *domain[0].shape))
        for i, func in enumerate(funcs):
            result[i] = func(*domain)
        return result

    def get_domain_target(
        self, domain_data: str = "train"
    ) -> (torch.tensor, torch.tensor, torch.tensor):
        assert domain_data in ["train", "valid"]
        if domain_data == "train":
            domain: list = self.domain.get_domain_copy('train')
        else:
            domain: list = self.domain.get_domain_copy('valid')
        appr_val = ReportMaker.get_func_value(self.nn_models, domain)
        analytical_val = ReportMaker.get_func_value(self.analytical_solutions, domain)
        domain = [ReportMaker.torch_to_numpy(domain_part) for domain_part in domain]
        appr_val = ReportMaker.torch_to_numpy(appr_val)
        analytical_val = ReportMaker.torch_to_numpy(analytical_val)
        return domain, appr_val, analytical_val

    def plot_abs_residual_distr(self, phase: str='train') -> None:
        if self.main_eq_residuals[0] is None:
            raise ValueError("You have to provide main equation in residual form")
        assert phase in ['train', 'test'], "datarray may be only train or valid"
        if phase == 'train':
            title = "abs res value on train distr"
            domain = self.domain.get_domain_copy()
        else:
            title = "abs res value on valid distr"
            domain = self.domain.get_domain_copy('valid')
        n = len(self.main_eq_residuals)
        residuals = torch.zeros((n, *domain[0].shape))
        zero_arr = np.zeros((n, *domain[0].shape))
        for i in range(n):
            residuals[i] = self.main_eq_residuals[i](*domain, *self.nn_models)
        residuals = self.torch_to_numpy(residuals)
        for i in range(len(domain)):
            domain[i] = self.torch_to_numpy(domain[i])
        abs_residuals = FunctionErrorMetrics.calculate_absolute_error(
            residuals, zero_arr
        )
        self.domain.plot_error_distribution(domain, abs_residuals, title)


    def print_loss_history(self, phase: str = "train"):
        assert phase in ["train", "valid"]
        if phase == "train":
            loss = self.loss_history_train
        else:
            loss = self.loss_history_valid
        self.plot_1d_function(
            self.epochs, loss, "Max abs residual value on train", "epoch", "abs value"
        )

    def compare_appr_with_analytical(self) -> None:
        if self.analytical_solutions is None:
            raise ValueError(
                "You have to provide analytical solution to compare it with the approximation"
            )
        (
            train_domain,
            nn_approximation_train,
            analytical_solution_train,
        ) = self.get_domain_target()
        (
            valid_domain,
            nn_approximation_valid,
            analytical_solution_valid,
        ) = self.get_domain_target("valid")
        abs_error_train = FunctionErrorMetrics.calculate_absolute_error(
            analytical_solution_train, nn_approximation_train
        )
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

        self.domain.plot_error_distribution(train_domain, abs_error_train)

        if self.compare_two_functions is not None:
            self.compare_two_functions(
                valid_domain,
                analytical_solution_valid,
                nn_approximation_valid,
                "Compare True func Vs Approximation",
                "Analytical sol",
                "Approximation",
            )


    def print_comparison_table(
        self, domain_data: str = "train", filename="comparison.csv"
    ) -> None:
        if domain_data == "train":
            print("train data")
            domain, appr_val, analytical_val = self.get_domain_target()
        else:
            print("valid data")
            domain, appr_val, analytical_val = self.get_domain_target("valid")
        error = FunctionErrorMetrics.calculate_absolute_error(appr_val, analytical_val)
        data = dict()
        dimensionality = len(domain)
        if dimensionality == 1:
            data["Input_X"] = np.ravel(domain[0])
        else:
            for i in range(dimensionality):
                new_name = "Input_X{}".format(i + 1)
                data[new_name] = np.ravel(domain[i])
        n_outputs = len(analytical_val)
        if n_outputs == 1:
            data["Analytical_F"] = np.ravel(analytical_val)
            data["ANN_F"] = np.ravel(appr_val)
        else:
            for i in range(n_outputs):
                data["Analytical_F{}".format(i + 1)] = np.ravel(analytical_val[i])
                data["ANN_F{}".format(i + 1)] = np.ravel(appr_val[i])
        data["Abs error"] = np.ravel(error)
        df = DataFrame(data=data)
        print(df)
        df.to_csv(filename, index_label="obs")

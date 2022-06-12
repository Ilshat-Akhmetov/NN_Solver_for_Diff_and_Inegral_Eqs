import torch
from CustomNeuralNetworkClass import CustomNeuralNetworkClass
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import random


class NNSolver:
    def __init__(
            self, main_eq, init_conditions, true_solution: Callable[[float], float]
    ):
        self.set_seed(77)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_eq = main_eq
        self.init_conditions = init_conditions
        self.batch_size = 1
        self.loss = lambda x: torch.mean(torch.pow(x, 2))
        n_inputs = 1
        n_hidden_neurons = 100
        n_outputs = 1
        self.nn_model = CustomNeuralNetworkClass(n_inputs, n_hidden_neurons, n_outputs)
        # self.nn_model.to(self.device)
        self.num_epochs = 200
        self.mse_loss_train = torch.zeros(self.num_epochs)
        self.mse_loss_valid = torch.zeros(self.num_epochs)
        self.optimizer = torch.optim.Adam(
            self.nn_model.parameters(), lr=5e-3  # , betas=(0.99, 0.9999)
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=1
        )
        self.true_solution = true_solution
        self.AbsoluteError = lambda analytical_solution, approximation: torch.abs(
            analytical_solution - approximation
        )
        self.MaxAbsoluteError = lambda analytical_solution, approximation: torch.max(
            self.AbsoluteError(analytical_solution, approximation)
        )
        self.MAPE = (
            lambda analytical_solution, approximation: torch.sum(
                torch.abs(analytical_solution - approximation)
            ) / torch.sum(torch.abs(analytical_solution))
        )

    @staticmethod
    def set_seed(seed: int = 77) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def fit(self) -> None:
        train_domain = self.main_eq.get_train_domain()
        train_domain = train_domain.reshape(-1, 1)
        valid_domain = self.main_eq.get_valid_domain()
        valid_domain = valid_domain.reshape(-1, 1)
        self.nn_model.train()
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}:".format(epoch, self.num_epochs - 1), flush=True)
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    domain = train_domain
                    self.nn_model.train()  # Set model to training mode
                else:
                    domain = valid_domain
                    self.nn_model.eval()  # Set model to evaluate mode

                epoch_loss = self.get_loss(domain, phase)
                if phase == "train":
                    self.scheduler.step()
                    self.mse_loss_train[epoch] = epoch_loss
                else:
                    self.mse_loss_valid[epoch] = epoch_loss

                print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
        self.nn_model.eval()

    def get_loss(self, domain: torch.Tensor, phase: str) -> torch.tensor:
        boundary_coefficient = 1.0
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            domain.requires_grad = True
            nn_model_prediction = self.nn_model(domain)
            residual = self.main_eq.get_value(domain, nn_model_prediction)
            epoch_loss = self.loss(residual)

            for init_condition in self.init_conditions:
                boundary_residual = init_condition.get_value(self.nn_model)
                epoch_loss += boundary_coefficient * self.loss(boundary_residual)

                # backward + optimize only if in training phase
            if phase == "train":
                epoch_loss.backward(retain_graph=True)
                self.optimizer.step()
        return epoch_loss

    @staticmethod
    def plot_function(domain: torch.tensor, function: callable, title: str, x_label: str, y_label: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.plot(domain.cpu().detach().numpy(), function.cpu().detach().numpy())
        plt.show()

    def make_report(self) -> None:
        self.nn_model.eval()
        train_domain = self.main_eq.get_train_domain()
        valid_domain = self.main_eq.get_valid_domain()
        analytical_solution_valid = self.true_solution(valid_domain)
        analytical_solution_train = self.true_solution(train_domain)

        nn_approximation_valid = self.nn_model(
            valid_domain.view(self.main_eq.n_points - 1, 1)
        )
        nn_approximation_train = self.nn_model(
            train_domain.view(self.main_eq.n_points, 1)
        )

        print(
            "Train max absolute error: {}".format(
                self.MaxAbsoluteError(analytical_solution_train, nn_approximation_train)
            )
        )

        print(
            "Valid max absolute error: {}".format(
                self.MaxAbsoluteError(analytical_solution_valid, nn_approximation_valid)
            )
        )

        print(
            "Mean average percentage error on train data: {} %".format(
                100 * self.MAPE(analytical_solution_train, nn_approximation_train)
            )
        )

        print(
            "Mean average percentage error on validation data: {} %".format(
                100 * self.MAPE(analytical_solution_valid, nn_approximation_valid)
            )
        )

        self.plot_function(
            valid_domain,
            self.AbsoluteError(analytical_solution_valid, nn_approximation_valid),
            "Absolute error on validation domain: true sol - Approximation",
            "X",
            "Error",
        )

        self.plot_function(
            train_domain,
            self.AbsoluteError(analytical_solution_train, nn_approximation_train),
            "Absolute error on train domain: true sol - Approximation",
            "X",
            "Error",
        )

        self.plot_function(
            valid_domain,
            analytical_solution_valid,
            "True Solution",
            "domain_value",
            "Function_value",
        )
        self.plot_function(
            valid_domain,
            nn_approximation_valid,
            "Approximation",
            "domain_value",
            "Function_value",
        )

        epochs = torch.arange(self.num_epochs)

        self.plot_function(
            epochs, self.mse_loss_valid, "MSE loss validation", "epoch", "loss"
        )
        self.plot_function(
            epochs, self.mse_loss_train, "MSE loss train", "epoch", "loss"
        )

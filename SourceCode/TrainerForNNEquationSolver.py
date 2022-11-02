import torch
from .NeuralNetworkFunction import NeuralNetworkFunction
from .EquationAndDomain import AbstractEquation
import numpy as np
import random


class TrainerForNNEquationSolver:
    def __init__(
            self, main_eq: AbstractEquation, init_conditions: list, n_epochs: int = 100
    ):
        self.set_seed(77)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_eq = main_eq
        self.init_conditions = init_conditions
        self.batch_size = 1
        self.square_value = lambda x: torch.mean(torch.pow(x, 2))
        n_inputs = 1
        n_hidden_neurons = 100
        n_outputs = 1
        self.nn_model = NeuralNetworkFunction(n_inputs, n_hidden_neurons, n_outputs)
        # self.nn_model.to(self.device)
        self.num_epochs = n_epochs
        lr = 1e-1
        # self.optimizer = torch.optim.Adam(
        #     self.nn_model.parameters(), lr=lr , betas=(0.99, 0.9999)
        # )
        self.optimizer = torch.optim.LBFGS(self.nn_model.parameters(), lr=lr, max_iter=20)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=1
        )

    @staticmethod
    def set_seed(seed: int = 77) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def fit(self, verbose: bool = False) -> (torch.Tensor, torch.Tensor, torch.nn):
        mse_loss_train = torch.zeros(self.num_epochs)
        mse_loss_valid = torch.zeros(self.num_epochs)
        for epoch in range(self.num_epochs):
            if verbose:
                print("Epoch {}/{}:".format(epoch, self.num_epochs - 1), flush=True)
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.nn_model.train()  # Set model to training mode
                else:
                    self.nn_model.eval()  # Set model to evaluate mode
                epoch_loss = self.get_loss(phase)
                if phase == "train":
                    self.scheduler.step()
                    mse_loss_train[epoch] = epoch_loss
                else:
                    mse_loss_valid[epoch] = epoch_loss
                if verbose:
                    print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
        return mse_loss_train, mse_loss_valid, self.nn_model

    def get_loss(self, phase: str) -> torch.tensor:
        zero_val = torch.tensor(0, dtype=torch.float32)
        boundary_coefficient = 1
        # self.optimizer.zero_grad()
        # with torch.set_grad_enabled(True):
        #
        #     if phase == "train":
        #         residuals = self.main_eq.get_residuals_train(self.nn_model)
        #     else:
        #         residuals = self.main_eq.get_residuals_valid(self.nn_model)
        #     epoch_loss = self.loss(residuals)
        #
        #     for init_condition in self.init_conditions:
        #         boundary_residuals = init_condition.get_boundary_residuals(
        #             self.nn_model
        #         )
        #         epoch_loss += boundary_coefficient * self.loss(boundary_residuals)
        #
        #         # backward + optimize only if in training phase
        #     if phase == "train":
        #         epoch_loss.backward(retain_graph=True)

        def closure():
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                if phase == "train":
                    residuals = self.main_eq.get_residuals_train(self.nn_model)
                else:
                    residuals = self.main_eq.get_residuals_valid(self.nn_model)
                loss_val = torch.sum(self.square_value(residuals))


                for init_condition in self.init_conditions:
                    boundary_residuals = init_condition.get_boundary_residuals(
                        self.nn_model
                    )
                    loss_val += torch.sum(boundary_coefficient * self.square_value(boundary_residuals))

                    # backward + optimize only if in training phase
                #loss = self.loss(loss_val, zero_val)
                if phase == "train":
                    loss_val.backward(retain_graph=True)
            return loss_val
        self.optimizer.step(closure=closure)
        epoch_loss = closure()
        return epoch_loss

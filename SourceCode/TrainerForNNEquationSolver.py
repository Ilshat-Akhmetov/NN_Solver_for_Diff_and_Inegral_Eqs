import torch
from .NeuralNetworkFunction import NeuralNetworkFunction
from .EquationClass import AbstractEquation
import numpy as np
import random


class TrainerForNNEquationSolver:
    def __init__(
            self,
            main_eq: AbstractEquation,
            init_conditions: list=None,
            n_epochs: int = 20
    ):
        self.set_seed(77)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_eq = main_eq
        self.init_conditions = init_conditions
        self.batch_size = 1
        # self.norm = lambda x: torch.mean(torch.abs(x))
        self.norm = lambda x: torch.pow(x, 2)
        self.loss = torch.nn.L1Loss()
        # self.loss = torch.nn.MSELoss()
        n_inputs = 1
        n_hidden_neurons = 50
        n_outputs = 1
        n_layers = 2
        nn_model = NeuralNetworkFunction(n_inputs, n_hidden_neurons, n_outputs, n_layers)
        self.nn_model = nn_model
        # self.nn_model.to(self.device)
        self.num_epochs = n_epochs
        lr = 1e-1
        # self.optimizer = torch.optim.Adam(
        #     self.nn_model.parameters(), lr=lr , betas=(0.99, 0.9999)
        # )
        self.optimizer = torch.optim.LBFGS(self.nn_model.parameters(), lr=lr, max_iter=20)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=1
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

    def get_loss(self, phase: str) -> float:
        zero_val = torch.tensor(0.0, dtype=torch.float32)
        boundary_coefficient = 1

        def closure():
            self.optimizer.zero_grad()
            max_residual_loss = torch.tensor(0.0, dtype=torch.float32)

            with torch.set_grad_enabled(True):
                residuals = self.main_eq.get_residuals(self.nn_model, phase)
                loss_val = self.norm(residuals)
                total_loss = torch.sum(loss_val)
                max_residual_loss = torch.max(max_residual_loss, max(loss_val))

                for init_condition in self.init_conditions:
                    boundary_residuals = init_condition.get_boundary_residuals(
                        self.nn_model
                    )
                    boundary_loss = torch.sum(boundary_coefficient * self.norm(boundary_residuals))
                    max_residual_loss = torch.max(max_residual_loss, boundary_loss)
                    total_loss += boundary_loss

                total_loss = self.loss(total_loss, zero_val)
                if phase == "train":
                    total_loss .backward()
            return max_residual_loss.item()

        self.optimizer.step(closure=closure)
        epoch_loss = closure()
        return epoch_loss

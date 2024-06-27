import torch
from .EquationClass import AbstractEquation
import numpy as np
from .SeedGen import SeedGen
from typing import List, Callable
from .DomainClass import AbstractDomain


class TrainerForNNEquationSolver(SeedGen):
    def __init__(
        self,
        main_eq: AbstractEquation,
        nn_models: List[torch.nn.Module],
        n_epochs: int = 20,
        lr: float = 1e-1,
        seed: int = 77,
        optimizer_type: str = "lbfgs",
    ):
        TrainerForNNEquationSolver.set_seed(seed)
        self.main_eq = main_eq
        self.nn_models = nn_models
        model_params = []
        for model in nn_models:
            model_params += list(model.parameters())
        self.n_epochs = n_epochs
        self.seed = seed

        optimizers = {
            "lbfgs": torch.optim.LBFGS(params=model_params, lr=lr, max_iter=20),
            "adam": torch.optim.Adam(params=model_params, lr=lr, betas=(0.99, 0.999)),
        }
        self.optimizer = optimizers[optimizer_type]
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.001,
            threshold_mode="abs",
        )

    def fit(
        self, verbose: bool = False
    ) -> (torch.tensor, torch.tensor, Callable[[torch.tensor], torch.tensor]):
        self.set_seed(self.seed)
        loss_train = torch.zeros(self.n_epochs)
        loss_valid = torch.zeros(self.n_epochs)
        for epoch in range(self.n_epochs):
            if verbose:
                print("Epoch {}/{}:".format(epoch, self.n_epochs - 1), flush=True)
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                epoch_loss = self.get_loss(phase, self.optimizer)
                if phase == "train":
                    self.scheduler.step(epoch_loss)
                    loss_train[epoch] = epoch_loss
                else:
                    loss_valid[epoch] = epoch_loss
                if verbose:
                    print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)

        return loss_train, loss_valid, self.nn_models

    def fit_with_abs_err_history(
        self,
        domain: AbstractDomain,
        analytical_sols: List[Callable],
        verbose: bool = False,
    ) -> (torch.tensor, torch.tensor, Callable[[torch.tensor], torch.tensor]):
        self.set_seed(self.seed)
        res_abs_loss_train = torch.zeros(self.n_epochs)
        res_abs_loss_valid = torch.zeros(self.n_epochs)
        abs_error_train = np.zeros(self.n_epochs)
        abs_error_valid = np.zeros(self.n_epochs)
        for epoch in range(self.n_epochs):
            if verbose:
                print("Epoch {}/{}:".format(epoch, self.n_epochs - 1), flush=True)
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                epoch_loss = self.get_loss(phase, self.optimizer)
                if phase == "train":
                    self.scheduler.step(epoch_loss)
                    res_abs_loss_train[epoch] = epoch_loss
                else:
                    res_abs_loss_valid[epoch] = epoch_loss
                _, appr_val, analytical_val = domain.get_domain_and_target(
                    phase,
                    domain.offset,
                    nn_models=self.nn_models,
                    analytical_solutions=analytical_sols,
                )
                if phase == "train":
                    abs_error_train[epoch] = np.max(np.abs(appr_val - analytical_val))
                else:
                    abs_error_valid[epoch] = np.max(np.abs(appr_val - analytical_val))
                if verbose:
                    print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
        return (
            res_abs_loss_train,
            res_abs_loss_valid,
            abs_error_train,
            abs_error_valid,
            self.nn_models,
        )

    def get_loss(self, phase: str, optimizer) -> float:
        def closure():
            optimizer.zero_grad()
            total_loss, max_residual_norm = self.main_eq.get_residuals_norm(
                self.nn_models, phase
            )
            if phase == "train":
                total_loss.backward()

            return max_residual_norm.item()

        max_residual_norm = closure()
        optimizer.step(closure=closure)
        return max_residual_norm

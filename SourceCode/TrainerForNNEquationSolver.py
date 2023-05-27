import torch
from .EquationClass import AbstractEquation
import numpy as np
import random
from typing import List, Callable


class TrainerForNNEquationSolver:
    def __init__(
            self,
            main_eq: AbstractEquation,
            n_epochs: int = 20,
            lr: float = 1e-1,
            n_hidden_neurons: int = 20,
            act_func: Callable = torch.tanh,
            boundary_satisfying_models: List[Callable] = None,
            n_hidden_layers: int=1,
            seed: int = 77
    ):
        self.set_seed(seed)
        self.n_hidden_layers = n_hidden_layers
        self.act_func = act_func
        self.main_eq = main_eq
        self.batch_size = 1
        self.norm = lambda x: torch.pow(x, 2)
        self.n_hidden_neurons = n_hidden_neurons

        self.num_epochs = n_epochs
        self.nn_type = main_eq.get_nn_model_type()
        if boundary_satisfying_models is None:
            self.boundary_satisfying_models = [None] * self.main_eq.count_equations()
        elif not isinstance(boundary_satisfying_models, list):
            self.boundary_satisfying_models = [boundary_satisfying_models]
        else:
            self.boundary_satisfying_models = boundary_satisfying_models
        self.nn_models, model_params = self.get_nn_models(
            self.boundary_satisfying_models
        )
        self.lbfgs_optimizer = torch.optim.LBFGS(params=model_params, lr=lr, max_iter=20)
        #self.adam_opt = torch.optim.Adam(params=model_params, lr=1e-3)#, betas=(0.99, 0.9999))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.lbfgs_optimizer, mode='min',
                                                               factor=0.1, patience=5, threshold=0.001,
                                                               threshold_mode='abs')

    def get_nn_models(
            self, boundary_satisfying_models
    ) -> (List[Callable[[torch.tensor], torch.tensor]], List[torch.tensor],):
        n = self.main_eq.count_equations()
        model_params = []
        nn_models = []
        for i in range(n):
            nn_model = self.nn_type(
                boundary_satisfying_models[i],
                self.n_hidden_neurons,
                self.n_hidden_layers,
                act=self.act_func,
            )
            model_params += list(nn_model.parameters())
            nn_models.append(nn_model)
        return nn_models, model_params

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def fit(
            self, verbose: bool = False
    ) -> (torch.tensor, torch.tensor, Callable[[torch.tensor], torch.tensor]):
        loss_train = torch.zeros(self.num_epochs)
        loss_valid = torch.zeros(self.num_epochs)
        for epoch in range(self.num_epochs):
            if verbose:
                print("Epoch {}/{}:".format(epoch, self.num_epochs - 1), flush=True)
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    (
                        nn_model.train() for nn_model in self.nn_models
                    )  # Set model to training mode
                else:
                    (
                        nn_model.eval() for nn_model in self.nn_models
                    )  # Set model to evaluate mode
                epoch_loss = self.get_loss(phase, epoch)
                if phase == "train":
                    self.scheduler.step(epoch_loss)
                    loss_train[epoch] = epoch_loss
                else:
                    loss_valid[epoch] = epoch_loss
                if verbose:
                    print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
        return loss_train, loss_valid, self.nn_models

    def get_loss(self, phase: str, epoch: int) -> float:
        n_epochs_for_adam =  0
        def closure():
            if epoch < n_epochs_for_adam:
                self.adam_opt.zero_grad()
            else:
                self.lbfgs_optimizer.zero_grad()
            total_loss, max_residual_norm = self.main_eq.get_residuals_norm(
                self.nn_models, phase
            )
            if phase == "train":
                total_loss.backward()

            return max_residual_norm.item()

        epoch_loss = closure()
        if epoch < n_epochs_for_adam:
            self.adam_opt.step(closure=closure)
        else:
            self.lbfgs_optimizer.step(closure=closure)
        return epoch_loss

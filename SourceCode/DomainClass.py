import abc
import torch
from numpy import array as np_array
import matplotlib.pyplot as plt
from typing import List, Union
from .NeuralNetworkFunction import NeuralNetworkModel1d, NeuralNetworkModel2d


class AbstractDomain(abc.ABC):
    @abc.abstractmethod
    def get_train_domain(self) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_domain(self) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain_size(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_nn_type(self) -> object:
        raise NotImplementedError


class OneDimensionalSimpleDomain(AbstractDomain):
    def __init__(self, left_bound: Union[float, int],
                 right_bound: Union[float, int], n_points: int):
        self.n_points = n_points
        self.left_point = left_bound
        self.right_point = right_bound
        self.dx = (right_bound - left_bound) / (n_points - 1)
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()
        self.__nn_type = NeuralNetworkModel1d

    def get_nn_type(self) -> object:
        return self.__nn_type

    def get_domain_size(self) -> int:
        return self.n_points

    def get_domain_unit(self) -> float:
        return self.dx

    def make_train_domain(self) -> torch.tensor:
        train_domain = torch.linspace(
            self.left_point, self.right_point, self.n_points
        )
        train_domain.requires_grad = True
        return [train_domain]

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = self.train_domain[0] + self.get_domain_unit() / 4
        valid_domain[-1] -= self.get_domain_unit() / 2
        return [valid_domain]

    def get_train_domain(self) -> List[torch.tensor]:
        return self.train_domain

    def get_valid_domain(self) -> List[torch.tensor]:
        return self.valid_domain

    @staticmethod
    def plot_error_distribution(domain: List[np_array],
                                func_value: List[np_array]):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title("Abs error |an_sol(x) - approx(x)|")
        ax.set_xlabel("X")
        ax.set_ylabel("Error")
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.plot(domain[0], func_value)
        plt.show()


class TwoDimensionalSimpleDomain(AbstractDomain):
    def __init__(self,
                 x1_n_points: int,
                 x1_left: Union[int, float],
                 x1_right: Union[int, float],
                 x2_n_points: int,
                 x2_left: Union[int, float],
                 x2_right: Union[int, float], ):
        self.x1_n_points = x1_n_points
        self.x1_left = x1_left
        self.x1_right = x1_right
        self.x2_n_points = x2_n_points
        self.x2_left = x2_left
        self.x2_right = x2_right
        self.dx1 = (x1_right - x1_left) / (x1_n_points - 1)
        self.dx2 = (x2_right - x2_left) / (x2_n_points - 1)
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()
        self.__nn_type = NeuralNetworkModel2d

    def get_nn_type(self) -> NeuralNetworkModel2d:
        return self.__nn_type

    def get_domain_size(self) -> int:
        return self.x1_n_points * self.x2_n_points

    def get_domain_unit(self) -> float:
        return self.dx1 * self.dx2

    def make_train_domain(self) -> torch.tensor:
        x1_train_domain = torch.linspace(self.x1_left, self.x1_right, self.x1_n_points)
        x2_train_domain = torch.linspace(self.x2_left, self.x2_right, self.x2_n_points)

        x1_mesh, x2_mesh = torch.meshgrid((x1_train_domain, x2_train_domain))
        x1_mesh.requires_grad = True
        x2_mesh.requires_grad = True
        return [x1_mesh, x2_mesh]

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = self.train_domain
        valid_domain[0] = (valid_domain[0] + valid_domain[0] + self.dx1) / 2
        valid_domain[1] = (valid_domain[1] + valid_domain[1] + self.dx2) / 2
        return valid_domain

    def get_train_domain(self) -> List[torch.tensor]:
        return self.train_domain

    def get_valid_domain(self) -> List[torch.tensor]:
        return self.valid_domain

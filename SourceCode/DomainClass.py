import abc
import torch
from numpy import array as np_array
import matplotlib.pyplot as plt

from typing import List, Union, Callable
from .NeuralNetworkFunction import (
    NeuralNetworkFunctionWrapper1D,
    NeuralNetworkFunctionWrapper2D,
)


class AbstractDomain(abc.ABC):
    @staticmethod
    def plot_error_distribution(domain: List[np_array], func_value: List[np_array]):
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain_size(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def make_train_domain(self, offset: float = 0) -> torch.tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def make_valid_domain(self) -> torch.tensor:
        raise NotImplementedError

    def get_domain(self, phase: str = "train"):
        assert phase in ["train", "valid"]
        if phase == "train":
            return self.train_domain
        else:
            return self.valid_domain

    def get_domain_copy(self, phase: str = "train", offset: float = 0.0):
        assert phase in ["train", "valid"]
        if phase == "train":
            return self.make_train_domain(offset)
        else:
            return self.make_valid_domain()


class OneDimensionalSimpleDomain(AbstractDomain):
    def __init__(
        self,
        left_bound: Union[float, int],
        right_bound: Union[float, int],
        n_points: int,
        offset: float = 1e-2,  # we need this so main domain does not include boundaries
    ):
        self.n_points = n_points
        self.left_point = left_bound
        self.right_point = right_bound
        self.offset = offset
        self.dx = (right_bound - left_bound) / (n_points - 1)
        self.train_domain = self.make_train_domain(offset)
        self.valid_domain = self.make_valid_domain()
        self.__nn_type = NeuralNetworkFunctionWrapper1D

    def get_domain_size(self) -> int:
        return self.n_points

    def get_domain_unit(self) -> float:
        return self.dx

    def make_train_domain(self, offset: float = 0) -> torch.tensor:
        train_domain = torch.linspace(
            self.left_point + offset, self.right_point - offset, self.n_points
        )
        train_domain.requires_grad = True
        return [train_domain]

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = self.get_domain_copy()
        valid_domain[0] = (valid_domain[0][:-1] + valid_domain[0][1:]) / 2
        return valid_domain

    def get_nn_type(self) -> Callable[[torch.tensor, torch.tensor], torch.tensor]:
        return self.__nn_type

    @staticmethod
    def plot_error_distribution(
        domain: List[np_array],
        func_value: List[np_array],
        title: str = "Abs error |an_sol(x) - approx(x)|",
        figsize: tuple = (10, 12),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Error")
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.plot(domain[0], func_value)
        plt.show()


class TwoDimensionalSimpleDomain(AbstractDomain):
    def __init__(
        self,
        x1_n_points: int,
        x1_left: Union[int, float],
        x1_right: Union[int, float],
        x2_n_points: int,
        x2_left: Union[int, float],
        x2_right: Union[int, float],
        offset: float = 1e-2,  # we need this so main domain does not include boundaries
    ):
        self.x1_n_points = x1_n_points
        self.x1_left = x1_left
        self.x1_right = x1_right
        self.x2_n_points = x2_n_points
        self.x2_left = x2_left
        self.x2_right = x2_right
        self.offset = offset
        self.dx1 = (x1_right - x1_left) / (x1_n_points - 1)
        self.dx2 = (x2_right - x2_left) / (x2_n_points - 1)
        self.train_domain = self.make_train_domain(offset)
        self.valid_domain = self.make_valid_domain()
        self.__nn_type = NeuralNetworkFunctionWrapper2D

    def get_domain_size(self) -> int:
        return self.x1_n_points * self.x2_n_points

    def get_domain_unit(self) -> float:
        return self.dx1 * self.dx2

    def make_train_domain(self, offset: float = 0) -> torch.tensor:
        x1_train_domain = torch.linspace(
            self.x1_left + offset, self.x1_right - offset, self.x1_n_points
        )
        x2_train_domain = torch.linspace(
            self.x2_left + offset, self.x2_right - offset, self.x2_n_points
        )

        x1_mesh, x2_mesh = torch.meshgrid((x1_train_domain, x2_train_domain), indexing='ij')
        x1_mesh.requires_grad = True
        x2_mesh.requires_grad = True
        return [x1_mesh, x2_mesh]

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = self.get_domain_copy()
        valid_domain[0] = (valid_domain[0][:-1] + valid_domain[0][1:]) / 2
        valid_domain[1] = (valid_domain[1][:-1] + valid_domain[1][1:]) / 2
        return valid_domain

    def get_nn_type(self) -> Callable[[torch.tensor, torch.tensor], torch.tensor]:
        return self.__nn_type

    @staticmethod
    def plot_error_distribution(
        domain: List[np_array],
        func_value: List[np_array],
        title: str = "Abs error |an_sol(x,y) - approx(x,y)|",
        figsize: tuple = (10, 12),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("Error")
        ax.grid(True, which="both")
        ax.plot_surface(domain[0], domain[1], func_value)
        plt.show()

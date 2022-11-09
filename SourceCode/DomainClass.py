import abc
import torch


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


class OneDimensionalSimpleDomain(AbstractDomain):
    def __init__(self, left_bound: float, right_bound: float, n_points: int):
        self.n_points = n_points
        self.left_point = left_bound
        self.right_point = right_bound
        self.dx = (right_bound - left_bound) / (n_points - 1)
        self.train_domain = self.make_train_domain()
        self.valid_domain = self.make_valid_domain()

    def get_domain_size(self) -> int:
        return self.n_points

    def get_domain_unit(self):
        return self.dx

    def make_train_domain(self) -> torch.tensor:
        train_domain = torch.linspace(
            self.left_point, self.right_point, self.n_points
        )
        train_domain.requires_grad = True
        return train_domain

    def make_valid_domain(self) -> torch.tensor:
        valid_domain = self.train_domain + self.dx / 4
        valid_domain[-1] -= self.dx / 2
        return valid_domain

    def get_train_domain(self) -> torch.tensor:
        return self.train_domain.unsqueeze(dim=1)

    def get_valid_domain(self) -> torch.tensor:
        return self.valid_domain.unsqueeze(dim=1)

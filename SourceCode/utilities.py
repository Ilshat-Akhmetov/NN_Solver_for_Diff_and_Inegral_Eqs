import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Callable


def plot_two_curves(
    x_value: np.array,
    f1_value: np.array,
    f2_value: np.array,
    title: str,
    x_label: str,
    f1_label: str,
    f2_label: str,
) -> None:
    fig = plt.figure('Parametric curve', figsize=(7, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")
    ax.grid(True, which="both")
    ax.plot(np.ravel(x_value),
            np.ravel(f1_value[0]),
            np.ravel(f1_value[1]),
            color="lime", label=f1_label, linewidth=7.0)
    ax.plot(np.ravel(x_value),
            np.ravel(f2_value[0]),
            np.ravel(f2_value[1]),
            color="mediumblue", label=f2_label, linewidth=3.0)
    ax.legend(loc="best")
    plt.show()


def plot_two_1d_functions(
    x_value: torch.Tensor,
    f1_value: torch.Tensor,
    f2_value: torch.Tensor,
    title: str,
    x_label: str,
    f1_label: str,
    f2_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot(x_value, f1_value[0], color="lime", label=f1_label, linewidth=5.0)
    ax.plot(x_value, f2_value[0], color="mediumblue", label=f2_label, linewidth=3.0)
    ax.legend(loc="best")
    plt.show()


def nth_derivative(
    nn_model: Callable, variable: torch.tensor, derivatives_degree: int
) -> torch.tensor:
    derivative_value = nn_model(variable)
    for i in range(derivatives_degree):
        derivative_value = torch.autograd.grad(
            derivative_value,
            variable,
            grad_outputs=torch.ones_like(variable),
            create_graph=True,
            retain_graph=True,
        )[0]
    return derivative_value

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Union


def torch_to_numpy(arr: torch.tensor) -> np.array:
    return arr.cpu().detach().numpy()


def gen_2d_points_line(
    non_const_var_left: Union[int, float],
    non_const_var_right: Union[int, float],
    non_const_var_size: int,
    const_var_value: Union[int, float],
    const_var_ind: int = 1,
) -> tuple:
    assert const_var_ind in (1, 2)
    const_var = torch.Tensor([const_var_value])
    const_var.requires_grad = True
    non_const_domain = torch.linspace(
        non_const_var_left, non_const_var_right, non_const_var_size
    )
    non_const_domain.requires_grad = True
    if const_var_ind == 1:
        xm, ym = torch.meshgrid((const_var, non_const_domain))
    else:
        xm, ym = torch.meshgrid((non_const_domain, const_var))
    return xm, ym


def plot_1d_function(
    x_value: np.array,
    y_value: np.array,
    title: str,
    x_label: str,
    y_label: str,
    figsize: tuple = (10, 12),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot(x_value, y_value)
    plt.show()


def plot_two_curves(
    x_value: List[np.array],
    f1_value: np.array,
    f2_value: np.array,
    title: str,
    f1_label: str,
    f2_label: str,
    figsize: tuple = (10, 12),
) -> None:
    fig = plt.figure("Parametric curve", figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("Variable")
    ax.set_ylabel("f1")
    ax.set_zlabel("f2")
    ax.grid(True, which="both")
    ax.plot(
        np.ravel(x_value),
        np.ravel(f1_value[0]),
        np.ravel(f1_value[1]),
        color="lime",
        label=f1_label,
        linewidth=7.0,
    )
    ax.plot(
        np.ravel(x_value),
        np.ravel(f2_value[0]),
        np.ravel(f2_value[1]),
        color="mediumblue",
        label=f2_label,
        linewidth=3.0,
    )
    ax.legend(loc="best")
    plt.show()


def plot_two_1d_functions(
    x_value: List[np.array],
    f1_value: np.array,
    f2_value: np.array,
    title: str,
    f1_label: str,
    f2_label: str,
    figsize: tuple = (10, 12),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Function")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot(x_value[0], f1_value[0], color="lime", label=f1_label, linewidth=2.0)
    ax.scatter(
        x_value[0], f2_value[0], color="mediumblue", label=f2_label, linewidth=4.0
    )
    ax.legend(loc="best")
    plt.show()


def plot_two_2d_functions(
    domain: List[np.array],
    f1_value: np.array,
    f2_value: np.array,
    title: str,
    f1_label: str,
    f2_label: str,
    figsize: tuple = (10, 12),
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("Var1")
    ax.set_ylabel("Var2")
    ax.set_zlabel("Function")
    ax.grid(True, which="both")
    ax.plot_surface(
        domain[0], domain[1], f1_value[0], color="lime", label=f1_label, linewidth=5
    )
    ax.scatter(
        domain[0],
        domain[1],
        f2_value[0],
        color="mediumblue",
        label=f2_label,
        linewidth=2,
    )
    # ax.legend(loc="best")
    plt.show()


def nth_derivative(
    nn_model_value: torch.tensor, variable: torch.tensor, derivatives_degree: int
) -> torch.tensor:
    derivative_value = nn_model_value
    for i in range(derivatives_degree):
        derivative_value = torch.autograd.grad(
            derivative_value,
            variable,
            grad_outputs=torch.ones_like(variable),
            create_graph=True,
            retain_graph=True,
        )[0]
    return derivative_value

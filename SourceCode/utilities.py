import torch.autograd
import matplotlib.pyplot as plt
from torch import ones_like

def plot_1d_function(
    x_value: torch.Tensor, y_value: torch.Tensor, title: str, x_label: str, y_label: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot(x_value, y_value)
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
    ax.plot(
        x_value,
        f1_value,
        color="lime",
        label=f1_label,
        linewidth=7.0
    )
    ax.plot(
        x_value,
        f2_value,
        color="mediumblue",
        label=f2_label,
        linewidth=3.0
    )
    ax.legend(loc="best")
    plt.show()

def nth_derivative(
    nn_model: torch.Tensor, variable: torch.Tensor, derivatives_degree: int
) -> torch.Tensor:
    derivative_value = nn_model(variable)
    for i in range(derivatives_degree):
        derivative_value = torch.autograd.grad(
            derivative_value,
            variable,
            grad_outputs = ones_like(variable),
            create_graph=True,
            retain_graph=True,
        )[0]
    return derivative_value

import torch.autograd
import matplotlib.pyplot as plt
from torch import ones_like

def plot_two_curves(
    x_value: torch.Tensor,
    f1_value: torch.Tensor,
    f2_value: torch.Tensor,
    title: str,
    x_label: str,
    f1_label: str,
    f2_label: str,
) -> None:
    #fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot3D(
        *f1_value,
        x_value,
        color="lime",
        label=f1_label,
        linewidth=7.0
    )
    ax.plot3D(
        *f2_value,
        x_value,
        color="mediumblue",
        label=f2_label,
        linewidth=3.0
    )
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
    ax.plot(
        x_value,
        f1_value[0],
        color="lime",
        label=f1_label,
        linewidth=7.0
    )
    ax.plot(
        x_value,
        f2_value[0],
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

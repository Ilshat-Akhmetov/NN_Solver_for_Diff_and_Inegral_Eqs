import torch.autograd
from torch import ones_like
import matplotlib.pyplot as plt


def nth_derivative(
    function_value: torch.Tensor, variable: torch.Tensor, derivatives_degree: int
) -> torch.Tensor:
    derivative_value = function_value
    for i in range(derivatives_degree):
        derivative_value = torch.autograd.grad(
            derivative_value,
            variable,
            grad_outputs=ones_like(variable),
            create_graph=True,
            retain_graph=True,
        )[0]
    return derivative_value


def plot_one_dimensional_function(
    x_value: torch.Tensor, y_value: torch.Tensor, title: str, x_label: str, y_label: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.plot(x_value.cpu().detach().numpy(), y_value.cpu().detach().numpy())
    plt.show()
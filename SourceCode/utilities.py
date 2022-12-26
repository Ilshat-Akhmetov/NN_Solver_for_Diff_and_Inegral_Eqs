import torch.autograd
from torch import ones_like
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import abs
from torch import flatten

def print_comparison_table(domain, nn_model, analytical_sol):
    appr_val = nn_model(domain)
    analyt_val = analytical_sol(domain)

    domain = flatten(domain)
    appr_val = flatten(appr_val)
    analyt_val = flatten(analyt_val)

    appr_val = appr_val.cpu().detach().numpy()
    analyt_val = analyt_val.cpu().detach().numpy()
    domain = domain.cpu().detach().numpy()
    error = abs(appr_val - analyt_val)
    data = {"Input": domain, "Analytical": analyt_val, "ANN": appr_val, "Error": error}
    df = DataFrame(data=data)
    print(df)




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
    ax.plot(x_value.cpu().detach().numpy(), y_value.cpu().detach().numpy())
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
        x_value.cpu().detach().numpy(),
        f1_value.cpu().detach().numpy(),
        color="lime",
        label=f1_label,
        linewidth=7.0
    )
    ax.plot(
        x_value.cpu().detach().numpy(),
        f2_value.cpu().detach().numpy(),
        color="mediumblue",
        label=f2_label,
        linewidth=3.0
    )
    ax.legend(loc="best")
    plt.show()

import torch.autograd
from torch import ones_like


def nth_derivative(function_value: torch.tensor, variable: torch.tensor, derivatives_degree: int) -> torch.tensor:
    derivative_value = function_value
    for i in range(derivatives_degree):
        derivative_value = torch.autograd.grad(
                                derivative_value,
                                variable,
                                grad_outputs=ones_like(variable),
                                create_graph=True,
                                retain_graph=True)[0]
    return derivative_value

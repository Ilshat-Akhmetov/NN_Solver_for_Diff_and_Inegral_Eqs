## An application for numerical solving ODEs and integral equations with artificial neural networks.

The main idea is to minimize the sum of residual squares for each boundary or initial condition
and for the main equation on its domain. 

For example, if we have the equation 

$$ u' +2xu = 5, x \in [0,1] $$

with initial condition

$$ u'(0) = 3 $$

then the proposed algorithm will try to find a neural network minimizing the following expression:

$$ argmin_{NN}((NN(x)' +2xNN(x) - 5)^2 + \alpha (NN(0)' - 3)^2) $$

Here coefficient $\alpha$ defines how much attention 
should be paid to boundary condition during training. 
This way we approximate unknown function **u(x)** with a neural network (**NN(x)**). 
For integral equations idea actually is the same. 
As you might guess, this method tries to minimize the sum of squares of residuals for the domain
of the main equation and boundary conditions simultaneously at each epoch.
It is assumed that if residual square gets close to zero, then
NN-approximation becomes quite accurate at representing unknown function **u**.

In **SourceCode** you can take a look at code and find out how classes and methods are implemented.

In **JupyterPresentations\Solving_ODE_with_NN.ipynb** you can find various examples of NN's
application for solving ordinary differential equations. Sometimes it does not converge. 
I cannot say now why it not always work, this question requires further investigation.

In **JupyterPresentations\Solving_Integral_Equation_with_NN.ipynb** you can find various examples of NN's
application for solving integral equations.

In **JupyterPresentations\Solving_System_of_ODE_with_NN.ipynb** you can also find examples of NN's
application for solving system of ODEs.

In **JupyterPresentations\Solving_2D_PDE_with_NN.ipynb** you can see 
how to apply this program for solving 2D equations

This version is rather a prototype than a finished project. In the future, maybe I will expand this program for three 
and 4-dimensional cases

How to use this?

**Example 1**. Fredholm's equation.

$$y(x) = sin(\pi*x) + \frac{1}{2}\int_{0}^1y(t)dt $$
with analytical solution:
analytical solution: $$y(x) = sin(\pi x) + \frac{2}{\pi} $$
$$x \in [0, 1] $$

Here is an example of how this equation can be solved with this repo

```python
import torch
from SourceCode import *
from math import pi

left_border = 0
right_border = 1
n_points = 20
integration_func = lambda curr_v, int_domain, nn_model: nn_model(int_domain)
main_eq_res = (lambda curr_v, nn_model: nn_model(curr_v) -
                                        torch.sin(pi * curr_v) -
                                        0.5 * IntegralEquations.calculate_fredholm_equation_1d(
 integration_func,
 nn_model,
 curr_v,
 left_border,
 right_border,
 n_points))
main_domain = OneDimensionalSimpleDomain(0, 1, n_points)

main_eq = MainEquationClass(main_domain, main_eq_res)
n_epochs = 10
models = NNGenerator.generate_models(
            nn_count=1, nn_type='ResMLP'
        )

nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs, nn_models=models
)
loss_train, loss_valid, nn_model = nn_ode_solver.fit()

# if you want to show max residual value on domain or compare with the exact solution
analytical_solution = lambda x_var: torch.sin(pi * x_var) + 2 / pi
report = ReportMaker(nn_model,
                         loss_train,
                         loss_valid,
                         main_domain,
                         compare_to_functions=plot_two_1d_functions,
                         analytical_solutions=analytical_solution
                         )
report.compare_appr_with_analytical()
```


**Example 2**. Simple system of 2 ODEs. Here we also specify an approximation 
so that it satisfies the boundary conditions, and we don't have to specify them exmplicitly.
$$x'= -y $$
$$y' = x + cos(t) $$
$$x(0) = 0$$ 
$$y(0) = 0$$
$$t \in [0,5] $$
Analytucal solution: $$x = -\frac{1}{2}tsin(t)$$ $$y = tcos(t) + \frac{1}{2}sin(t)$$
Approximation satisfying boundary conditions: $ apprX(t) = t*nn_1(t) $$
$$apprY(t) = t*nn_2(t) $$

```python
import torch
from SourceCode import *


n_epochs = 10
n_points = 10
dh = 1e-3


left_bound = 0
right_bound = 5
main_eq_residual1 = lambda t, x, y: nth_derivative(x(t), t, 1) + y(t)
main_eq_residual2 = lambda t, x, y: nth_derivative(y(t), t, 1) - x(t) - torch.cos(t)
main_eq_residuals = [main_eq_residual1, main_eq_residual2]
n_points = 20

dh = 0.001
main_domain = OneDimensionalSimpleDomain(left_bound + dh, right_bound, n_points)

main_eq = MainEquationClass(main_domain, main_eq_residuals)
boundary_satisfying_models = [
        lambda x, model1: x * model1(x),
        lambda x, model2: x * model2(x)
    ]

n_epochs = 20
models = NNGenerator.generate_models(
            nn_count=2, boundary_satisfying_models=boundary_satisfying_models
        )

nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs, nn_models=models
)
loss_train, loss_valid, nn_models = nn_ode_solver.fit()

analytical_solution1 = lambda x: -1 / 2 * x * torch.sin(x)
analytical_solution2 = lambda x: 1 / 2 * (x * torch.cos(x) + torch.sin(x))
analytical_solutions = [analytical_solution1, analytical_solution2]
report = ReportMaker(nn_models,
                         loss_train,
                         loss_valid,
                         main_domain,
                         compare_to_functions=plot_two_curves,
                         analytical_solutions=analytical_solutions
                         )
report.compare_appr_with_analytical()
```

Currently, there are 4 types of NN architectures available: MLP, Residual-MLP, BasisFuncMLP and KAN. 
BasisFuncMLP tries to approximate unknown function by a linear combination of some basis functions. 
In fact, at each domain point it predicts coefficients for these basis functions.
In NNGenerator you can choose one of them as nn_type parameter (default mlp).
Residual-MLP is essentially the same MLP but with residual connections between its blocks. 
KAN implementation was taken from this repo https://github.com/Blealtan/efficient-kan.



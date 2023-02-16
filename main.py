import torch
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import OneDimensionalMainEquation
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.IntegralEquations import IntegralEquations
from SourceCode.InitConditionClass import OnePointInitialCondition
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker
from math import pi

if __name__ == "__main__":
    left_bound = 0
    right_bound = 1
    main_eq_residual = (
        lambda x, nn_model: nth_derivative(nn_model, x, 2)
        + 0.2 * nth_derivative(nn_model, x, 1)
        + nn_model(x)
        + 0.2 * torch.exp(-x / 5) * torch.cos(x)
    )
    n_points = 10
    dh = 1e-3
    main_domain = OneDimensionalSimpleDomain(left_bound + dh, right_bound - dh, n_points)


    first_init_cond_res = lambda x, nn_model: nn_model(x) - 0
    first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

    second_init_cond_res = lambda x, nn_model: nn_model(x) - torch.sin(
        torch.Tensor([1])
    ) * torch.exp(torch.Tensor([-0.2]))
    second_init_cond = OnePointInitialCondition(right_bound, second_init_cond_res)

    boundary_conditions = [first_init_cond, second_init_cond]

    main_eq = OneDimensionalMainEquation(main_domain, main_eq_residual, boundary_conditions)

    true_solution = lambda x: torch.exp(-x / 5) * torch.sin(x)
    n_epochs = 10
    nn_ode_solver = TrainerForNNEquationSolver(main_eq, n_epochs)
    loss_train, loss_valid, nn_model = nn_ode_solver.fit()
    report = ReportMaker(
        true_solution, nn_model, main_eq, loss_train, loss_valid, main_domain, n_epochs
    )
    report.make_report()
    report.print_comparison_table()


    # true_solution = lambda x_var: torch.sin(x_var) + pi * x_var / 4
    # left_border = 0
    # right_border = pi / 2
    # n_points = 20
    # integration_func = lambda curr_v, int_domain, nn_model: curr_v * torch.sin(int_domain) * nn_model(int_domain)
    # main_eq_res = (lambda curr_v, nn_model: nn_model(curr_v) - 0.5 * IntegralEquations.FredholmEquation1D(
    #     integration_func,
    #     nn_model,
    #     curr_v,
    #     left_border,
    #     right_border,
    #     n_points) - torch.sin(curr_v))
    # main_domain = OneDimensionalSimpleDomain(0, 1, n_points)
    #
    # main_eq = OneDimensionalMainEquation(main_domain, main_eq_res)
    # n_epochs = 10
    # nn_ode_solver = TrainerForNNEquationSolver(main_eq, n_epochs)
    # loss_train, loss_valid, nn_model = nn_ode_solver.fit()
    # report = ReportMaker(
    #     true_solution, nn_model, main_eq, loss_train, loss_valid, main_domain, n_epochs
    # )
    # report.make_report()

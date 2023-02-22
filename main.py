import torch
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import OneDimensionalMainEquation
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.IntegralEquations import IntegralEquations
from SourceCode.InitConditionClass import OnePointInitialCondition
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker
from SourceCode.utilities import plot_two_curves
from math import pi

if __name__ == "__main__":
    left_bound = 0
    right_bound = 1
    main_eq_residual1 = lambda t, x, y: nth_derivative(x, t, 1) + y(t)
    main_eq_residual2 = lambda t, x, y: nth_derivative(y, t, 1) - x(t) - torch.cos(t)
    main_eq_residuals = [main_eq_residual1, main_eq_residual2]
    n_points = 20
    true_sol1 = lambda x: -1 / 2 * x * torch.sin(x)
    true_sol2 = lambda x: 1 / 2 * (x * torch.cos(x) + torch.sin(x))
    true_solutions = [true_sol1, true_sol2]
    dh = 0.001
    main_domain = OneDimensionalSimpleDomain(left_bound + dh, right_bound, n_points)

    first_init_cond_res = lambda x, nn_model1, nn_model2: nn_model1(x) - 0
    first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

    second_init_cond_res = lambda x, nn_model1, nn_model2: nn_model2(x) - 0
    second_init_cond = OnePointInitialCondition(left_bound, second_init_cond_res)

    boundary_conditions = [first_init_cond, second_init_cond]

    main_eq = OneDimensionalMainEquation(main_domain, main_eq_residuals, boundary_conditions)

    n_epochs = 20
    nn_ode_solver = TrainerForNNEquationSolver(main_eq)
    loss_train, loss_valid, nn_models = nn_ode_solver.fit()
    report = ReportMaker(true_solutions, nn_models, main_eq, loss_train, loss_valid,
                         main_domain,
                         num_epochs=n_epochs,
                         plot2functions=plot_two_curves,
                         do_plot_func=True)
    report.make_report()
    report.print_comparison_table()

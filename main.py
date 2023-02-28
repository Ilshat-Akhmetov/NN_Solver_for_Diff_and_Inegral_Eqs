import torch
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import MainEquationClass
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker
from SourceCode.utilities import plot_two_curves

if __name__ == "__main__":
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
        lambda x, model2: x * model2(x),
    ]

    n_epochs = 20
    nn_ode_solver = TrainerForNNEquationSolver(main_eq,
                                               n_epochs=n_epochs,
                                               act_func=torch.tanh,
                                               boundary_satisfying_models=boundary_satisfying_models)
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
    report.print_loss_history()
    report.compare_appr_with_analytical()
    report.print_comparison_table()

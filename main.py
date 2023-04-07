import torch
from SourceCode.utilities import nth_derivative
from SourceCode.EquationClass import MainEquationClass
from SourceCode.DomainClass import (
    OneDimensionalSimpleDomain,
    TwoDimensionalSimpleDomain,
)
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker
from SourceCode.utilities import plot_two_curves, plot_two_2d_functions
from SourceCode.InitConditionClass import TwoDimInitialCondition
from math import pi, e

if __name__ == "__main__":
    # left_bound = 0
    # right_bound = 5
    # main_eq_residual1 = lambda t, x, y: nth_derivative(x(t), t, 1) + y(t)
    # main_eq_residual2 = lambda t, x, y: nth_derivative(y(t), t, 1) - x(t) - torch.cos(t)
    # main_eq_residuals = [main_eq_residual1, main_eq_residual2]
    # n_points = 20
    # main_domain = OneDimensionalSimpleDomain(left_bound, right_bound, n_points)
    # # approximation satisfies boundary conditions
    # main_eq = MainEquationClass(main_domain, main_eq_residuals)
    # boundary_satisfying_models = [
    #     lambda x, model1: x * model1(x),
    #     lambda x, model2: x * model2(x),
    # ]
    #
    # n_epochs = 30
    # nn_ode_solver = TrainerForNNEquationSolver(main_eq,
    #                                            n_epochs=n_epochs,
    #                                            act_func=torch.tanh,
    #                                            boundary_satisfying_models=boundary_satisfying_models)
    # loss_train, loss_valid, nn_models = nn_ode_solver.fit()
    #
    # analytical_solution1 = lambda x: -1 / 2 * x * torch.sin(x)
    # analytical_solution2 = lambda x: 1 / 2 * (x * torch.cos(x) + torch.sin(x))
    # analytical_solutions = [analytical_solution1, analytical_solution2]
    # report = ReportMaker(nn_models,
    #                      loss_train,
    #                      loss_valid,
    #                      main_domain,
    #                      analytical_solutions=analytical_solutions,
    #                      compare_to_functions=plot_two_curves,
    #                      main_eq_residuals=main_eq_residuals
    #                      )
    # report.plot_abs_residual_distr()
    # report.print_loss_history()
    # report.compare_appr_with_analytical()
    # report.print_comparison_table()

    x1_points = 20
    x1_left = -1
    x1_right = 1
    x2_points = 20
    x2_left = -1
    x2_right = 1


    def true_sol(x, y):
        total_sum = 0
        lim = 100
        for i in range(1, lim, 2):
            for j in range(1, lim, 2):
                total_sum += pow(-1, (i + j) // 2 - 1) / (i * j * (i * i + j * j)) * torch.cos(
                    i * pi / 2 * x) * torch.cos(j * pi / 2 * y)
        total_sum *= 8*8/(pi*pi*pi*pi)
        return total_sum


    true_solution = true_sol

    main_domain = TwoDimensionalSimpleDomain(x1_points,
                                             x1_left,
                                             x1_right,
                                             x2_points,
                                             x2_left,
                                             x2_right)
    main_eq_residuals = lambda x, y, nn_model: (nth_derivative(nn_model(x, y), x, 2) +
                                                nth_derivative(nn_model(x, y), y, 2) +
                                                torch.tensor(1.0, requires_grad=True))

    main_eq = MainEquationClass(main_domain, main_eq_residuals)
    boundary_satisfying_models = [
        lambda x, y, model: model(x, y) * (1 - torch.pow(x, 2)) * (1 - torch.pow(y, 2))
    ]

    n_epochs = 50
    nn_ode_solver = TrainerForNNEquationSolver(main_eq,
                                               n_epochs=n_epochs,
                                               boundary_satisfying_models=boundary_satisfying_models,
                                               n_hidden_neurons=20,
                                               lr=1)
    loss_train, loss_valid, nn_models = nn_ode_solver.fit()
    report = ReportMaker(nn_models,
                         loss_train,
                         loss_valid,
                         main_domain,
                         compare_to_functions=plot_two_2d_functions,
                         analytical_solutions=true_solution,
                         main_eq_residuals=main_eq_residuals
                         )
    report.plot_abs_residual_distr()
    report.print_loss_history()
    report.compare_appr_with_analytical()
    report.print_comparison_table()

    # x1_points = 20
    # x1_left = -1
    # x1_right = 1
    # x2_points = 20
    # x2_left = -1
    # x2_right = 1
    #
    # def true_sol(x, y):
    #     total_sum = 0
    #     lim = 100
    #     for i in range(1, lim, 2):
    #         for j in range(1, lim, 2):
    #             total_sum += (
    #                 pow(-1, (i + j) // 2 - 1)
    #                 / (i * j * (i * i + j * j))
    #                 * torch.cos(i * pi / 2 * x)
    #                 * torch.cos(j * pi / 2 * y)
    #             )
    #     total_sum *= 8 * 8 / (pi * pi * pi * pi)
    #     return total_sum
    #
    # true_solution = true_sol
    #
    # main_domain = TwoDimensionalSimpleDomain(
    #     x1_points, x1_left, x1_right, x2_points, x2_left, x2_right
    # )
    # main_eq_residuals = lambda x, y, nn_model: (
    #     nth_derivative(nn_model(x, y), x, 2)
    #     + nth_derivative(nn_model(x, y), y, 2)
    #     + torch.tensor(1.0, requires_grad=True)
    # )
    #
    # bc = lambda x, y, model: model(x, y)
    # bc1 = TwoDimInitialCondition(
    #     const_var=-1,
    #     non_const_var_left=-1,
    #     non_const_var_right=1,
    #     non_const_var_size=10,
    #     equation=bc,
    #     const_var_ind=1,
    # )
    # bc2 = TwoDimInitialCondition(1, -1, 1, 10, bc, 1)
    # bc3 = TwoDimInitialCondition(-1, -1, 1, 10, bc, 2)
    # bc4 = TwoDimInitialCondition(1, -1, 1, 10, bc, 2)
    # boundary_conditions = [bc1, bc2, bc3, bc4]
    # main_eq = MainEquationClass(
    #     main_domain, main_eq_residuals, boundary_conditions=boundary_conditions
    # )
    #
    # n_epochs = 50
    # nn_ode_solver = TrainerForNNEquationSolver(main_eq, n_epochs=n_epochs, lr=1e-1)
    # loss_train, loss_valid, nn_models = nn_ode_solver.fit()
    # report = ReportMaker(nn_models,
    #                      loss_train,
    #                      loss_valid,
    #                      main_domain,
    #                      compare_to_functions=plot_two_2d_functions,
    #                      analytical_solutions=true_solution,
    #                      main_eq_residuals=main_eq_residuals
    #                      )
    # report.plot_abs_residual_distr()
    # report.print_loss_history()
    # report.compare_appr_with_analytical()
    # report.print_comparison_table()

from SourceCode import *

import torch
from math import pi, e

if __name__ == "__main__":

    def true_sol_test23(x, y):
        total_sum = 0
        lim = 100
        for i in range(1, lim, 2):
            for j in range(1, lim, 2):
                total_sum += (
                    pow(-1, (i + j) // 2 - 1)
                    / (i * j * (i * i + j * j))
                    * torch.cos(i * pi / 2 * x)
                    * torch.cos(j * pi / 2 * y)
                )
        total_sum *= 8 * 8 / (pi * pi * pi * pi)
        return total_sum

    def test_1():  # system of odes
        left_bound = 0
        right_bound = 5
        main_eq_residual1 = lambda t, x, y: nth_derivative(x(t), t, 1) + y(t)
        main_eq_residual2 = (
            lambda t, x, y: nth_derivative(y(t), t, 1) - x(t) - torch.cos(t)
        )
        main_eq_residuals = [main_eq_residual1, main_eq_residual2]
        n_points = 20
        main_domain = OneDimensionalSimpleDomain(left_bound, right_bound, n_points)
        # approximation satisfies boundary conditions
        main_eq = MainEquationClass(main_domain, main_eq_residuals)
        boundary_satisfying_models = [
            lambda x, model1: x * model1(x),
            lambda x, model2: x * model2(x),
        ]
        n_epochs = 40
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
        report = ReportMaker(
            nn_models,
            loss_train,
            loss_valid,
            main_domain,
            analytical_solutions=analytical_solutions,
            compare_to_functions=plot_two_curves,
            main_eq_residuals=main_eq_residuals,
        )
        report.plot_abs_residual_distr()
        report.print_loss_history()
        report.compare_appr_with_analytical()
        report.print_comparison_table()

    def test_2():  # 2d pde satisfying boundary conditions
        x1_points = 20
        x1_left = -1
        x1_right = 1
        x2_points = 20
        x2_left = -1
        x2_right = 1

        true_solution = true_sol_test23

        main_domain = TwoDimensionalSimpleDomain(
            x1_points, x1_left, x1_right, x2_points, x2_left, x2_right, offset=0.1
        )
        main_eq_residuals = lambda x, y, nn_model: (
            nth_derivative(nn_model(x, y), x, 2)
            + nth_derivative(nn_model(x, y), y, 2)
            + torch.tensor(1.0, requires_grad=True)
        )

        main_eq = MainEquationClass(main_domain, main_eq_residuals)
        boundary_satisfying_models = [
            lambda x, y, model: model(x, y)
            * (1 - torch.pow(x, 2))
            * (1 - torch.pow(y, 2))
        ]

        n_epochs = 50
        models = NNGenerator.generate_models(
            nn_count=1, inp_dim=2, boundary_satisfying_models=boundary_satisfying_models,
            nn_type='BasisFuncMLP'
        )

        nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs, nn_models=models
        )
        loss_train, loss_valid, nn_models = nn_ode_solver.fit()
        report = ReportMaker(
            nn_models,
            loss_train,
            loss_valid,
            main_domain,
            compare_to_functions=plot_two_2d_functions,
            analytical_solutions=true_solution,
            main_eq_residuals=main_eq_residuals,
        )
        report.plot_abs_residual_distr()
        report.print_loss_history()
        report.compare_appr_with_analytical()
        report.print_comparison_table()

    def test_3():  # 2d pde

        x1_points = 20
        x1_left = -1
        x1_right = 1
        x2_points = 20
        x2_left = -1
        x2_right = 1

        true_solution = true_sol_test23
        offset = 0.01
        main_domain = TwoDimensionalSimpleDomain(
            x1_points, x1_left, x1_right, x2_points, x2_left, x2_right, offset=offset
        )

        main_eq_residuals = lambda x, y, nn_model: (
            nth_derivative(nn_model(x, y), x, 2)
            + nth_derivative(nn_model(x, y), y, 2)
            + torch.tensor(1.0, requires_grad=True)
        )

        bc = lambda x, y, model: model(x, y)
        bc1 = TwoDimInitialCondition(
            non_const_var_left_value=-1,
            non_const_var_right_value=1,
            non_const_var_size=10,
            equation=bc,
            const_var_value=1,
            const_var_ind=1,
        )
        bc2 = TwoDimInitialCondition(-1, 1, 10, bc, -1, 1)
        bc3 = TwoDimInitialCondition(-1+offset, 1-offset, 10, bc, 1, 2)
        bc4 = TwoDimInitialCondition(-1+offset, 1-offset, 10, bc, -1, 2)
        boundary_conditions = [bc1, bc2, bc3, bc4]
        main_eq = MainEquationClass(
            main_domain,
            main_eq_residuals,
            boundary_conditions=boundary_conditions,
            bound_cond_coef=0.1
        )

        n_epochs = 30
        nn_params = {'hidden_dim': 50, 'num_hidden_layers': 1}
        models = NNGenerator.generate_models(nn_count=1, inp_dim=2,
                                             nn_params=nn_params)

        nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs, nn_models=models,
            lr=1
        )
        loss_train, loss_valid, nn_models = nn_ode_solver.fit()
        report = ReportMaker(
            nn_models,
            loss_train,
            loss_valid,
            main_domain,
            compare_to_functions=plot_two_2d_functions,
            analytical_solutions=true_solution,
            main_eq_residuals=main_eq_residuals,
        )
        report.plot_abs_residual_distr()
        report.print_loss_history()
        report.compare_appr_with_analytical()
        report.print_comparison_table()

    def test_4():  # ode with a parameter, in fact a 2d equation
        x_points = 20
        x_left = 0
        x_right = 1
        param_points = 20
        param_left = 0
        param_right = 1

        true_solution = (
            lambda x, param: -(e ** (-2 * x))
            + 2 * e ** (2 * x)
            - param / 8 * torch.sin(2 * x)
        )

        main_domain = TwoDimensionalSimpleDomain(
            x_points,
            x_left,
            x_right,
            param_points,
            param_left,
            param_right,
            offset=0.01,
        )
        main_eq_residuals = lambda x, param, nn_model: (
            nth_derivative(nn_model(x, param), x, 2)
            - 4 * nn_model(x, param)
            - param * torch.sin(2 * x)
        )

        init1_eq = lambda x, param, nn_model: nn_model(x, param) - torch.tensor(
            1.0, requires_grad=True
        )
        init_cond1 = TwoDimInitialCondition(
            non_const_var_left_value=0,
            non_const_var_right_value=1,
            non_const_var_size=10,
            equation=init1_eq,
            const_var_value=0,
            const_var_ind=1,
        )

        init2_eq = lambda x, param, nn_model: nth_derivative(
            nn_model(x, param), x, 1
        ) - torch.tensor(6.0, requires_grad=True)
        init_cond2 = TwoDimInitialCondition(
            non_const_var_left_value=0,
            non_const_var_right_value=1,
            non_const_var_size=10,
            equation=init2_eq,
            const_var_value=0,
            const_var_ind=1,
        )
        init_conds = [init_cond1, init_cond2]

        main_eq = MainEquationClass(
            main_domain, main_eq_residuals, boundary_conditions=init_conds
        )

        n_epochs = 40
        models = NNGenerator.generate_models(nn_count=1, inp_dim=2)

        nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs, nn_models=models, lr=0.5
        )
        loss_train, loss_valid, nn_models = nn_ode_solver.fit()
        report = ReportMaker(
            nn_models,
            loss_train,
            loss_valid,
            main_domain,
            compare_to_functions=plot_two_2d_functions,
            analytical_solutions=true_solution,
            main_eq_residuals=main_eq_residuals,
        )
        report.plot_abs_residual_distr(offset=0.01)
        report.print_loss_history()
        report.compare_appr_with_analytical(offset=0.01)
        report.print_comparison_table()

    def test_5():
        left_border = 0
        right_border = 1
        n_points = 20
        integration_func = lambda curr_v, int_domain, nn_model: nn_model(int_domain)
        main_eq_res = (lambda curr_v, nn_model: nn_model(curr_v) -
                                                torch.sin(pi * curr_v) -
                                                0.5 * IntegralEquations.calculate_fredholm_equation_1d_gauss_quadratures(
            integration_func,
            nn_model,
            curr_v,
            left_border,
            right_border))
        main_domain = OneDimensionalSimpleDomain(0, 1, n_points, offset=0)

        main_eq = MainEquationClass(main_domain, main_eq_res)
        n_epochs = 40
        nn_params = {'layers_hidden': [1, 10, 10, 1]}
        models = NNGenerator.generate_models(
            nn_count=1,
            nn_params=nn_params,
            nn_type='KAN'
        )

        nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs,
            nn_models=models,
            lr=0.3,
        )
        loss_train, loss_valid, nn_model = nn_ode_solver.fit()
        analytical_solution = lambda x_var: torch.sin(pi * x_var) + 2 / pi
        report = ReportMaker(nn_model,
                             loss_train,
                             loss_valid,
                             main_domain,
                             compare_to_functions=plot_two_1d_functions,
                             analytical_solutions=analytical_solution,
                             main_eq_residuals=main_eq_res
                             )
        report.print_loss_history()
        report.compare_appr_with_analytical(offset=0)
        report.plot_abs_residual_distr(offset=0)

    def test_6():
        left_bound = 0
        right_bound = 1
        main_eq_residual = (
            lambda x, nn_model: nth_derivative(nn_model(x), x, 2)
                                + 0.2 * nth_derivative(nn_model(x), x, 1)
                                + nn_model(x)
                                + 0.2 * torch.exp(-x / 5) * torch.cos(x)
        )
        n_points = 10

        main_domain = OneDimensionalSimpleDomain(left_bound, right_bound, n_points)

        first_init_cond_res = lambda x, nn_model: nn_model(x) - 0
        first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

        second_init_cond_res = lambda x, nn_model: nn_model(x) - torch.sin(
            torch.Tensor([1])
        ) * torch.exp(torch.Tensor([-0.2]))
        second_init_cond = OnePointInitialCondition(right_bound, second_init_cond_res)

        boundary_conditions = [first_init_cond, second_init_cond]

        main_eq = MainEquationClass(main_domain, main_eq_residual, boundary_conditions)

        n_epochs = 20
        nn_params = {'hidden_dim': 20, 'num_hidden_layers': 2}
        models = NNGenerator.generate_models(
            nn_count=1,
            nn_params=nn_params,
            nn_type='BasisFuncMLP'
        )

        nn_ode_solver = TrainerForNNEquationSolver(
            main_eq, n_epochs=n_epochs,
            nn_models=models,
            lr=0.2,
        )
        analytical_solution = lambda x: torch.exp(-x / 5) * torch.sin(x)
        loss_train, loss_valid, abs_error_train, abs_error_valid, nn_model = \
            nn_ode_solver.fit_with_abs_err_history(main_domain, analytical_sols=[analytical_solution])
        report = ReportMaker(nn_model,
                             loss_train,
                             loss_valid,
                             main_domain,
                             compare_to_functions=plot_two_1d_functions,
                             analytical_solutions=analytical_solution,
                             main_eq_residuals=main_eq_residual
                             )
        report.print_loss_history()
        report.compare_appr_with_analytical()


test_6()

import torch
from SourceCode.utilities import nth_derivative, print_comparison_table
from SourceCode.EquationClass import OneDimensionalMainEquation
from SourceCode.DomainClass import OneDimensionalSimpleDomain
from SourceCode.InitConditionClass import OnePointInitialCondition
from SourceCode.TrainerForNNEquationSolver import TrainerForNNEquationSolver
from SourceCode.ReportMaker import ReportMaker

if __name__ == "__main__":
    left_bound = 0
    right_bound = 1
    main_eq_residual = (
        lambda x, nn_model_value: nth_derivative(nn_model_value, x, 2)
        + 0.2 * nth_derivative(nn_model_value, x, 1)
        + nn_model_value
        + 0.2 * torch.exp(-x / 5) * torch.cos(x)
    )
    n_points = 10
    dh = 1e-3
    main_domain = OneDimensionalSimpleDomain(left_bound + dh, right_bound - dh, n_points)
    main_eq = OneDimensionalMainEquation(main_domain, main_eq_residual)

    first_init_cond_res = lambda x, nn_model_value: nn_model_value - 0
    first_init_cond = OnePointInitialCondition(left_bound, first_init_cond_res)

    second_init_cond_res = lambda x, nn_model_value: nn_model_value - torch.sin(
        torch.Tensor([1])
    ) * torch.exp(torch.Tensor([-0.2]))
    second_init_cond = OnePointInitialCondition(right_bound, second_init_cond_res)

    boundary_conditions = [first_init_cond, second_init_cond]

    true_solution = lambda x: torch.exp(-x / 5) * torch.sin(x)
    n_epochs = 10
    nn_ode_solver = TrainerForNNEquationSolver(main_eq, boundary_conditions, n_epochs)
    loss_train, loss_valid, nn_model = nn_ode_solver.fit()
    report = ReportMaker(
        true_solution, nn_model, main_eq, loss_train, loss_valid, main_domain, n_epochs
    )
    report.make_report()
    print_comparison_table(main_domain.get_valid_domain(), nn_model, true_solution)
from typing import List, Callable, Union
from .nn_architectures import *
from .NeuralNetworkFunction import (
    NeuralNetworkFunctionWrapper1D,
    NeuralNetworkFunctionWrapper2D,
)
from .utilities import set_seed

import torch.nn


class NNGenerator:
    @staticmethod
    def generate_models(
        nn_type: str = "MLP",
        nn_params: dict = None,
        nn_count: int = 1,
        inp_dim: int = 1,
        boundary_satisfying_models: List[Callable] = None,
        seed: int = 42
    ):
        """
        :param nn_type: can be MLP, ResMLP, KAN or BasisFuncMLP
        :param nn_params: params for the NN you have chosen
        :param nn_count: number of neural networks, should be equal to the number of equations
        :param inp_dim: The dimension of the main domain
        :param boundary_satisfying_models: optionally, list of models satisfying boundary conditions
        :return:
        """
        assert nn_type in ["MLP", "ResMLP", "KAN", 'BasisFuncMLP']
        assert inp_dim in [1, 2], 'currently only 1-d and 2-d domains are supported'
        model_types = {"MLP": MLP, "ResMLP": ResidualMLP, "KAN": KAN, 'BasisFuncMLP': BasisFuncMLP}

        basis_funcs = {1: [lambda x: torch.ones(x.shape), lambda x: x, lambda x: x * x],
                       2: [lambda x, y: torch.ones(x.shape), lambda x, y: x, lambda x, y: y,
                           lambda x, y: x*y]}
        def_params = {
            "MLP": {"input_dim": inp_dim, "hidden_dim": 20, "num_hidden_layers": 1},
            "ResMLP": {"input_dim": inp_dim, "hidden_dim": 20, "num_hidden_layers": 1},
            "KAN": {"layers_hidden": [inp_dim, 20, 1]},
            "BasisFuncMLP": {"input_dim": inp_dim, "hidden_dim": 20, "num_hidden_layers": 1,
                             'basis_funcs': basis_funcs[inp_dim]}
        }

        if nn_params is None:
            nn_params = def_params[nn_type]

        if nn_type in ['MLP', 'ResMLP', 'BasisFuncMLP']:
            nn_params['input_dim'] = inp_dim
        if nn_type == 'BasisFuncMLP' and 'basis_funcs' not in nn_params:
            nn_params['basis_funcs'] = basis_funcs[inp_dim]


        nn_dim_type = {
            1: NeuralNetworkFunctionWrapper1D,
            2: NeuralNetworkFunctionWrapper2D,
        }
        model_type = model_types[nn_type]
        if boundary_satisfying_models is None:
            boundary_satisfying_models = [None for _ in range(nn_count)]
        elif not isinstance(boundary_satisfying_models, list):
            boundary_satisfying_models = [boundary_satisfying_models]
        else:
            boundary_satisfying_models = boundary_satisfying_models
        set_seed(seed)
        nn_models = NNGenerator.__get_nn_models(
            boundary_satisfying_models, nn_params, model_type, nn_dim_type[inp_dim]
        )
        return nn_models

    @staticmethod
    def __get_nn_models(
        boundary_satisfying_models,
        nn_params,
        model_type,
        model_dim_type: Union[
            NeuralNetworkFunctionWrapper1D, NeuralNetworkFunctionWrapper2D
        ],
    ) -> List[Callable[[torch.tensor], torch.tensor]]:
        nn_models = []
        n = len(boundary_satisfying_models)
        for i in range(n):
            nn_model = model_dim_type(
                boundary_satisfying_models[i], nn_params, model_type
            )
            nn_models.append(nn_model)
        return nn_models

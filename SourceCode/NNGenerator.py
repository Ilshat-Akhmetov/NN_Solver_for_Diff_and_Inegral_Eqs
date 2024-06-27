from typing import List, Callable, Union
from .nn_architectures import *
from .NeuralNetworkFunction import (
    NeuralNetworkFunctionWrapper1D,
    NeuralNetworkFunctionWrapper2D,
)
from .SeedGen import SeedGen

import torch.nn


class NNGenerator(SeedGen):
    @staticmethod
    def generate_models(
        nn_type: str = "MLP",
        nn_params: dict = None,
        nn_count: int = 1,
        inp_dim: int = 1,
        boundary_satisfying_models: List[Callable] = None,
    ):
        """
        :param nn_type: can be mlp, ResMLP or KAN
        :param nn_params: params for the NN you have chosen
        :param nn_count: number of neural networks, should be equal to the number of equations
        :param inp_dim: The dimension of the main domain
        :param boundary_satisfying_models: optionally, list of models satisfying boundary conditions
        :return:
        """
        assert nn_type in ["MLP", "ResMLP", "KAN"]
        model_types = {"MLP": MLP, "ResMLP": ResidualMLP, "KAN": KAN}

        def_params = {
            "MLP": {"input_dim": inp_dim, "hidden_dim": 20, "num_hidden_layers": 1},
            "ResMLP": {"input_dim": inp_dim, "hidden_dim": 20, "num_hidden_layers": 1},
            "KAN": {"layers_hidden": [inp_dim, 20, 1]},
        }
        if nn_params is None:
            nn_params = def_params[nn_type]

        if nn_type in ['MLP', 'ResMLP']:
            nn_params['input_dim'] = inp_dim

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
        NNGenerator.set_seed(77)
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

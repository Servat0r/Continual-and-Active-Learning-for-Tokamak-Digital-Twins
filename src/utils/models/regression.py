from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models.base_model import BaseModel

from .utils import *


class SimpleRegressionMLP(nn.Module, BaseModel):
    def __init__(
        self,
        output_size: int = 4,
        input_size: int = 15,
        hidden_size: int = 32,
        hidden_layers: int = 1,
        drop_rate: float = 0.5,
        dtype: str = 'float32',
        final_layer: Literal["softplus", "relu", "elu", "id"] = None,
        activation: Literal["relu", "tanh", "elu", "softplus"] = 'relu',
    ):
        """
        :param output_size: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        :param dtype: data type of weights and biases
        :param final_layer: Final layer of the model. It can be either
        "softplus", "relu", "elu" or "id". If None, it is the identity function.
        :param activation: activation function. Options are: 'relu', 'tanh',
        'elu', 'softplus'.
        """
        super().__init__()
        dtype = get_dtype_from_str(dtype)

        if activation == 'relu':
            activation_class = nn.ReLU
            activation_kwargs = {'inplace': False}
        elif activation == 'elu':
            activation_class = nn.ELU
            activation_kwargs = {'inplace': False, 'alpha': 0.5}
        elif activation == 'softplus':
            activation_class = nn.Softplus
            activation_kwargs = {'inplace': False}
        elif activation == 'tanh':
            activation_class = nn.Tanh
            activation_kwargs = {}
        else:
            raise ValueError(f'Activation {activation} not supported')

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size, dtype=dtype),
                activation_class(**activation_kwargs),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size, dtype=dtype),
                        activation_class(**activation_kwargs),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_size, output_size, dtype=dtype)
        if final_layer == 'softplus':
            final = nn.Softplus()
        elif final_layer == 'relu':
            final = nn.ReLU()
        elif final_layer == 'elu':
            final = nn.ELU()
        else:
            final = nn.Identity()
        self.final = final
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.regressor(x)
        x = self.final(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x


class GaussianRegressionMLP(nn.Module, BaseModel):

    def __init__(
        self,
        output_size=4,
        input_size=15,
        hidden_size=8,
        hidden_layers=1,
        drop_rate=0.5,
        dtype='float32',
        final_layer: Literal["softplus", "relu", "elu", "id"] = None,
        activation: Literal["relu", "tanh", "elu", "softplus"] = 'relu',
    ):
        """
        :param output_size: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        :param dtype: data type of weights and biases
        :param final_layer: Final layer of the model. It can be either
        "softplus", "relu", "elu" or "id". If None, it is the identity function.
        :param activation: activation function. Options are: 'relu', 'tanh',
        'elu', 'softplus'.
        """
        super().__init__()
        dtype = get_dtype_from_str(dtype)

        if activation == 'relu':
            activation_class = nn.ReLU
            activation_kwargs = {'inplace': True}
        elif activation == 'elu':
            activation_class = nn.ELU
            activation_kwargs = {'inplace': True, 'alpha': 0.5}
        elif activation == 'softplus':
            activation_class = nn.Softplus
            activation_kwargs = {'inplace': True}
        elif activation == 'tanh':
            activation_class = nn.Tanh
            activation_kwargs = {}
        else:
            raise ValueError(f'Activation {activation} not supported')

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size, dtype=dtype),
                activation_class(**activation_kwargs),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size, dtype=dtype),
                        activation_class(**activation_kwargs),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_size, output_size, dtype=dtype)
        if final_layer == 'softplus':
            final = nn.Softplus()
        elif final_layer == 'relu':
            final = nn.ReLU()
        elif final_layer == 'elu':
            final = nn.ELU()
        else:
            final = nn.Identity()
        self.final = final
        self.variance_layer = nn.Linear(hidden_size, output_size, dtype=dtype)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        mean = self.mean_layer(x)
        mean = self.final(mean)
        std = F.softplus(self.variance_layer(x)) # ?
        #std = self.variance_layer(x)
        return torch.cat([mean, std], dim=1)

    def get_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x


class SimpleConv1DModel(nn.Module):

    def __init__(
            self, in_features: int = 15, out_channels1: int = 32, out_channels2: int = 64,
            hidden_size: int = 128, out_features: int = 4, kernel_size: int = 3,
            padding: int = 1, dtype='float64'
    ):
        super(SimpleConv1DModel, self).__init__()
        dtype = get_dtype_from_str(dtype)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=out_channels1, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size, padding=padding
        )

        # Update the fc1 layer input size to match the flattened output from conv2
        self.fc1 = nn.Linear(out_channels2 * in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)  # Output layer with 4 dimensions

        # Set the model to the specified dtype
        self.to(dtype=dtype)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension to make shape [batch_size, 1, 15]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the output before feeding it into the fully connected layers
        x = x.view(x.size(0), -1)
        # Fully connected part
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


__all__ = [
    "SimpleRegressionMLP",
    "GaussianRegressionMLP",
    "SimpleConv1DModel"
]
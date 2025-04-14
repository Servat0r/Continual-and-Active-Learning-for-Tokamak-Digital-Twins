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

    def get_raw_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x
    
    def get_features(self, x, with_final: bool = True): # TODO True or False?
        x = self.get_raw_features(x)
        x = self.regressor(x)
        if with_final:
            x = self.final(x)
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
        #x = x.contiguous()
        #x = self.features(x)
        return self.forward(x)


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


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=8, num_layers=2, dropout=0.25):
        """
        Initializes a Transformer-based model for tabular regression.

        Args:
            input_size (int): Number of features (columns) in the tabular dataset.
            output_size (int): Number of regression outputs.
            d_model (int): Dimension of the embedding and hidden representations.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout probability used in encoder layers.
        """
        super(TransformerRegressor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        # Embed each scalar feature into a d_model-dimensional vector.
        self.embedding = nn.Linear(1, d_model)
        
        # Create one Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            activation='relu'
        )
        # Stack encoder layers.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Adaptive average pooling to aggregate outputs over token dimension.
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.first_fc = nn.Linear(self.d_model * self.input_size, self.d_model * 2)
        self.second_fc = nn.Linear(self.d_model * 2, self.d_model)

        # Final fully connected layer maps d_model to output_size.
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        """
        Forward pass through the Transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) containing regression predictions.
        """
        # x has shape: (batch_size, input_size)
        # Unsqueeze to shape (batch_size, input_size, 1) so that each feature can be embedded.
        x = x.unsqueeze(-1)
        
        # Embed each scalar input to shape (batch_size, input_size, d_model)
        x = self.embedding(x)
        
        # Transformer encoder expects (seq_length, batch_size, d_model)
        # Here each token represents a feature so seq_length = input_size.
        x = x.transpose(0, 1)
        
        # Pass through the Transformer encoder (no positional encoding applied).
        x = self.transformer_encoder(x)
        
        # Transpose back to (batch_size, input_size, d_model)
        x = x.transpose(0, 1)

        # Reshape x into (batch_size, d_model * input_size)
        x = x.reshape(x.size(0), -1)

        # Apply two fully connected layers with ReLU activation
        x = nn.functional.relu(self.first_fc(x))
        x = self.second_fc(x)
        """
        # OLD CODE!
        
        # Rearrange tensor to (batch_size, d_model, input_size) for pooling.
        x = x.transpose(1, 2)
        
        # Pool across the feature tokens to produce (batch_size, d_model, 1), then squeeze.
        x = self.pool(x).squeeze(-1)  # shape becomes (batch_size, d_model)
        """
        
        # Final regression outputs.
        output = self.fc(x)  # shape (batch_size, output_size)
        return output


__all__ = [
    "SimpleRegressionMLP",
    "GaussianRegressionMLP",
    "SimpleConv1DModel",
    "TransformerRegressor"
]
from torch import float32, float64, float16
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models.base_model import BaseModel


def _get_dtype_from_str(dtype_str: str):
    if dtype_str == "float32":
        return float32
    elif dtype_str == "float64":
        return float64
    elif dtype_str == "float16":
        return float16
    else:
        raise ValueError(
            f"Unsupported data type \"{dtype_str}\". Supported data types are: float32, float64, float16"
        )

def get_model_size(model):
    trainables, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainables += p.numel()
    return trainables, total


class SimpleRegressionMLP(nn.Module, BaseModel):
    def __init__(
        self,
        output_size=4,
        input_size=15,
        hidden_size=8,
        hidden_layers=1,
        drop_rate=0.5,
        dtype='float32',
    ):
        """
        :param output_size: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        dtype = _get_dtype_from_str(dtype)

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size, dtype=dtype),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size, dtype=dtype),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_size, output_size, dtype=dtype)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.regressor(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x


class SimpleClassificationMLP(nn.Module, BaseModel):

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
        dtype='float32',
    ):
        super().__init__()
        dtype = _get_dtype_from_str(dtype)

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size, dtype=dtype),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size, dtype=dtype),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.logits_layer = nn.Linear(hidden_size, num_classes, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.logits_layer(x)
        x = self.sigmoid(x)
        return x

    def get_logits(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.logits_layer(x)
        return x

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
        dtype = _get_dtype_from_str(dtype)
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
    "SimpleClassificationMLP",
    "SimpleConv1DModel",
    "get_model_size"
]
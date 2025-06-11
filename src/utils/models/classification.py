import torch
from avalanche.models import BaseModel
from torch import nn as nn

from ..misc import get_dtype_from_str


class SimpleClassificationMLP(nn.Module, BaseModel):

    def __init__(
        self,
        output_size=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
        dtype='float32',
    ):
        super().__init__()
        dtype = get_dtype_from_str(dtype)

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
        self.logits_layer = nn.Linear(hidden_size, output_size, dtype=dtype)
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

    def classify(self, x, as_type=torch.int):
        x = self.forward(x)
        return (x >= 0.5).type(as_type)


__all__ = ['SimpleClassificationMLP']
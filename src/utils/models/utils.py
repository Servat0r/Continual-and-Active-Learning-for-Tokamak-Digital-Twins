from torch import float32, float64, float16
from torch import equal as torch_equal
import torch.nn as nn


def get_dtype_from_str(dtype_str: str):
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


def are_models_equal(model1, model2):
    # Get the state_dict of both models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if they have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Compare each corresponding tensor
    for key in state_dict1.keys():
        if not torch_equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def initialize_weights_low(module, scale=1e-2):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -scale, scale)  # Small uniform values
        if module.bias is not None:
            nn.init.uniform_(module.bias, -scale, scale)  # Small bias values


__all__ = [
    'get_model_size', 'get_dtype_from_str', 'are_models_equal', 'initialize_weights_low'
]
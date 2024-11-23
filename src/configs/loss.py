from typing import Any
from .parser import *
from torch.nn import MSELoss, HuberLoss, BCELoss, BCEWithLogitsLoss
from ..utils import GaussianNLLLoss


@ConfigParser.register_handler('loss')
def loss_handler(data: dict[str, Any], **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if (name == 'mse') or (name == 'MSE'):
        return MSELoss(**parameters)
    elif (name == 'huber') or (name == 'Huber'):
        return HuberLoss(**parameters)
    elif (name == 'BCE') or (name == 'bce'):
        return BCELoss(**parameters)
    elif (name == 'BCEWithLogits') or (name == 'bce_with_logits'):
        return BCEWithLogitsLoss(**parameters)
    elif (name == 'GaussianNLL') or (name == 'gaussian_nll'):
        return GaussianNLLLoss(**parameters)
    else:
        raise ValueError(f"Invalid loss name \"{name}\"")


__all__ = ['loss_handler']
from typing import Any
from .parser import *
from torch.nn import MSELoss, HuberLoss, BCELoss, BCEWithLogitsLoss
from ..utils import GaussianNLLLoss, MSECosineSimilarityLoss, RootMSELoss


@ConfigParser.standardizer('loss')
def loss_std(config: dict[str, Any], key: str):
    return config[key]


@ConfigParser.processor('loss')
def loss_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    match name.lower():
        case 'mse':
            return MSELoss(**parameters)
        case 'rmse' | 'rootmse' | 'root_mse':
            return RootMSELoss(**parameters)
        case 'huber':
            return HuberLoss(**parameters)
        case 'bce':
            return BCELoss(**parameters)
        case 'bcewithlogits' | 'bce_with_logits':
            return BCEWithLogitsLoss(**parameters)
        case 'gaussiannll' | 'gaussian_nll':
            return GaussianNLLLoss(**parameters)
        case 'msecosinesimilarity' | 'mse_cosine_similarity':
            return MSECosineSimilarityLoss(**parameters)
        case _:
            raise ValueError(f"Invalid loss name \"{name}\"")


__all__ = ['loss_std', 'loss_handler']
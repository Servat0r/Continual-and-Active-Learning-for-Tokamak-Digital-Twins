from typing import Any
from .parser import *
from torch.optim import SGD, Adam, AdamW, Adagrad, RMSprop


@ConfigParser.register_handler('optimizer')
def optimizer_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if (name == 'SGD') or (name == 'sgd'):
        return {
            'class': SGD,
            'parameters': parameters,
        }
    elif (name == 'Adam') or (name == 'adam'):
        return {
            'class': Adam,
            'parameters': parameters,
        }
    elif (name == 'AdamW') or (name == 'adamw'):
        return {
            'class': AdamW,
            'parameters': parameters,
        }
    elif (name == 'Adagrad') or (name == 'adagrad'):
        return {
            'class': Adagrad,
            'parameters': parameters,
        }
    elif (name == 'RMSprop') or (name == 'rmsprop'):
        return {
            'class': RMSprop,
            'parameters': parameters,
        }
    else:
        raise ValueError(f"Invalid optimizer name \"{name}\"")


__all__ = ['optimizer_handler']
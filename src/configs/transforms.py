from typing import Any

from .parser import *
from ..utils import LogarithmTransform, ReversableTransform, SqrtTransform

__names_dict__: dict[str, type(ReversableTransform)] = {
    'LogarithmTransform': LogarithmTransform,
    'SqrtTransform': SqrtTransform,
}


def __transform_inside_loop(data):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if name not in __names_dict__:
        raise ValueError(f"Invalid transform name \"{name}\"")
    transform_class = __names_dict__[name]
    transform = transform_class(**parameters)
    return transform


@ConfigParser.register_handler('transform')
def transform_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    transform = __transform_inside_loop(data)
    return {
        'transform': transform,
        'preprocess_ytrue': transform.inverse(),
        'preprocess_ypred': transform.inverse(),
    }


@ConfigParser.register_handler('target_transform')
def target_transform_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    transform = __transform_inside_loop(data)
    return {
        'target_transform': transform,
        'preprocess_ytrue': transform.inverse(),
        'preprocess_ypred': transform.inverse(),
    }


__all__ = ['transform_handler', 'target_transform_handler']
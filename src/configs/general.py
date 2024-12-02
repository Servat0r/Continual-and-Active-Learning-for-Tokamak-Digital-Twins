from typing import Any
from .parser import *


@ConfigParser.register_handler('general')
def general_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    default_config = {
        'train_mb_size': 512,
        'eval_mb_size': 2048,
        'train_epochs': 250,
        'num_campaigns': 10,
        'dtype': 'float32',
        'task': 'regression',
    }
    default_config.update(data)
    assert isinstance(default_config['train_mb_size'], int) and default_config['train_mb_size'] > 0
    assert isinstance(default_config['eval_mb_size'], int) and default_config['eval_mb_size'] > 0
    assert isinstance(default_config['train_epochs'], int) and default_config['train_epochs'] > 0
    assert isinstance(default_config['num_campaigns'], int) and default_config['num_campaigns'] > 0
    assert default_config['dtype'] in ['float16', 'float32', 'float64']
    assert default_config['task'] in ['regression', 'classification']
    return default_config


__all__ = ['general_handler']
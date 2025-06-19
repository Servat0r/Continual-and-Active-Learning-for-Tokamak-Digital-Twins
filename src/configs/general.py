from typing import Any
from .parser import *


_general_al_conv = {
    'full_first_train_set': 'full_first_set',
    'first_train_set_size': 'first_set_size',
    'downsampling_factor': 'downsampling_factor'
}


@ConfigParser.standardizer('general')
def general_handler(config: dict[str, Any], key: str):
    default_config = {
        'mode': 'CL', # Available modes are: "CL" (pure Continual Learning),
        # "AL(CL)" (CL with experiences data selected with AL methods)
        'full_first_train_set': True,
        'downsampling_factor': 1,
        'first_train_set_size': None, # By default, it is ENTIRE!
        'train_mb_size': 4096,
        'eval_mb_size': 4096,
        'train_epochs': 200,
        'num_campaigns': 10,
        'dtype': 'float32', # Either "float16", "float32", "float64"
        'task': 'regression', # Either "regression" or "classification"
    }
    default_config.update(config[key])
    assert isinstance(default_config['mode'], str) and default_config['mode'] in ['CL', 'AL(CL)']
    assert isinstance(default_config['full_first_train_set'], bool)
    assert (default_config['first_train_set_size'] is None) or (isinstance(default_config['first_train_set_size'], int))
    assert isinstance(default_config['downsampling_factor'], int) and default_config['downsampling_factor'] > 0
    # Extract AL fields from general (backwards compatibility)
    for field, conv_field in _general_al_conv.items():
        if field in default_config:
            val = default_config.pop(field)
            if 'active_learning' in config:
                config['active_learning'][conv_field] = val
    assert isinstance(default_config['train_mb_size'], int) and default_config['train_mb_size'] > 0
    assert isinstance(default_config['eval_mb_size'], int) and default_config['eval_mb_size'] > 0
    assert isinstance(default_config['train_epochs'], int) and default_config['train_epochs'] > 0
    assert isinstance(default_config['num_campaigns'], int) and default_config['num_campaigns'] > 0
    assert default_config['dtype'] in ['float16', 'float32', 'float64']
    assert default_config['task'] in ['regression', 'classification']
    return default_config


__all__ = ['general_handler']
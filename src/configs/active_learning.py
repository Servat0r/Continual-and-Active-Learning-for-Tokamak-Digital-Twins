from typing import Any

from src.utils.active_learning import *
from .parser import *


_BATCH_SELECTORS = {
    'bmdal': BMDALBatchSelector,
    'mc_dropout': MCDropoutBatchSelector,
    'deep_ensemble': DeepEnsembleBatchSelector,
}


def _bmdal_params_handler(parameters):
    assert isinstance(parameters['batch_size'], int)
    assert isinstance(parameters['selection_method'], str) and \
        (parameters['selection_method'] in ['lcmd', ])
    assert isinstance(parameters['base_kernel'], str) and \
        (parameters['base_kernel'] in ['grad', ])
    assert isinstance(parameters['kernel_transforms'], list)
    kernel_transforms = [
        tuple(item) if isinstance(item, list) else item for item in parameters['kernel_transforms']
    ]
    parameters['kernel_transforms'] = kernel_transforms
    return parameters


def _mcdropout_params_handler(parameters):
    return parameters


def _deep_ensemble_params_handler(parameters):
    return parameters


@ConfigParser.register_handler('active_learning')
def active_learning_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    print("Starting AL Handler ...")
    default_config = {
        "framework": "bmdal",
        "parameters": {
            "batch_size": 100,
            "selection_method": "lcmd",
            "base_kernel": "grad",
            "kernel_transforms": [("rp", [512])]
        }
    }
    default_config.update(data)
    assert isinstance(default_config['framework'], str) and \
        (default_config['framework'] in ['bmdal', 'mc_dropout', 'deep_ensemble'])
    if default_config['framework'] == 'bmdal':
        default_config['parameters'] = _bmdal_params_handler(default_config['parameters'])
    elif default_config['framework'] == 'mc_dropout':
        default_config['parameters'] = _mcdropout_params_handler(default_config['parameters'])
    elif default_config['framework'] == 'deep_ensemble':
        default_config['parameters'] = _deep_ensemble_params_handler(default_config['parameters'])
    else:
        raise ValueError(f"Unknown framework \"{default_config['framework']}\"")
    batch_selector = _BATCH_SELECTORS[default_config['framework']](**default_config['parameters'])
    return {
        'parameters': default_config,
        'batch_selector': batch_selector,
    }


__all__ = ['active_learning_handler']

from typing import Any

from src.utils.active_learning.batch_selectors import *
from .parser import *


_BATCH_SELECTORS = {
    'bmdal': BMDALBatchSelector,
    'mc_dropout': MCDropoutBatchSelector,
    'deep_ensemble': DeepEnsembleBatchSelector,
}


def _bmdal_params_handler(parameters: dict):
    assert isinstance(parameters['batch_size'], int)
    assert isinstance(parameters['max_batch_size'], int)
    assert isinstance(parameters['reload_initial_weights'], bool)
    # We distinguish two cases: the first is that of predefined algorithms, the second that of
    # full customization. Predefined algorithms are described in: https://arxiv.org/pdf/2203.09410
    # and are:
    # CoreSet: 'maxdist' + 'll' (no transformation) + sel_with_train=True
    # Badge: 'kmeanspp' + 'll' (no transformation) + sel_with_train=False
    # BALD: 'maxdiag' + 'll' + scale(X_train) + post(X_train, sigma^2) with sigma = 0.1
    # BatchBALD: 'maxdet' + 'll' + scale(X_train) + post(X_train, sigma^2) with sigma = 0.1
    # BAIT: 'bait' + 'll' + scale(X_train) + post(X_train, sigma^2)
    # Sketch LCMD: 'lcmd' + 'grad' + sketch[512]
    # (as shown as an example in the library: https://github.com/dholzmueller/bmdal_reg/blob/main)
    # For simplicity, we have assumed here that sigma = 0.1 for BALD and BatchBALD, since it is a configurable
    # parameter that should be estimated BEFORE the actual computation!
    # TODO: Verify if this assumption holds!
    method = None
    if 'standard_method' in parameters.keys():
        method = parameters.pop('standard_method', None)
        if method == 'coreset':
            parameters.update({
                'selection_method': 'maxdist',
                'initial_selection_method': 'maxdist',
                'base_kernel': 'll',
                'kernel_transforms': [],
                'sel_with_train': True
            })
        elif method == 'badge':
            parameters.update({
                'selection_method': 'kmeanspp',
                'initial_selection_method': 'kmeanspp',
                'base_kernel': 'll',
                'kernel_transforms': [],
                'sel_with_train': False
            })
        elif method == 'bald':
            sigma = parameters.get('sigma', 0.01) # sigma = 0.1 by default
            parameters.update({
                'selection_method': 'maxdiag',
                'initial_selection_method': 'maxdiag',
                'base_kernel': 'll',
                'kernel_transforms': [('train', [sigma])],
                'sel_with_train': False
            })
        elif method == 'batchbald':
            sigma = parameters.get('sigma', 0.01) # sigma = 0.1 by default
            parameters.update({
                'selection_method': 'maxdet',
                'initial_selection_method': 'maxdet',
                'base_kernel': 'll',
                'kernel_transforms': [('train', [sigma])],
                'sel_with_train': False
            })
        elif method == 'bait':
            sigma = parameters.get('sigma', 0.01) # sigma = 0.1 by default
            parameters.update({
                'selection_method': 'bait',
                'initial_selection_method': 'bait',
                'base_kernel': 'll',
                'kernel_transforms': [('train', [sigma])],
                'sel_with_train': False
            })
        elif method == 'lcmd_sketch_grad':
            parameters.update({
                'selection_method': 'lcmd',
                'initial_selection_method': 'lcmd',
                'base_kernel': 'grad',
                'kernel_transforms': [
                    ('rp', [512])
                ],
                'sel_with_train': True
            })
        elif method == 'random_sketch_grad':
            parameters.update({
                'selection_method': 'random',
                'initial_selection_method': 'random',
                'base_kernel': 'grad',
                'kernel_transforms': [
                    ('rp', [512])
                ],
                'sel_with_train': False
            })
        elif method == 'random_sketch_ll':
            parameters.update({
                'selection_method': 'random',
                'initial_selection_method': 'random',
                'base_kernel': 'll',
                'kernel_transforms': [
                    ('rp', [512])
                ],
                'sel_with_train': False
            })
        elif method is not None:
            raise ValueError(f"Unknown standard method: {method}")
    else:
        for key in ['selection_method', 'initial_selection_method']:
            assert isinstance(parameters[key], str) and (parameters[key] in
            ['random', 'maxdiag', 'maxdet', 'bait', 'fw', 'maxdist', 'kmeanspp', 'lcmd'])
            # We are temporarily ignoring the experimental options: 'fw-kernel', 'rmds' and 'sosd'
        assert isinstance(parameters['base_kernel'], str) and \
            (parameters['base_kernel'] in ['ll', 'grad', 'lin', 'nngp', 'ntk', 'laplace'])
        assert isinstance(parameters['kernel_transforms'], list)
        kernel_transforms = [
            tuple(item) if isinstance(item, list) else item for item in parameters['kernel_transforms']
        ]
        parameters['kernel_transforms'] = kernel_transforms
    return parameters, method


def _mcdropout_params_handler(parameters):
    return parameters


def _deep_ensemble_params_handler(parameters):
    return parameters


@ConfigParser.register_handler('active_learning')
def active_learning_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    default_config = {
        "framework": "bmdal",
        "parameters": {
            "batch_size": 128,
            "max_batch_size": 2048, # 16 iterations by default
            "reload_initial_weights": False,
            "selection_method": "lcmd",
            "sel_with_train": False,
            "base_kernel": "grad",
            "kernel_transforms": [("rp", [512])]
        }
    }
    default_config.update(data)
    al_method = None
    assert isinstance(default_config['framework'], str) and \
        (default_config['framework'] in ['bmdal', 'mc_dropout', 'deep_ensemble'])
    if default_config['framework'] == 'bmdal':
        default_config['parameters'], al_method = _bmdal_params_handler(default_config['parameters'])
    elif default_config['framework'] == 'mc_dropout':
        default_config['parameters'] = _mcdropout_params_handler(default_config['parameters'])
    elif default_config['framework'] == 'deep_ensemble':
        default_config['parameters'] = _deep_ensemble_params_handler(default_config['parameters'])
    else:
        raise ValueError(f"Unknown framework \"{default_config['framework']}\"")
    max_batch_size = default_config['parameters'].pop("max_batch_size", 2048)
    reload_initial_weights = default_config['parameters'].pop("reload_initial_weights", False)
    batch_selector = _BATCH_SELECTORS[default_config['framework']](**default_config['parameters'])
    if al_method is None:
        params = default_config['parameters']
        al_method = f'{params["selection_method"]} {params["initial_selection_method"]}'
    return {
        'parameters': default_config,
        'batch_selector': batch_selector,
        'batch_size': default_config['parameters']['batch_size'],
        'max_batch_size': max_batch_size,
        'reload_initial_weights': reload_initial_weights,
        'al_method': al_method
    }


__all__ = ['active_learning_handler']

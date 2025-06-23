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
        method = parameters.get('standard_method', None)
        if method == 'coreset':
            parameters.update({
                'selection_method': 'maxdist',
                'initial_selection_method': 'maxdist',
                'base_kernel': 'll',
                'kernel_transforms': [],
                'sel_with_train': True
            })
        elif method == 'badge':
            sigma = parameters.get('sigma', 0.01) # sigma = 0.1 by default
            parameters.update({
                'selection_method': 'kmeanspp',
                'initial_selection_method': 'kmeanspp',
                'base_kernel': 'll',
                'kernel_transforms': [('train', [sigma])],
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
        elif method in {'lcmd_sketch_grad', 'lcmd'}:
            parameters.update({
                'selection_method': 'lcmd',
                'initial_selection_method': 'lcmd',
                'base_kernel': 'grad',
                'kernel_transforms': [
                    ('rp', [512])
                ],
                'sel_with_train': True
            })
        elif method in {'random_sketch_grad', 'random'}:
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


@ConfigParser.standardizer('active_learning')
def active_learning_std(config: dict[str, Any], key: str):
    default_config = {
        "framework": "bmdal",
        "parameters": {
            "batch_size": 256,
            "max_batch_size": 1024, # 16 iterations by default
            "full_first_set": True,
            "first_set_size": 5120,
            "downsampling_factor": 0.5,
            "reload_initial_weights": False,
            "selection_method": "lcmd",
            "sel_with_train": False,
            "base_kernel": "grad",
            "kernel_transforms": [("rp", [512])]
        }
    }
    default_config.update(config[key])
    al_method = None
    framework = default_config['framework']
    if framework == 'bmdal':
        default_config['parameters'], al_method = _bmdal_params_handler(default_config['parameters'])
    elif framework in {'mc_dropout', 'deep_ensemble'}:
        raise NotImplementedError("Frameworks \"mc_dropout\" and \"deep_ensemble\" not implemented")
    else:
        raise ValueError(f"Unknown framework \"{framework}\"")
    for field, value in default_config['parameters'].items():
        default_config[field] = value
    default_config.pop('parameters')
    default_config["@extra"] = {
        'standard_method': al_method,
        'batch_size': default_config['batch_size'],
        'max_batch_size': default_config['max_batch_size'],
        'reload_initial_weights': default_config['reload_initial_weights'],
    }
    return default_config


@ConfigParser.processor('active_learning')
def active_learning_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    extra = data.pop("@extra", None)
    params = [
        "batch_size", "selection_method",
        "sel_with_train", "base_kernel", "kernel_transforms"
    ]
    batch_selector_params = {k: data[k] for k in params}
    max_batch_size = data["max_batch_size"]
    reload_initial_weights = data["reload_initial_weights"]
    framework = data['framework']
    batch_selector = _BATCH_SELECTORS[framework](**batch_selector_params)
    batch_selector_params['max_batch_size'] = max_batch_size
    batch_selector_params['reload_initial_weights'] = reload_initial_weights
    return {
        'batch_selector': batch_selector,
        'parameters': batch_selector_params,
        'al_method': extra.get('standard_method', None)
    }


__all__ = ['active_learning_std', 'active_learning_handler']

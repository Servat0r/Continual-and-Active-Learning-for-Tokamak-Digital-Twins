from typing import Any
from .parser import *
from ..utils import QUALIKIZ_HIGHPOW_OUTPUTS, QUALIKIZ_HIGHPOW_INPUTS


@ConfigParser.register_handler('dataset')
def dataset_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    default_config = {
        'input_columns': QUALIKIZ_HIGHPOW_INPUTS,
        'output_columns': QUALIKIZ_HIGHPOW_OUTPUTS,
        'input_size': 15,
        'output_size': 4,
        'simulator_type': 'qualikiz',
        'pow_type': 'highpow',
        'cluster_type': 'Ip_Pin_based',
        'dataset_type': 'not_null',
        'normalize_inputs': False,
        'normalize_outputs': False,
        'load_saved_final_data': False,
    }
    default_config.update(data)
    assert isinstance(default_config['input_columns'], list)
    assert isinstance(default_config['output_columns'], list)
    assert all([isinstance(item, str) for item in default_config['input_columns']])
    assert all([isinstance(item, str) for item in default_config['output_columns']])
    assert isinstance(default_config['input_size'], int) and default_config['input_size'] > 0
    assert isinstance(default_config['output_size'], int) and default_config['output_size'] > 0
    assert isinstance(default_config['simulator_type'], str) and default_config['simulator_type'] in ['qualikiz', 'tglf']
    assert default_config['pow_type'] in ['highpow', 'lowpow', 'mixed']
    assert default_config['cluster_type'] in ['Ip_Pin_based', 'tau_based', 'pca_based', 'wmhd_based', 'beta_based']
    assert default_config['dataset_type'] in ['not_null', 'complete']
    return default_config


__all__ = ['dataset_handler']
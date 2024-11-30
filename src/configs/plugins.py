from typing import Any
from .parser import *
from ..utils import ValidationStreamPlugin
from ..utils.plugins.early_stopping import ValidationEarlyStoppingPlugin


@ConfigParser.register_handler('early_stopping')
def early_stopping_handler(data: dict[str, Any], **kwargs):
    default_config = {
        'patience': 10,
        'metric': 'Loss',
        'delta': 0.1,
        'type': 'min',
        'restore_best_weights': True,
        'val_stream_name': 'test_stream',
        'when_above': None,
        'when_below': None,
    }
    default_config.update(data)
    return ValidationEarlyStoppingPlugin(**default_config)


#@ConfigParser.register_handler('validation_stream')
def validation_stream_handler(data: dict[str, Any], **kwargs):
    default_config = {
        'val_stream': 'test_stream'
    }
    default_config.update(data)
    val_stream = default_config['val_stream']
    plugin = ValidationStreamPlugin(eval(f"benchmark.{val_stream}"))
    return plugin


__all__ = ['early_stopping_handler', 'validation_stream_handler']
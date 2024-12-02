from typing import Any

from .parser import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


@ConfigParser.register_handler('scheduler')
def scheduler_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    metric = parameters.pop('metric', 'eval_loss')
    first_epoch_only = parameters.pop('first_epoch_only', False)
    first_exp_only = parameters.pop('first_exp_only', False)
    if (name == 'StepLR') or (name == 'step_lr'):
        return {
            'class': StepLR,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'parameters': parameters,
        }
    elif (name == 'ReduceLROnPlateau') or (name == 'reduce_lr_on_plateau'):
        return {
            'class': ReduceLROnPlateau,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'parameters': parameters,
        }
    elif (name == 'CosineAnnealingLR') or (name == 'cosine_annealing_lr'):
        return {
            'class': CosineAnnealingLR,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'parameters': parameters,
        }
    else:
        raise ValueError(f"Invalid scheduler name \"{name}\"")


__all__ = ['scheduler_handler']
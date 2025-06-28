from typing import Any

from .parser import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


@ConfigParser.standardizer('scheduler')
def scheduler_std(config: dict[str, Any], key: str):
    return config[key]


@ConfigParser.processor('scheduler')
def scheduler_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    metric = parameters.pop('metric', 'eval_loss')
    first_epoch_only = parameters.pop('first_epoch_only', False)
    first_exp_only = parameters.pop('first_exp_only', False)
    reset_lr = parameters.pop('reset_lr', True)
    if (name == 'StepLR') or (name == 'step_lr'):
        return {
            'class': StepLR,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'reset_lr': reset_lr,
            'parameters': parameters,
        }
    elif (name == 'ReduceLROnPlateau') or (name == 'reduce_lr_on_plateau'):
        return {
            'class': ReduceLROnPlateau,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'reset_lr': reset_lr,
            'parameters': parameters,
        }
    elif (name == 'CosineAnnealingLR') or (name == 'cosine_annealing_lr'):
        return {
            'class': CosineAnnealingLR,
            'metric': metric,
            'first_epoch_only': first_epoch_only,
            'first_exp_only': first_exp_only,
            'reset_lr': reset_lr,
            'parameters': parameters,
        }
    else:
        raise ValueError(f"Invalid scheduler name \"{name}\"")


__all__ = ['scheduler_std', 'scheduler_handler']
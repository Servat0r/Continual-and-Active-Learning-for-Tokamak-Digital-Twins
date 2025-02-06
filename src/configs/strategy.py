from typing import Any

import torch
from torch.optim import Adam
from avalanche.training import Naive, Replay, EWC, SynapticIntelligence, \
    GEM, MAS, Cumulative, JointTraining, FromScratchTraining, LFL
from avalanche.training.plugins import FromScratchTrainingPlugin

from .parser import *
from ..utils.strategies import *

__strategy_dict = {
    'Naive': Naive,
    'Cumulative': Cumulative,
    'JointTraining': JointTraining,
    'FromScratch': FromScratchTraining,
    'FromScratchTraining': FromScratchTraining,
    'Replay': Replay,
    'PercentageReplay': PercentageReplay,
    'EWC': EWC,
    'SI': SynapticIntelligence,
    'GEM': GEM,
    'MAS': MAS,
    'LFL': LFL,
    'EWCReplay': EWCReplay,
    'MASReplay': MASReplay,
    'GEMReplay': GEMReplay,
    'SIReplay': SIReplay,
}


@ConfigParser.register_handler('strategy')
def strategy_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    extra_log_folder = data.get('extra_log_folder', None)
    from_scratch = data.get('from_scratch', False)
    if name not in __strategy_dict:
        raise ValueError(f"Invalid strategy name \"{name}\"")
    strategy_class = __strategy_dict[name]
    return {
        'class': strategy_class,
        'parameters': parameters,
        'extra_log_folder': extra_log_folder,
        'from_scratch': from_scratch,
    }


__all__ = ['strategy_handler']
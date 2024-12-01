from typing import Any
from .parser import *
from avalanche.training import Naive, Replay, EWC, SynapticIntelligence, GenerativeReplay, \
    GEM, MAS, Cumulative, JointTraining, FromScratchTraining


__strategy_dict = {
    'Naive': Naive,
    'Cumulative': Cumulative,
    'JointTraining': JointTraining,
    'FromScratch': FromScratchTraining,
    'FromScratchTraining': FromScratchTraining,
    'Replay': Replay,
    'EWC': EWC,
    'SI': SynapticIntelligence,
    'GenerativeReplay': GenerativeReplay,
    'GEM': GEM,
    'MAS': MAS,
}


@ConfigParser.register_handler('strategy')
def strategy_handler(data: dict[str, Any], **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if name not in __strategy_dict:
        raise ValueError(f"Invalid strategy name \"{name}\"")
    strategy_class = __strategy_dict[name]
    return {
        'class': strategy_class,
        'parameters': parameters,
    }


__all__ = ['strategy_handler']
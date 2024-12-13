from typing import Any

import torch
from avalanche.models import MlpVAE
from torch.optim import Adam
from avalanche.training import Naive, Replay, EWC, SynapticIntelligence, GenerativeReplay, \
    GEM, MAS, Cumulative, JointTraining, FromScratchTraining

from .parser import *
from ..utils.strategies import *


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
    if name not in __strategy_dict:
        raise ValueError(f"Invalid strategy name \"{name}\"")
    strategy_class = __strategy_dict[name]
    # Handle the case of GenerativeReplay
    if strategy_class == GenerativeReplay:
        print("Handler in GenerativeReplay")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if "input_size" in kwargs:
            input_size = kwargs["input_size"]
        elif ("dataset" in kwargs) and ("input_size" in kwargs["dataset"]):
            input_size = kwargs["dataset"]["input_size"]
        else:
            input_size = 15 # highpow
        if "output_size" in kwargs:
            output_size = kwargs["output_size"]
        elif ("dataset" in kwargs) and ("output_size" in kwargs["dataset"]):
            output_size = kwargs["dataset"]["output_size"]
        else:
            output_size = 1 # 1 output
        nhid = parameters.pop("nhid", 2)
        print(f"Using VAE encoding space size of {nhid} and input size of {input_size}")
        generator = MlpVAE((1, input_size), nhid=nhid, n_classes=output_size, device=device) # nhid = 2 in GenerativeReplay code
        # optimzer:
        to_optimize = list(
            filter(lambda p: p.requires_grad, generator.parameters())
        )
        optimizer_generator = Adam(
            to_optimize,
            lr=1e-2,
            weight_decay=0.0001,
        )
        # strategy (with plugin):
        parameters["generator_strategy"] = {
            'model': generator,
            'optimizer': optimizer_generator,
        }
    return {
        'class': strategy_class,
        'parameters': parameters,
        'extra_log_folder': extra_log_folder,
    }


__all__ = ['strategy_handler']
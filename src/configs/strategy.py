from typing import Any

import torch
from avalanche.models import VAE_loss, MlpVAE
from avalanche.training.plugins import GenerativeReplayPlugin
from torch.optim import Adam
from avalanche.training import Naive, Replay, EWC, SynapticIntelligence, GenerativeReplay, \
    GEM, MAS, Cumulative, JointTraining, FromScratchTraining, VAETraining

from .parser import *


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
def strategy_handler(data: dict[str, Any], task_id: int = 0, **kwargs):
    if 'name' not in data:
        raise ValueError(f"\"name\" field not present in configuration")
    if 'parameters' not in data:
        raise ValueError(f"\"parameters\" field not present in configuration")
    name, parameters = data['name'], data['parameters']
    if name not in __strategy_dict:
        raise ValueError(f"Invalid strategy name \"{name}\"")
    strategy_class = __strategy_dict[name]
    # Handle the case of GenerativeReplay
    if strategy_class == GenerativeReplay:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if "input_size" in parameters:
            input_size = parameters["input_size"]
        elif ("dataset" in parameters) and ("input_size" in parameters["dataset"]):
            input_size = parameters["dataset"]["input_size"]
        else:
            input_size = 15 # highpow
        generator = MlpVAE((1, input_size), nhid=2, device=device) # nhid = 2 in GenerativeReplay code
        # optimzer:
        to_optimize = list(
            filter(lambda p: p.requires_grad, generator.parameters())
        )
        optimizer_generator = Adam(
            to_optimize,
            lr=1e-2,
            weight_decay=0.0001,
        )
        train_mb_size = parameters["train_mb_size"]
        train_epochs = parameters["train_epochs"]
        eval_mb_size = parameters["eval_mb_size"]
        replay_size = parameters.get("replay_size", None)
        increasing_replay_size = parameters.get("increasing_replay_size", False)
        is_weighted_replay = parameters.get("is_weighted_replay", False)
        weight_replay_loss_factor = parameters.get("weight_replay_loss_factor", 1.0)
        weight_replay_loss = parameters.get("weight_replay_loss", 1e-4)
        # strategy (with plugin):
        generator_strategy = VAETraining(
            model=generator,
            optimizer=optimizer_generator,
            criterion=VAE_loss,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=[
                GenerativeReplayPlugin(
                    replay_size=replay_size,
                    increasing_replay_size=increasing_replay_size,
                    is_weighted_replay=is_weighted_replay,
                    weight_replay_loss_factor=weight_replay_loss_factor,
                    weight_replay_loss=weight_replay_loss,
                )
            ],
        )
        parameters["generator_strategy"] = generator_strategy
    return {
        'class': strategy_class,
        'parameters': parameters,
    }


__all__ = ['strategy_handler']
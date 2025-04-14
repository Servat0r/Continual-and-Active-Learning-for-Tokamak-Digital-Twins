from typing import Optional, Union, List, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

from .constant_size_gem import *


class ConstantSizeGEMReplay(ConstantSizeGEM):
    """
    Simple combination of ConstantSizeGEM with a Replay plugin.
    """
    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            gem_mem_size: int = 2000,
            memory_strength: float = 0.5,
            replay_mem_size: int = 200,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = None,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin, Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            batch_size: Optional[int] = None,
            batch_size_mem: Optional[int] = None,
            task_balanced_dataloader: bool = False,
            storage_policy: Optional["ExemplarsBuffer"] = None,
            **base_kwargs
    ):
        replay_plugin = ReplayPlugin(
            mem_size=replay_mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=task_balanced_dataloader,
            storage_policy=storage_policy,
        )
        plugins = plugins if plugins is not None else []
        plugins.append(replay_plugin)
        super().__init__(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=gem_mem_size, memory_strength=memory_strength,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=evaluator,
            eval_every=eval_every, plugins=plugins, **base_kwargs,
        )


__all__ = ['ConstantSizeGEMReplay']
from typing import Callable, Optional, List, Union
import torch

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import (
    default_evaluator,
)
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

from .plugins import ConstantSizeGEMPlugin


class ConstantSizeGEM(SupervisedTemplate):

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        mem_size: int = 2000,
        memory_strength: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        gem = ConstantSizeGEMPlugin(mem_size, memory_strength)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


__all__ = ['ConstantSizeGEM']
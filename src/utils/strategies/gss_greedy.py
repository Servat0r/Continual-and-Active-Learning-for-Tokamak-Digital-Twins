from typing import Optional, List, Union, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

from .plugins.gss_greedy import RegressionGSS_greedyPlugin


class RegressionGSS_greedy(SupervisedTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        mem_size: int = 200,
        mem_strength: int = 1,
        update_every: int = 1,
        input_size: list = [],
        output_size: list = [],
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
        gss = RegressionGSS_greedyPlugin(
            mem_size=mem_size, mem_strength=mem_strength, input_size=input_size,
            output_size=output_size, update_every=update_every
        )
        if plugins is None:
            plugins = [gss]
        else:
            plugins.append(gss)

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


__all__ = ['RegressionGSS_greedy']

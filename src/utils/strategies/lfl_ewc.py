from typing import Callable, Optional, List, Union
import torch

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import (
    default_evaluator,
)
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
    EWCPlugin, LFLPlugin
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


class LFLEWC(SupervisedTemplate):
    """
    Variant of EWC allowing dynamic lambdas.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        ewc_lambda: float | list[float],
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        lambda_e: float = 1.0,
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
        lfl = LFLPlugin(lambda_e)
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [lfl, ewc]
        else:
            plugins.append(lfl)
            plugins.append(ewc)

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


__all__ = ['LFLEWC']

from typing import Callable, Optional, Sequence, List, Union
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

from .plugins import DoubleLFLPlugin


class DoubleLFL(SupervisedTemplate):
    """
    A variant of Avalanche LFL strategy.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        lambda_e: Union[float, Sequence[float]],
        evaluation_metric: str = 'R2Score_Exp',
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
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param lambda_e: euclidean loss hyper parameter. It can be either a
                float number or a list containing lambda_e for each experience.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        lfl = DoubleLFLPlugin(lambda_e, evaluation_metric)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)
        self.double_lfl_plugin = lfl

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

    def set_eval_stream(self, eval_stream):
        self.double_lfl_plugin.set_eval_stream(eval_stream)


__all__ = ['DoubleLFL']

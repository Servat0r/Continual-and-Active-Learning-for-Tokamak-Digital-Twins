from typing import Optional, List, Union, Callable
import torch
from torch.nn import Module
from torch.optim import Optimizer
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator

from .plugins import PercentageReplayPlugin


class PercentageReplay(SupervisedTemplate):
    """Percentage Replay Strategy.

    This strategy uses the PercentageReplayPlugin, which dynamically adjusts
    the buffer size to maintain a fixed percentage of the total training data.
    Task identities are not used in this strategy.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion,
        mem_percentage: float = 0.1,
        min_buffer_size: int = 0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        pr_plugin_kwargs: dict = {},
        **base_kwargs
    ):
        """
        Initialize the PercentageReplay strategy.

        :param model: The model to be trained.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_percentage: The percentage of total training data to
            maintain in the buffer. Defaults to 0.1 (10%).
        :param train_mb_size: The minibatch size for training. Defaults to 1.
        :param train_epochs: The number of epochs for training. Defaults to 1.
        :param eval_mb_size: The minibatch size for evaluation. Defaults to None.
        :param device: The device to use for training and evaluation.
        :param plugins: Additional plugins to use. Defaults to None.
        :param evaluator: The evaluation plugin to use for logging and metrics.
        :param eval_every: The frequency of evaluation during training.
            Defaults to -1 (no intermediate evaluation).
        :param pr_plugin_kwargs: Additional arguments for the PercentageReplayPlugin.
        :param **base_kwargs: Additional arguments for the base template.
        """
        prp = PercentageReplayPlugin(
            mem_percentage=mem_percentage, min_buffer_size=min_buffer_size, **pr_plugin_kwargs
        )
        if plugins is None:
            plugins = [prp]
        else:
            plugins.append(prp)

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


__all__ = ["PercentageReplay"]

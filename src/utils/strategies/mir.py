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
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.metrics import loss_metrics
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.models.expert_gate import AE_loss
from avalanche.logging import InteractiveLogger
from avalanche.training.templates.strategy_mixin_protocol import CriterionType

from .plugins.mir import RegressionMIRPlugin


class RegressionMIR(SupervisedTemplate):
    """Maximally Interfered Replay Strategy
    See ER_MIR plugin for details.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        mem_size: int,
        subsample: int,
        batch_size_mem: int = 1,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
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
        :param mem_size: Amount of fixed memory to use
        :param subsample: Size of the initial sample
                from which to select the replay batch
        :param batch_size_mem: Size of the replay batch after
                loss-based selection
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
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        # Instantiate plugin
        mir = RegressionMIRPlugin(
            mem_size=mem_size,
            subsample=subsample,
            batch_size_mem=batch_size_mem,
        )

        # Add plugin to the strategy
        if plugins is None:
            plugins = [mir]
        else:
            plugins.append(mir)

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


__all__ = ['RegressionMIR']

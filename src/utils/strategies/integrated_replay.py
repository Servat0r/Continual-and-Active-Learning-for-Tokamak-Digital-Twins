from typing import Any

import torch
from avalanche.models import MlpVAE
from torch.optim import Adam
from avalanche.training import Naive, Replay, EWC, SynapticIntelligence, GenerativeReplay, \
    GEM, MAS, Cumulative, JointTraining, FromScratchTraining
from avalanche.training.plugins import ReplayPlugin


class EWCReplay(EWC):

    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            ewc_lambda: float,
            mode: str = "separate",
            decay_factor: Optional[float] = None,
            keep_importance_data: bool = False,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = None,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin, Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            mem_size: int = 200,
            batch_size: Optional[int] = None,
            batch_size_mem: Optional[int] = None,
            task_balanced_dataloader: bool = False,
            storage_policy: Optional["ExemplarsBuffer"] = None,
            **base_kwargs
    ):
        replay_plugin = ReplayPlugin(
            mem_size=mem_size,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=task_balanced_dataloader,
            storage_policy=storage_policy,
        )
        plugins = plugins if plugins is not None else []
        plugins.append(replay_plugin)
        super().__init__(
            model, optimizer, criterion, ewc_lambda, mode, decay_factor,
            keep_importance_data, train_mb_size, train_epochs, eval_mb_size,
            device, plugins, evaluator, eval_every, **base_kwargs
        )


__all__ = ['EWCReplay']

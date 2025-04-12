from typing import Optional, Union, List, Callable, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.training import EWC, SynapticIntelligence, GEM, MAS, LFL
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


class EWCReplay(EWC):
    """
    Simple combination of Elastic Weight Consolidation with a Replay plugin.
    """
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
        super(EWCReplay, self).__init__(
            model=model, optimizer=optimizer, criterion=criterion, ewc_lambda=ewc_lambda,
            mode=mode, decay_factor=decay_factor, keep_importance_data=keep_importance_data,
            train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size,
            device=device, plugins=plugins, evaluator=evaluator, eval_every=eval_every, **base_kwargs
        )


class MASReplay(MAS):
    """
    Simple combination of MAS with a Replay plugin.
    """
    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            lambda_reg: float = 1.0,
            alpha: float = 0.5,
            verbose: bool = False,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
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
            model=model, optimizer=optimizer, criterion=criterion,
            lambda_reg=lambda_reg, alpha=alpha, verbose=verbose,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=evaluator,
            eval_every=eval_every, plugins=plugins, **base_kwargs,
        )


class GEMReplay(GEM):
    """
    Simple combination of GEM with a Replay plugin.
    """
    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            patterns_per_exp: int,
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
            model=model, optimizer=optimizer, criterion=criterion,
            patterns_per_exp=patterns_per_exp, memory_strength=memory_strength,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=evaluator,
            eval_every=eval_every, plugins=plugins, **base_kwargs,
        )


class SIReplay(SynapticIntelligence):
    """
    Simple combination of Synaptic Intelligence with a Replay plugin.
    """
    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            si_lambda: Union[float, Sequence[float]],
            eps: float = 0.0000001,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
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
            model=model, optimizer=optimizer, criterion=criterion, si_lambda=si_lambda,
            eps=eps, train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=evaluator,
            eval_every=eval_every, plugins=plugins, **base_kwargs,
        )


class LFLReplay(LFL):
    """
    Simple combination of LFL with a Replay plugin.
    """
    def __init__(
            self,
            *,
            model: Module,
            optimizer: Optimizer,
            criterion: CriterionType,
            lambda_e: Union[float, Sequence[float]],
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
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
            model=model, optimizer=optimizer, criterion=criterion, lambda_e=lambda_e,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, evaluator=evaluator,
            eval_every=eval_every, plugins=plugins, **base_kwargs,
        )


__all__ = ['EWCReplay', 'MASReplay', 'GEMReplay', 'SIReplay', 'LFLReplay']

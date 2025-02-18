from typing import Callable, Optional, Sequence, List, Union, TYPE_CHECKING, Type
from abc import abstractmethod
import torch

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training import Replay, GEM, MAS, EWC, Cumulative, Naive, FromScratchTraining

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from .al_plugins import *
from ...strategies import PercentageReplay


def al_cl_strategy_converter(cl_strategy_class: Type[SupervisedTemplate]) -> Type[SupervisedTemplate]:
    _dict = {
        Naive: ALNaive,
        FromScratchTraining: ALFromScratchTraining, # TODO: Actually not!
        Replay: ALReplay,
        PercentageReplay: ALPercentageReplay,
        GEM: ALGEM,
        MAS: ALMAS,
        EWC: ALEWC,
        Cumulative: ALCumulative
    }
    converted_class = _dict.get(cl_strategy_class, None)
    if converted_class is None:
        raise NotImplementedError(f"Class {cl_strategy_class.__name__} is not implemented by now!")
    else:
        return converted_class


class ALCLTemplate(SupervisedTemplate):
    """Base class for Active Learning CL strategies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_in_active_learning = False # By default, we are NOT in an Active Learning cycle

    def start_active_learning_cycle(self):
        self._is_in_active_learning = True
        for p in self.plugins:
            if isinstance(p, ContinualActiveLearningPlugin):
                p.start_active_learning_cycle()

    def stop_active_learning_cycle(self):
        self._is_in_active_learning = False
        for p in self.plugins:
            if isinstance(p, ContinualActiveLearningPlugin):
                p.stop_active_learning_cycle()
    
    def is_in_active_learning(self) -> bool:
        return self._is_in_active_learning
    
    @classmethod
    @abstractmethod
    def base_cl_class(cls):
        pass


class ALNaive(ALCLTemplate):

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
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
    
    @classmethod
    def base_cl_class(cls):
        return Naive


class ALFromScratchTraining(ALCLTemplate):

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        reset_optimizer: bool = True,
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
        fstp = ALFromScratchTrainingPlugin(reset_optimizer=reset_optimizer)
        if plugins is None:
            plugins = [fstp]
        else:
            plugins.append(fstp)
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
    
    @classmethod
    def base_cl_class(cls):
        return FromScratchTraining


class ALReplay(ALCLTemplate):
    """Replay strategy with Active Learning capabilities."""
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        mem_size: int = 200,
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
        rp = ALReplayPlugin(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
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
    
    @classmethod
    def base_cl_class(cls):
        return Replay


class ALPercentageReplay(ALCLTemplate):
    """PercentageReplay strategy with Active Learning capabilities."""
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
        prp = ALPercentageReplayPlugin(
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
    
    @classmethod
    def base_cl_class(cls):
        return PercentageReplay


class ALGEM(ALCLTemplate):
    """GEM strategy with Active Learning capabilities."""
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
        **base_kwargs
    ):
        gem = ALGEMPlugin(patterns_per_exp, memory_strength)
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
    
    @classmethod
    def base_cl_class(cls):
        return GEM


class ALMAS(ALCLTemplate):
    """MAS strategy with Active Learning capabilities."""
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
        **base_kwargs
    ):

        # Instantiate plugin
        mas = ALMASPlugin(lambda_reg=lambda_reg, alpha=alpha, verbose=verbose)

        # Add plugin to the strategy
        if plugins is None:
            plugins = [mas]
        else:
            plugins.append(mas)

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
    
    @classmethod
    def base_cl_class(cls):
        return MAS


class ALEWC(ALCLTemplate):

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
        **base_kwargs
    ):
        ewc = ALEWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
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
    
    @classmethod
    def base_cl_class(cls):
        return EWC


class ALCumulative(ALCLTemplate):

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **kwargs
    ):
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
            **kwargs
        )
        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        exp = self.experience
        assert exp is not None
        if self.dataset is None:
            self.dataset = exp.dataset
        elif not self.is_in_active_learning():
            self.dataset = concat_datasets([self.dataset, exp.dataset])
        self.adapted_dataset = self.dataset
    
    @classmethod
    def base_cl_class(cls):
        return Cumulative


__all__ = [
    'al_cl_strategy_converter',
    'ALCLTemplate',
    'ALNaive',
    'ALFromScratchTraining',
    'ALReplay',
    'ALPercentageReplay',
    'ALGEM',
    'ALMAS',
    'ALEWC',
    'ALCumulative'
]

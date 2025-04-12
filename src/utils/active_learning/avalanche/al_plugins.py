# Active Learning - aware Avalanche Plugins
from typing import Any, Optional, TYPE_CHECKING, TextIO
from abc import ABC
import sys
import torch
from avalanche.core import Template

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models import avalanche_forward
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, GEMPlugin, MASPlugin, \
    FromScratchTrainingPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, ParamData
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from ...strategies.plugins import PercentageReplayPlugin


# TODO: Add a Stopping Criterion? (Object that tells when to stop the AL cycle)
class ContinualActiveLearningPlugin(SupervisedPlugin, supports_distributed=True):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_active_learning = False # Within an Active Learning cycle
        self._train_exp_counter = 0
    
    def start_active_learning_cycle(self):
        self._in_active_learning = True
    
    def stop_active_learning_cycle(self):
        self._in_active_learning = False
    
    def is_in_active_learning(self) -> bool:
        return self._in_active_learning
    
    def get_train_exp_counter(self) -> int:
        return self._train_exp_counter
    
    def before_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        super().before_training_exp(strategy, *args, **kwargs)
    
    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        super().after_training_exp(strategy, *args, **kwargs)
        if not self.is_in_active_learning():
            self._train_exp_counter += 1


# Actual Subclasses of Avalanche plugins
class ALReplayPlugin(ContinualActiveLearningPlugin, ReplayPlugin):
    """
    Extended ReplayPlugin that integrates Active Learning awareness
    """
    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__(
            mem_size, batch_size, batch_size_mem, task_balanced_dataloader, storage_policy
        )
    
    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        super().before_training_exp(strategy, num_workers, shuffle, drop_last, **kwargs)
    
    def after_training_exp(self, strategy: Any, *args, **kwargs) -> Any:
        if not self.is_in_active_learning():
            super().after_training_exp(strategy, *args, **kwargs)


class ALGEMPlugin(ContinualActiveLearningPlugin, GEMPlugin):

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        super().__init__(patterns_per_experience, memory_strength)
        self._train_exp_counter = 0
    
    def before_training_iteration(self, strategy, **kwargs):
        """
        Adaptation of GEMPlugin.before_training_iteration()
        """
        if self._train_exp_counter > 0:
            G = []
            strategy.model.train()
            for t in range(self._train_exp_counter):
                strategy.model.train()
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
                loss = strategy._criterion(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten()
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=strategy.device)
                            )
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )
            
            self.G = torch.stack(G)  # (experiences, parameters)
    
    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if self._train_exp_counter > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            #stdout_debug_print(f"self.G = {self.G}", color='red')
            if self.G.device != g.device:
                self.G = self.G.to(g.device)

            #stdout_debug_print(f"self.G = {self.G}", color='red')
            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"
    
    def after_training_exp(self, strategy, **kwargs):
        if not self.is_in_active_learning():
            self.update_memory(
                strategy.experience.dataset,
                self._train_exp_counter,
                strategy.train_mb_size,
            )
        super().after_training_exp(strategy, **kwargs)


class ALMASPlugin(ContinualActiveLearningPlugin, MASPlugin):

    def __init__(self, lambda_reg: float = 1.0, alpha: float = 0.5, verbose=False):
        super().__init__(lambda_reg, alpha, verbose)
        self._train_exp_counter = 0
    
    def before_backward(self, strategy, **kwargs):
        # Check if the task is not the first
        exp_counter = self._train_exp_counter

        if exp_counter == 0:
            return

        loss_reg = 0.0

        # Check if properties have been initialized
        if not self.importance:
            raise ValueError("Importance is not available")
        if not self.params:
            raise ValueError("Parameters are not available")
        if not strategy.loss:
            raise ValueError("Loss is not available")

        # Apply penalty term
        for name, param in strategy.model.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(
                    self.importance[name].expand(param.shape)
                    * (param - self.params[name].expand(param.shape)).pow(2)
                )

        # Update loss
        strategy.loss += self._lambda * loss_reg
    
    def after_training_exp(self, strategy, **kwargs):
        if not self.is_in_active_learning():
            self.params = dict(copy_params_dict(strategy.model))

            # Get importance
            exp_counter = self._train_exp_counter
            if exp_counter == 0:
                self.importance = self._get_importance(strategy)
                return
            else:
                curr_importance = self._get_importance(strategy)

            # Check if previous importance is available
            if not self.importance:
                raise ValueError("Importance is not available")

            # Update importance
            for name in curr_importance.keys():
                new_shape = curr_importance[name].data.shape
                if name not in self.importance:
                    self.importance[name] = ParamData(
                        name,
                        curr_importance[name].shape,
                        device=curr_importance[name].device,
                        init_tensor=curr_importance[name].data.clone(),
                    )
                else:
                    self.importance[name].data = (
                        self.alpha * self.importance[name].expand(new_shape)
                        + (1 - self.alpha) * curr_importance[name].data
                    )
        super().after_training_exp(strategy, **kwargs)


class ALEWCPlugin(ContinualActiveLearningPlugin, EWCPlugin):

    def __init__(
        self,
        ewc_lambda,
        mode="separate",
        decay_factor=None,
        keep_importance_data=False,
    ):
        super().__init__(ewc_lambda, mode, decay_factor, keep_importance_data)
        self._train_exp_counter = 0
    
    def before_backward(self, strategy, **kwargs):
        exp_counter = self._train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == "separate":
            for experience in range(exp_counter):
                for k, cur_param in strategy.model.named_parameters():
                    # new parameters do not count
                    if k not in self.saved_params[experience]:
                        continue
                    saved_param = self.saved_params[experience][k]
                    imp = self.importances[experience][k]
                    new_shape = cur_param.shape
                    penalty += (
                        imp.expand(new_shape)
                        * (cur_param - saved_param.expand(new_shape)).pow(2)
                    ).sum()
        elif self.mode == "online":  # may need importance and param expansion
            prev_exp = exp_counter - 1
            for k, cur_param in strategy.model.named_parameters():
                # new parameters do not count
                if k not in self.saved_params[prev_exp]:
                    continue
                saved_param = self.saved_params[prev_exp][k]
                imp = self.importances[prev_exp][k]
                new_shape = cur_param.shape
                penalty += (
                    imp.expand(new_shape)
                    * (cur_param - saved_param.expand(new_shape)).pow(2)
                ).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty

    def after_training_exp(self, strategy, **kwargs):
        if not self.is_in_active_learning():
            exp_counter = self._train_exp_counter
            importances = self.compute_importances(
                strategy.model,
                strategy._criterion,
                strategy.optimizer,
                strategy.experience.dataset,
                strategy.device,
                strategy.train_mb_size,
                num_workers=kwargs.get("num_workers", 0),
            )
            self.update_importances(importances, exp_counter)
            self.saved_params[exp_counter] = copy_params_dict(strategy.model)
            # clear previous parameter values
            if exp_counter > 0 and (not self.keep_importance_data):
                del self.saved_params[exp_counter - 1]
        super().after_training_exp(strategy, **kwargs)


class ALFromScratchTrainingPlugin(ContinualActiveLearningPlugin, FromScratchTrainingPlugin):

    def __init__(self, reset_optimizer: bool = True):
        super().__init__(reset_optimizer)


class ALPercentageReplayPlugin(ContinualActiveLearningPlugin, PercentageReplayPlugin):

    def __init__(
        self,
        mem_percentage: float = 0.1,  # Percentage of total training set for the buffer
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        dump: bool = False, dump_fp: TextIO | str = sys.stdout,
        min_buffer_size: int = 0,
    ):
        super().__init__(
            mem_percentage, batch_size, batch_size_mem, task_balanced_dataloader,
            storage_policy, dump, dump_fp, min_buffer_size
        )
    
    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        return super().before_training_exp(strategy, num_workers, shuffle, drop_last, **kwargs)
    
    def after_eval_exp(self, strategy: Any, *args, **kwargs) -> Any:
        if not self.is_in_active_learning():
            return super().after_eval_exp(strategy, *args, **kwargs)


__all__ = [
    'ContinualActiveLearningPlugin',
    'ALReplayPlugin',
    'ALGEMPlugin',
    'ALMASPlugin',
    'ALEWCPlugin',
    'ALFromScratchTrainingPlugin',
    'ALPercentageReplayPlugin'
]

import os
import numpy as np
import torch
import copy
from typing import Any

from avalanche.core import Template
from avalanche.logging import BaseLogger

from .. import debug_print
from .utils import ValidationStreamPlugin


class ValidationEarlyStoppingPlugin(BaseLogger):

    @property
    def MIN(self):
        return 'min'

    @property
    def MAX(self):
        return 'max'

    def __initialize(self):
        self.best_metric = None
        self.wait = 0
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_weights = None
        self.inside_training = False
        self.validation_plugin = None
    
    def __init__(
            self, patience=3, delta=0.01, metric='Loss', type='min',
            restore_best_weights=True, val_stream_name='test_stream',
            when_above=float('-inf'), when_below=float('inf'), min_epochs=0,
            use_validation_plugin=True, immediate_stop_delta=None,
            debug_log_file=None,
    ):
        super().__init__()
        self.patience = patience
        self.delta = delta
        if metric.endswith('_Exp'):
            metric = metric.replace('_Exp', '_Epoch')
        self.metric = metric if metric.endswith('_Epoch') else metric + '_Epoch'
        self.type = type
        self.keep_best_weights = restore_best_weights
        self.val_stream_name = val_stream_name
        self.when_above = when_above
        self.when_below = when_below
        self.min_epochs = min_epochs
        self.use_validation_plugin = use_validation_plugin
        self.immediate_stop_delta = immediate_stop_delta
        if debug_log_file is not None:
            self.debug_log_file = open(debug_log_file, 'w')
        else:
            self.debug_log_file = os.devnull
        self.__initialize()

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = True

    def after_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = False

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = True

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.__initialize()

    def update(self, strategy, current_metric, **kwargs):
        if current_metric is not None:
            if self.best_metric is None:
                # First time setup: everything is new 'best'
                self.best_metric = current_metric
                if self.keep_best_weights:
                    self.best_weights = copy.deepcopy(strategy.model.state_dict())
                self.best_epoch = self.current_epoch
                self.wait = 0
            else:
                # Reset `wait` only if improvement >= delta
                if self.type == self.MAX:
                    if current_metric > (self.best_metric + self.delta):
                        self.wait = 0
                    else:
                        self.wait += 1
                else:  # self.type == self.MIN
                    if current_metric < (self.best_metric - self.delta):
                        self.wait = 0
                    else:
                        self.wait += 1

                # ANY improvement means store best_weights + update best_metric
                if (self.type == self.MAX and current_metric > self.best_metric) or \
                        (self.type == self.MIN and current_metric < self.best_metric):
                    self.best_metric = current_metric
                    if self.keep_best_weights:
                        self.best_weights = copy.deepcopy(strategy.model.state_dict())
                    self.best_epoch = self.current_epoch
                    if self.debug_log_file is not None:
                        print(
                            f"[{type(self).__name__}]: New best metric {self.best_metric} at epoch {self.best_epoch} (wait = {self.wait})",
                            file=self.debug_log_file, flush=True
                        )
                        
            # Compute conditions for immediate stop (e.g. if loss diverges)
            immediate_stop = False
            if current_metric in [torch.nan, np.nan]:
                immediate_stop = True
            if (self.best_metric is not None) and (self.immediate_stop_delta is not None):
                if (self.type == self.MAX) and (self.best_metric > current_metric + self.immediate_stop_delta):
                        immediate_stop = True
                elif (self.type == self.MIN) and (self.best_metric < current_metric - self.immediate_stop_delta):
                        immediate_stop = True
            if immediate_stop:
                print(
                    f"[{type(self).__name__}] [{self.current_epoch}]: Immediate stop triggered at epoch {self.current_epoch}",
                    file=self.debug_log_file, flush=True
                )

            if ((self.current_epoch > self.min_epochs) and
                    (current_metric > self.when_above) and
                    (current_metric < self.when_below) and
                    ((self.wait >= self.patience) or immediate_stop)
            ):
                debug_print(
                    f"[{type(self).__name__}]: Early stopping triggered after epoch {self.current_epoch}."
                )
                strategy.stop_training()
                if self.keep_best_weights:
                    self.restore_best_weights(strategy)
        else:
            debug_print(
                f"[{type(self).__name__}]: No {self.metric} metric found after epoch {self.current_epoch}."
            )
        self.current_epoch += 1

    def before_training_epoch(self, strategy, **kwargs):
        if self.use_validation_plugin:
            if self.inside_training:
                current_metric = None
                if self.validation_plugin is None:
                    for plugin in strategy.plugins:
                        if isinstance(plugin, ValidationStreamPlugin):
                            self.validation_plugin = plugin
                if self.validation_plugin is not None:
                    last_metrics = self.validation_plugin.last_results
                    if last_metrics is not None:
                        for metric in last_metrics:
                            if metric.startswith(self.metric):
                                current_metric = last_metrics.get(metric, None)
                                break
                self.update(strategy, current_metric, **kwargs)

    def restore_best_weights(self, strategy):
        strategy.model.load_state_dict(self.best_weights)
        print(
            f"[{type(self).__name__}] Restored best weights of epoch {self.best_epoch}",
            file=self.debug_log_file, flush=True
        )


__all__ = ['ValidationEarlyStoppingPlugin']
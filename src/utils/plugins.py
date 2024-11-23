from typing import Any
from tqdm import tqdm

from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin
from avalanche.logging import BaseLogger
from .misc import debug_print


class ValidationStreamPlugin(SupervisedPlugin):

    def __init__(self, val_stream: Any):
        super().__init__()
        self.training_exp_id = None
        self.val_stream = None
        self.val_experiences = None
        self.set_validation_stream(val_stream)
        self.last_results = None

    def set_validation_stream(self, val_stream: Any):
        self.val_stream = val_stream
        self.val_experiences = []
        for val_exp in val_stream:
            self.val_experiences.append(val_exp)

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.training_exp_id = strategy.experience.current_experience

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        val_exp = self.val_experiences[self.training_exp_id]
        self.last_results = strategy.eval(val_exp)
        return self.last_results


class TqdmTrainingEpochsPlugin(SupervisedPlugin):

    def __init__(self, num_exp: int, num_epochs: int):
        super().__init__()
        self.num_exp = num_exp
        self.num_epochs = num_epochs
        self.tqdm = None

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm = tqdm(total=self.num_epochs, desc=f'Training exp {self.num_exp}')

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm.update(1)

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.tqdm.close()


class ValidationEarlyStoppingPlugin(BaseLogger):

    @property
    def MIN(self):
        return 'min'

    @property
    def MAX(self):
        return 'max'

    def __init__(
            self, patience=3, delta=0.01, metric='Loss', type='min',
            restore_best_weights=True, val_stream_name='test_stream',
            when_above=None, when_below=None,
    ):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.best_metric = None
        self.wait = 0
        self.metric = metric if metric.endswith('_Exp') else metric + '_Exp'
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.type = type
        self.keep_best_weights = restore_best_weights
        self.best_weights = None
        self.inside_training = False
        self.val_stream_name = val_stream_name
        self.validation_plugin = None
        self.when_above = when_above
        self.when_below = when_below

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = True

    def after_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = False

    def before_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.inside_training = True

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.current_epoch = 0
        self.best_metric = None
        self.wait = 0
        self.inside_training = False

    def before_training_epoch(self, strategy, **kwargs):
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
            if current_metric is not None:
                #debug_print(
                #    f"[{type(self).__name__}] {self.metric} metric after epoch {self.current_epoch}: {current_metric:.4f}"
                #)

                if self.type == self.MAX:
                    if self.best_metric is None or current_metric > self.best_metric + self.delta:
                        self.best_metric = current_metric
                        self.wait = 0
                        if self.keep_best_weights:
                            self.best_weights = strategy.model.state_dict().copy() # Is it necessary to copy?
                    else:
                        self.wait += 1
                elif self.type == self.MIN:
                    if self.best_metric is None or current_metric < self.best_metric + self.delta:
                        self.best_metric = current_metric
                        self.wait = 0
                        if self.keep_best_weights:
                            self.best_weights = strategy.model.state_dict().copy() # Is it necessary to copy?
                    else:
                        self.wait += 1

                if (self.wait >= self.patience and
                    (self.when_above is None or current_metric > self.when_above) and
                    (self.when_below is None or current_metric < self.when_below)):
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

    def restore_best_weights(self, strategy):
        strategy.model.load_state_dict(self.best_weights)


__all__ = [
    'ValidationStreamPlugin',
    'TqdmTrainingEpochsPlugin',
    'ValidationEarlyStoppingPlugin'
]
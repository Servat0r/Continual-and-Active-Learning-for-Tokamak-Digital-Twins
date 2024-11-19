from typing import Any

from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin
from .misc import debug_print


class EarlyStoppingPlugin(SupervisedPlugin):

    @property
    def MIN(self):
        return 'min'

    @property
    def MAX(self):
        return 'max'

    def __init__(self, patience=3, delta=0.01, metric='Loss', type='min', restore_best_weights=True):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.best_metric = None
        self.wait = 0
        self.metric = metric if metric.endswith('_Epoch') else metric + '_Epoch'
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.type = type
        self.keep_best_weights = restore_best_weights
        self.best_weights = None

    def after_training_epoch(self, strategy, **kwargs):
        """
        Hook executed after each training experience (cycle).
        """
        # Assuming validation accuracy is the tracked metric
        debug_print([(k, v) for k, v in strategy.plugins[1].get_all_metrics().items()])
        current_metric = strategy.evaluator.get_last_metrics().get(f"{self.metric}/train_phase/train_stream", None)
        if current_metric is None:
            raise ValueError(
                f"{type(self).__name__} object did not find {self.metric} "
                f"within evaluator-registered metrics!"
            )

        debug_print(
            f"[{type(self).__name__}]: {self.metric} metric after epoch {self.current_epoch}: {current_metric:.4f}"
        )

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

        if self.wait >= self.patience:
            debug_print(
                f"[{type(self).__name__}]: Early stopping triggered after epoch {self.current_epoch}."
            )
            strategy.stop_training = True
        self.current_epoch += 1

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        self.current_epoch = 0
        self.best_metric = None
        self.wait = 0

    def restore_best_weights(self, strategy):
        strategy.model.load_state_dict(self.best_weights)


__all__ = ['EarlyStoppingPlugin']
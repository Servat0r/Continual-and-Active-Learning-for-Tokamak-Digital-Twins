from typing import Callable

import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean


class FunctionalMetric(Metric):

    def __init__(self, metric: Callable):
        super().__init__()
        self.metric = metric
        self._mean_metric_value = Mean()

    def reset(self) -> None:
        """Resets the metric to its initial state."""
        self._mean_metric_value.reset()

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
        """
        Updates the metric with the predicted and actual vectors.

        :param predicted: Predicted vector (batch of predictions)
        :param actual: Actual (ground truth) vector (batch of targets)
        """
        acc = self.metric(actual, predicted)
        self._mean_metric_value.update(acc.mean().item())

    def result(self) -> float:
        return self._mean_metric_value.result()

    def __str__(self):
        return type(self).__name__


class FunctionalMetricPlugin(GenericPluginMetric):

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

    def __str__(self):
        base = type(self).__name__
        if base.startswith('Minibatch'):
            base = base[9:]
        elif base.startswith('Epoch'):
            base = base[5:]
        elif base.startswith('RunningEpoch'):
            base = base[12:]
        elif base.startswith('Experience'):
            base = base[10:]
        elif base.startswith('Stream'):
            base = base[6:]
        return f"{base}_{self.suffix_type}"


__all__ = ['FunctionalMetric', 'FunctionalMetricPlugin']
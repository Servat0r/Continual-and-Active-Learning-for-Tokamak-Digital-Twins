import torch
import numpy as np
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean

_C = (0.5 * np.log(2 * np.pi)).item()

def gaussian_nll_batch(y_true, y_pred, batch_dim=0):
    num_features = y_pred.shape[1] // 2
    # Split predictions into mean and log_variance
    mean = y_pred[:, :num_features]
    variance = y_pred[:, num_features:]
    log_variance_term = torch.log(variance)
    mse_term = torch.sum((y_true - mean)**2, dim=batch_dim+1) / variance**2
    nll_term = torch.sum(log_variance_term + mse_term, dim=batch_dim+1) / 2 + _C
    return nll_term


class GaussianNLL(Metric[float]):

    def __init__(self, confidence_coefficient=0):
        super().__init__()
        self.confidence_coefficient = confidence_coefficient
        self._mean_relative_distance = Mean()

    def reset(self) -> None:
        """Resets the metric to its initial state."""
        self._mean_relative_distance.reset()

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
        """
        Updates the metric with the predicted and actual vectors.

        :param predicted: Predicted vector (batch of predictions)
        :param actual: Actual (ground truth) vector (batch of targets)
        """
        gaussian_mse = gaussian_nll_batch(actual, predicted)
        self._mean_relative_distance.update(gaussian_mse.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "GaussianNLL"


class PluginGaussianNLL(GenericPluginMetric[float, GaussianNLL]):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """
        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(GaussianNLL(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchGaussianNLL(PluginGaussianNLL):

    def __init__(self):
        """
        Creates an instance of the MinibatchGaussianNLL metric.
        """
        super(MinibatchGaussianNLL, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianNLL_MB"


class EpochGaussianNLL(PluginGaussianNLL):

    def __init__(self):
        """
        Creates an instance of the EpochGaussianNLL metric.
        """

        super(EpochGaussianNLL, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "GaussianNLL_Epoch"


class RunningEpochGaussianNLL(PluginGaussianNLL):

    def __init__(self):
        """
        Creates an instance of the RunningEpochGaussianNLL metric.
        """

        super(RunningEpochGaussianNLL, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianNLL_Epoch"


class ExperienceGaussianNLL(PluginGaussianNLL):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceGaussianNLL, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "GaussianNLL_Exp"


class StreamGaussianNLL(PluginGaussianNLL):

    def __init__(self):
        """
        Creates an instance of StreamGaussianNLL metric
        """
        super(StreamGaussianNLL, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "GaussianNLL_Stream"


def gaussian_nll_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PluginGaussianNLL] = []
    if minibatch:
        metrics.append(MinibatchGaussianNLL())
    if epoch:
        metrics.append(EpochGaussianNLL())
    if running_epoch:
        metrics.append(RunningEpochGaussianNLL())
    if experience:
        metrics.append(ExperienceGaussianNLL())
    if stream:
        metrics.append(StreamGaussianNLL())
    return metrics


__all__ = [
    'gaussian_nll_batch',
    'GaussianNLL',
    'PluginGaussianNLL',
    'MinibatchGaussianNLL',
    'EpochGaussianNLL',
    'RunningEpochGaussianNLL',
    'ExperienceGaussianNLL',
    'StreamGaussianNLL',
    'gaussian_nll_metrics',
]
import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean


def gaussian_variance_batch(y_true, y_pred, batch_dim=0):
    num_features = y_pred.shape[1] // 2
    # Split predictions into mean and log_variance
    variance = y_pred[:, num_features:] ** 2
    return variance


class GaussianVariance(Metric[float]):

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
        gaussian_variance = gaussian_variance_batch(actual, predicted)
        self._mean_relative_distance.update(gaussian_variance.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "GaussianVariance"


class PluginGaussianVariance(GenericPluginMetric[float, GaussianVariance]):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """
        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(GaussianVariance(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchGaussianVariance(PluginGaussianVariance):

    def __init__(self):
        """
        Creates an instance of the MinibatchGaussianVariance metric.
        """
        super(MinibatchGaussianVariance, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianVariance_MB"


class EpochGaussianVariance(PluginGaussianVariance):

    def __init__(self):
        """
        Creates an instance of the EpochGaussianVariance metric.
        """

        super(EpochGaussianVariance, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "GaussianVariance_Epoch"


class RunningEpochGaussianVariance(PluginGaussianVariance):

    def __init__(self):
        """
        Creates an instance of the RunningEpochGaussianVariance metric.
        """

        super(RunningEpochGaussianVariance, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianVariance_Epoch"


class ExperienceGaussianVariance(PluginGaussianVariance):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceGaussianVariance, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "GaussianVariance_Exp"


class StreamGaussianVariance(PluginGaussianVariance):

    def __init__(self):
        """
        Creates an instance of StreamGaussianVariance metric
        """
        super(StreamGaussianVariance, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "GaussianVariance_Stream"


def gaussian_variance_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PluginGaussianVariance] = []
    if minibatch:
        metrics.append(MinibatchGaussianVariance())
    if epoch:
        metrics.append(EpochGaussianVariance())
    if running_epoch:
        metrics.append(RunningEpochGaussianVariance())
    if experience:
        metrics.append(ExperienceGaussianVariance())
    if stream:
        metrics.append(StreamGaussianVariance())
    return metrics


__all__ = [
    'gaussian_variance_batch',
    'GaussianVariance',
    'PluginGaussianVariance',
    'MinibatchGaussianVariance',
    'EpochGaussianVariance',
    'RunningEpochGaussianVariance',
    'ExperienceGaussianVariance',
    'StreamGaussianVariance',
    'gaussian_variance_metrics',
]
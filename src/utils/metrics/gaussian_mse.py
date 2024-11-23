import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean


def gaussian_mse_batch(y_true, y_pred, confidence_coefficient=0, batch_dim=0):
    num_features = y_pred.shape[1] // 2
    # Split predictions into mean and log_variance
    mean = y_pred[:, :num_features]
    variance = y_pred[:, num_features:]
    lower_bound = mean - confidence_coefficient * variance
    upper_bound = mean + confidence_coefficient * variance
    sizes = torch.rand_like(upper_bound - lower_bound)
    results = sizes * (upper_bound - lower_bound) + lower_bound
    mse_total = torch.sum((y_true - results) ** 2, dim=batch_dim+1)
    return mse_total  # Unreduced


class GaussianMSE(Metric[float]):

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
        gaussian_mse = gaussian_mse_batch(actual, predicted)
        self._mean_relative_distance.update(gaussian_mse.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "GaussianMSE"


class PluginGaussianMSE(GenericPluginMetric[float, GaussianMSE]):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """
        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(GaussianMSE(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchGaussianMSE(PluginGaussianMSE):

    def __init__(self):
        """
        Creates an instance of the MinibatchGaussianMSE metric.
        """
        super(MinibatchGaussianMSE, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianMSE_MB"


class EpochGaussianMSE(PluginGaussianMSE):

    def __init__(self):
        """
        Creates an instance of the EpochGaussianMSE metric.
        """

        super(EpochGaussianMSE, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "GaussianMSE_Epoch"


class RunningEpochGaussianMSE(PluginGaussianMSE):

    def __init__(self):
        """
        Creates an instance of the RunningEpochGaussianMSE metric.
        """

        super(RunningEpochGaussianMSE, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "GaussianMSE_Epoch"


class ExperienceGaussianMSE(PluginGaussianMSE):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceGaussianMSE, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "GaussianMSE_Exp"


class StreamGaussianMSE(PluginGaussianMSE):

    def __init__(self):
        """
        Creates an instance of StreamGaussianMSE metric
        """
        super(StreamGaussianMSE, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "GaussianMSE_Stream"


def gaussian_mse_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PluginGaussianMSE] = []
    if minibatch:
        metrics.append(MinibatchGaussianMSE())
    if epoch:
        metrics.append(EpochGaussianMSE())
    if running_epoch:
        metrics.append(RunningEpochGaussianMSE())
    if experience:
        metrics.append(ExperienceGaussianMSE())
    if stream:
        metrics.append(StreamGaussianMSE())
    return metrics


__all__ = [
    'gaussian_mse_batch',
    'GaussianMSE',
    'PluginGaussianMSE',
    'MinibatchGaussianMSE',
    'EpochGaussianMSE',
    'RunningEpochGaussianMSE',
    'ExperienceGaussianMSE',
    'StreamGaussianMSE',
    'gaussian_mse_metrics',
]
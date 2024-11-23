import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean


class RelativeDistance(Metric[float]):

    def __init__(self):
        super().__init__()
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
        # Calculate the relative distance for each sample in the batch
        actual_norm = torch.norm(actual, dim=1)
        if actual_norm.sum() == 0:
            actual_norm = torch.ones_like(actual_norm)
        relative_distances = torch.norm(predicted - actual, dim=1) / actual_norm

        # Update the mean relative distance
        self._mean_relative_distance.update(relative_distances.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "RelativeDistance"


class PluginRelativeDistance(GenericPluginMetric[float, RelativeDistance]):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(RelativeDistance(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchRelativeDistance(PluginRelativeDistance):

    def __init__(self):
        """
        Creates an instance of the MinibatchRelativeDistance metric.
        """
        super(MinibatchRelativeDistance, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RelativeDistance_MB"


class EpochRelativeDistance(PluginRelativeDistance):

    def __init__(self):
        """
        Creates an instance of the EpochRelativeDistance metric.
        """

        super(EpochRelativeDistance, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "RelativeDistance_Epoch"


class RunningEpochRelativeDistance(PluginRelativeDistance):

    def __init__(self):
        """
        Creates an instance of the RunningEpochRelativeDistance metric.
        """

        super(RunningEpochRelativeDistance, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RelativeDistance_Epoch"


class ExperienceRelativeDistance(PluginRelativeDistance):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceRelativeDistance, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "RelativeDistance_Exp"


class StreamRelativeDistance(PluginRelativeDistance):

    def __init__(self):
        """
        Creates an instance of StreamRelativeDistance metric
        """
        super(StreamRelativeDistance, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "RelativeDistance_Stream"


def relative_distance_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PluginRelativeDistance] = []
    if minibatch:
        metrics.append(MinibatchRelativeDistance())
    if epoch:
        metrics.append(EpochRelativeDistance())
    if running_epoch:
        metrics.append(RunningEpochRelativeDistance())
    if experience:
        metrics.append(ExperienceRelativeDistance())
    if stream:
        metrics.append(StreamRelativeDistance())
    return metrics


__all__ = [
    'RelativeDistance',
    'PluginRelativeDistance',
    'MinibatchRelativeDistance',
    'EpochRelativeDistance',
    'ExperienceRelativeDistance',
    'StreamRelativeDistance',
    'relative_distance_metrics',
]
import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean


def r2_score_batch(y_true, y_pred, batch_dim=0):
    """
    Compute R² for a batch of values in PyTorch.

    Args:
        y_true (torch.Tensor): Tensor of true values with shape (batch_size, ...).
        y_pred (torch.Tensor): Tensor of predicted values with shape (batch_size, ...).
        batch_dim (int): Dimension representing the batch. Default is 0.

    Returns:
        torch.Tensor: R² values for each batch element.
    """
    size = y_true.shape[batch_dim + 1] # for the case of Gaussian NLL
    # Ensure we're computing R² across the batch dimension
    ss_total = torch.sum((y_true - y_true.mean(dim=batch_dim, keepdim=True)) ** 2, dim=batch_dim)
    ss_residual = torch.sum((y_true - y_pred[:, 0:size]) ** 2, dim=batch_dim) # mean for Gaussian NLL

    # Compute R²
    r2 = 1 - (ss_residual / ss_total)
    return r2


class R2Score(Metric[float]):

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
        r2_score = r2_score_batch(actual, predicted)
        # Update the mean r2 score
        self._mean_relative_distance.update(r2_score.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "R2Score"


class PluginR2Score(GenericPluginMetric[float, R2Score]):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(R2Score(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchR2Score(PluginR2Score):

    def __init__(self):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(MinibatchR2Score, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "R2Score_MB"


class EpochR2Score(PluginR2Score):

    def __init__(self):
        """
        Creates an instance of the EpochR2Score metric.
        """

        super(EpochR2Score, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "R2Score_Epoch"


class RunningEpochR2Score(PluginR2Score):

    def __init__(self):
        """
        Creates an instance of the RunningEpochR2Score metric.
        """

        super(RunningEpochR2Score, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "R2Score_Epoch"


class ExperienceR2Score(PluginR2Score):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceR2Score, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "R2Score_Exp"


class StreamR2Score(PluginR2Score):

    def __init__(self):
        """
        Creates an instance of StreamR2Score metric
        """
        super(StreamR2Score, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "R2Score_Stream"


def r2_score_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PluginR2Score] = []
    if minibatch:
        metrics.append(MinibatchR2Score())
    if epoch:
        metrics.append(EpochR2Score())
    if running_epoch:
        metrics.append(RunningEpochR2Score())
    if experience:
        metrics.append(ExperienceR2Score())
    if stream:
        metrics.append(StreamR2Score())
    return metrics


__all__ = [
    'R2Score',
    'PluginR2Score',
    'MinibatchR2Score',
    'EpochR2Score',
    'ExperienceR2Score',
    'StreamR2Score',
    'r2_score_metrics',
]

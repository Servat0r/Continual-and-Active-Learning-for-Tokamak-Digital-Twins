import torch
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics import Mean
from .templates import *


def binary_accuracy_batch(y_true, y_pred, threshold=0.5):
    """
    Computes binary accuracy with given threshold.
    :param y_true:
    :param y_pred:
    :param threshold:
    :return:
    """
    predicted_y = (y_pred >= threshold).type(torch.float32)
    if (len(y_pred.shape) > 1) and (len(y_true.shape) == 1): # Only batch dim
        y_true = y_true.unsqueeze(1)
    num_correct = (predicted_y == y_true).sum()
    return num_correct / len(y_true)


class BinaryAccuracy(FunctionalMetric):

    def __init__(self):
        super(BinaryAccuracy, self).__init__(metric=binary_accuracy_batch)


class BinaryAccuracyPlugin(FunctionalMetricPlugin):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False, suffix_type='MB'):
        super(FunctionalMetricPlugin, self).__init__(
            BinaryAccuracy(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.suffix_type = suffix_type


class MinibatchBinaryAccuracy(BinaryAccuracyPlugin):

    def __init__(self):
        """
        Creates an instance of the MinibatchBinaryAccuracy metric.
        """
        super(MinibatchBinaryAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", suffix_type='MB'
        )


class EpochBinaryAccuracy(BinaryAccuracyPlugin):

    def __init__(self):
        """
        Creates an instance of the EpochBinaryAccuracy metric.
        """

        super(EpochBinaryAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", suffix_type='Epoch'
        )


class RunningEpochBinaryAccuracy(BinaryAccuracyPlugin):

    def __init__(self):
        """
        Creates an instance of the RunningEpochBinaryAccuracy metric.
        """

        super(RunningEpochBinaryAccuracy, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", suffix_type='Epoch'
        )


class ExperienceBinaryAccuracy(BinaryAccuracyPlugin):

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceBinaryAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval", suffix_type='Exp'
        )


class StreamBinaryAccuracy(BinaryAccuracyPlugin):

    def __init__(self):
        """
        Creates an instance of StreamBinaryAccuracy metric
        """
        super(StreamBinaryAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", suffix_type='Stream'
        )


def binary_accuracy_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[BinaryAccuracyPlugin] = []
    if minibatch:
        metrics.append(MinibatchBinaryAccuracy())
    if epoch:
        metrics.append(EpochBinaryAccuracy())
    if running_epoch:
        metrics.append(RunningEpochBinaryAccuracy())
    if experience:
        metrics.append(ExperienceBinaryAccuracy())
    if stream:
        metrics.append(StreamBinaryAccuracy())
    return metrics


__all__ = [
    "BinaryAccuracy",
    "BinaryAccuracyPlugin",
    "MinibatchBinaryAccuracy",
    "EpochBinaryAccuracy",
    "RunningEpochBinaryAccuracy",
    "ExperienceBinaryAccuracy",
    "StreamBinaryAccuracy",
    "binary_accuracy_metrics",
]
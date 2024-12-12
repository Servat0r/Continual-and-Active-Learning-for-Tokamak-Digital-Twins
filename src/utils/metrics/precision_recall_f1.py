import torch
from .templates import *


# noinspection PyTypeChecker
def __internal(y_pred2, y_true):
    # Calculate true positives, false positives, and false negatives
    tp = torch.sum((y_pred2 == 1) & (y_true == 1))
    fp = torch.sum((y_pred2 == 1) & (y_true == 0))
    fn = torch.sum((y_pred2 == 0) & (y_true == 1))
    return tp, fp, fn


def precision_batch(y_true, y_pred, threshold=0.5):
    y_pred2 = (y_pred >= threshold).type(torch.float32)
    tp, fp, fn = __internal(y_pred2, y_true)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def recall_batch(y_true, y_pred, threshold=0.5):
    y_pred2 = (y_pred >= threshold).type(torch.float32)
    tp, fp, fn = __internal(y_pred2, y_true)

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def f1_batch(y_true, y_pred, threshold=0.5):
    y_pred2 = (y_pred >= threshold).type(torch.float32)
    tp, fp, fn = __internal(y_pred2, y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


# Metrics
class Precision(FunctionalMetric):

    def __init__(self):
        super(Precision, self).__init__(metric=precision_batch)


class Recall(FunctionalMetric):

    def __init__(self):
        super(Recall, self).__init__(metric=recall_batch)


class F1(FunctionalMetric):

    def __init__(self):
        super(F1, self).__init__(metric=f1_batch)


# Precision Plugins
class PrecisionPlugin(FunctionalMetricPlugin):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False, suffix_type='MB'):
        super(FunctionalMetricPlugin, self).__init__(
            Precision(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.suffix_type = suffix_type


class MinibatchPrecision(PrecisionPlugin):

    def __init__(self):
        super(MinibatchPrecision, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", suffix_type='MB'
        )


class EpochPrecision(PrecisionPlugin):

    def __init__(self):
        super(EpochPrecision, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", suffix_type='Epoch'
        )


class RunningEpochPrecision(PrecisionPlugin):

    def __init__(self):
        super(RunningEpochPrecision, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", suffix_type='Epoch'
        )


class ExperiencePrecision(PrecisionPlugin):

    def __init__(self):
        super(ExperiencePrecision, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval", suffix_type='Exp'
        )


class StreamPrecision(PrecisionPlugin):

    def __init__(self):
        super(StreamPrecision, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", suffix_type='Stream'
        )

# Recall Plugins
class RecallPlugin(FunctionalMetricPlugin):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False, suffix_type='MB'):
        super(FunctionalMetricPlugin, self).__init__(
            Recall(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.suffix_type = suffix_type


class MinibatchRecall(RecallPlugin):

    def __init__(self):
        super(MinibatchRecall, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", suffix_type='MB'
        )


class EpochRecall(RecallPlugin):

    def __init__(self):
        super(EpochRecall, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", suffix_type='Epoch'
        )


class RunningEpochRecall(RecallPlugin):

    def __init__(self):
        super(RunningEpochRecall, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", suffix_type='Epoch'
        )


class ExperienceRecall(RecallPlugin):

    def __init__(self):
        super(ExperienceRecall, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval", suffix_type='Exp'
        )


class StreamRecall(RecallPlugin):

    def __init__(self):
        super(StreamRecall, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", suffix_type='Stream'
        )


# F1 Plugins
class F1Plugin(FunctionalMetricPlugin):

    def __init__(self, reset_at, emit_at, mode, split_by_task=False, suffix_type='MB'):
        super(FunctionalMetricPlugin, self).__init__(
            F1(), reset_at=reset_at, emit_at=emit_at, mode=mode
        )
        self.suffix_type = suffix_type


class MinibatchF1(F1Plugin):

    def __init__(self):
        super(MinibatchF1, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train", suffix_type='MB'
        )


class EpochF1(F1Plugin):

    def __init__(self):
        super(EpochF1, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train", suffix_type='Epoch'
        )


class RunningEpochF1(F1Plugin):

    def __init__(self):
        super(RunningEpochF1, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train", suffix_type='Epoch'
        )


class ExperienceF1(F1Plugin):

    def __init__(self):
        super(ExperienceF1, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval", suffix_type='Exp'
        )


class StreamF1(F1Plugin):

    def __init__(self):
        super(StreamF1, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval", suffix_type='Stream'
        )


# Helper Functions
def precision_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[PrecisionPlugin] = []
    if minibatch:
        metrics.append(MinibatchPrecision())
    if epoch:
        metrics.append(EpochPrecision())
    if running_epoch:
        metrics.append(RunningEpochPrecision())
    if experience:
        metrics.append(ExperiencePrecision())
    if stream:
        metrics.append(StreamPrecision())
    return metrics


def recall_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[RecallPlugin] = []
    if minibatch:
        metrics.append(MinibatchRecall())
    if epoch:
        metrics.append(EpochRecall())
    if running_epoch:
        metrics.append(RunningEpochRecall())
    if experience:
        metrics.append(ExperienceRecall())
    if stream:
        metrics.append(StreamRecall())
    return metrics


def f1_metrics(
        *, minibatch=False, epoch=False, running_epoch=False, experience=False, stream=False
):
    metrics: list[F1Plugin] = []
    if minibatch:
        metrics.append(MinibatchF1())
    if epoch:
        metrics.append(EpochF1())
    if running_epoch:
        metrics.append(RunningEpochF1())
    if experience:
        metrics.append(ExperienceF1())
    if stream:
        metrics.append(StreamF1())
    return metrics


__all__ = [
    'Precision', 'Recall', 'F1', 'PrecisionPlugin', 'RecallPlugin', 'F1Plugin',
    'MinibatchPrecision', 'EpochPrecision', 'RunningEpochPrecision', 'ExperiencePrecision', 'StreamPrecision',
    'MinibatchRecall', 'EpochRecall', 'RunningEpochRecall', 'ExperienceRecall', 'StreamRecall',
    'MinibatchF1', 'EpochF1', 'RunningEpochF1', 'ExperienceF1', 'StreamF1',
    'precision_metrics', 'recall_metrics', 'f1_metrics',
]
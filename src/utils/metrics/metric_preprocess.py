from typing import Callable
import torch
from avalanche.evaluation import Metric, GenericPluginMetric

from .forgetting_bwt_fwt import ExperienceWiseForgetting, ExperienceWiseForwardTransfer, ExperienceWiseBWT, StreamWiseForgetting, \
    StreamWiseForwardTransfer, StreamWiseBWT


class PreprocessMetric(Metric[float]):

    def __init__(
        self, metric: Metric, preprocess_ytrue: Callable, preprocess_ypred: Callable
    ):
        super().__init__()
        self.metric = metric
        self.preprocess_ytrue = preprocess_ytrue
        self.preprocess_ypred = preprocess_ypred

    def reset(self) -> None:
        self.metric.reset()

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
        predicted = self.preprocess_ypred(predicted)
        actual = self.preprocess_ytrue(actual)
        self.metric.update(predicted, actual)

    def result(self) -> float:
        return self.metric.result()

    def __str__(self):
        return f"{type(self.metric).__name__}"


class PluginPreprocess(GenericPluginMetric[float, PreprocessMetric]):

    def __init__(
        self, reset_at, emit_at, mode, metric: Metric, preprocess_ytrue: Callable,
        preprocess_ypred: Callable, split_by_task=False,
    ):
        super().__init__(
            PreprocessMetric(
                metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
            ),
            reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)


class MinibatchPreprocess(PluginPreprocess):

    def __init__(self, metric, preprocess_ytrue, preprocess_ypred):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(MinibatchPreprocess, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train",
            metric=metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
        )

    def __str__(self):
        metric_name = str(self._metric)
        if metric_name == 'LossMetric':
            metric_name = 'Loss'
        return f"{metric_name}_MB"

    def reset(self) -> None:
        super(MinibatchPreprocess, self).reset()

    def result(self) -> float:
        return super(MinibatchPreprocess, self).result()

    def update(self, strategy):
        return super(MinibatchPreprocess, self).update(strategy)


class EpochPreprocess(PluginPreprocess):

    def __init__(self, metric, preprocess_ytrue, preprocess_ypred):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(EpochPreprocess, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train",
            metric=metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
        )

    def __str__(self):
        metric_name = str(self._metric)
        if metric_name == 'LossMetric':
            metric_name = 'Loss'
        return f"{metric_name}_Epoch"

    def reset(self) -> None:
        super(EpochPreprocess, self).reset()

    def result(self) -> float:
        return super(EpochPreprocess, self).result()

    def update(self, strategy):
        return super(EpochPreprocess, self).update(strategy)


class RunningEpochPreprocess(PluginPreprocess):

    def __init__(self, metric, preprocess_ytrue, preprocess_ypred):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(RunningEpochPreprocess, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train",
            metric=metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
        )

    def __str__(self):
        metric_name = str(self._metric)
        if metric_name == 'LossMetric':
            metric_name = 'Loss'
        return f"{metric_name}_RunningEpoch"

    def reset(self) -> None:
        super(RunningEpochPreprocess, self).reset()

    def result(self) -> float:
        return super(RunningEpochPreprocess, self).result()

    def update(self, strategy):
        return super(RunningEpochPreprocess, self).update(strategy)


class ExperiencePreprocess(PluginPreprocess):

    def __init__(self, metric, preprocess_ytrue, preprocess_ypred):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(ExperiencePreprocess, self).__init__(
            reset_at="experience", emit_at="experience", mode="train",
            metric=metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
        )

    def __str__(self):
        metric_name = str(self._metric)
        if metric_name == 'LossMetric':
            metric_name = 'Loss'
        return f"{metric_name}_Exp"

    def reset(self) -> None:
        super(ExperiencePreprocess, self).reset()

    def result(self) -> float:
        return super(ExperiencePreprocess, self).result()

    def update(self, strategy):
        return super(ExperiencePreprocess, self).update(strategy)


class StreamPreprocess(PluginPreprocess):

    def __init__(self, metric, preprocess_ytrue, preprocess_ypred):
        """
        Creates an instance of the MinibatchR2Score metric.
        """
        super(StreamPreprocess, self).__init__(
            reset_at="stream", emit_at="stream", mode="train",
            metric=metric, preprocess_ytrue=preprocess_ytrue, preprocess_ypred=preprocess_ypred
        )

    def __str__(self):
        metric_name = str(self._metric)
        if metric_name == 'LossMetric':
            metric_name = 'Loss'
        return f"{metric_name}_Stream"

    def reset(self) -> None:
        super(StreamPreprocess, self).reset()

    def result(self) -> float:
        return super(StreamPreprocess, self).result()

    def update(self, strategy):
        return super(StreamPreprocess, self).update(strategy)


def preprocessed_metrics(metrics: list[GenericPluginMetric], preprocess_ytrue, preprocess_ypred):
    result = []
    for metric in metrics:
        if hasattr(metric, '_metric'):
            print(f"Pippo {metric._metric}")
        elif hasattr(metric, '_current_metric'):
            print(f"Pippo {metric._current_metric}")
        # Special Metrics
        if any([
            isinstance(metric, cls) for cls in \
                [ExperienceWiseForgetting, ExperienceWiseForwardTransfer, ExperienceWiseBWT]
        ]):
            result.append(ExperiencePreprocess(metric._current_metric._metric, preprocess_ytrue, preprocess_ypred))
        elif any([
            isinstance(metric, cls) for cls in \
                [StreamWiseForgetting, StreamWiseForwardTransfer, StreamWiseBWT]
        ]):
            result.append(StreamPreprocess(metric._current_metric._metric, preprocess_ytrue, preprocess_ypred))
        # Generic Metrics
        elif metric._reset_at == "iteration":
            result.append(MinibatchPreprocess(metric._metric, preprocess_ytrue, preprocess_ypred))
        elif metric._reset_at == "experience":
            result.append(ExperiencePreprocess(metric._metric, preprocess_ytrue, preprocess_ypred))
        elif metric._reset_at == "stream":
            result.append(StreamPreprocess(metric._metric, preprocess_ytrue, preprocess_ypred))
        elif metric._reset_at == "epoch":
            if metric._emit_at == "iteration":
                result.append(RunningEpochPreprocess(metric._metric, preprocess_ytrue, preprocess_ypred))
            elif metric._emit_at == "epoch":
                result.append(EpochPreprocess(metric._metric, preprocess_ytrue, preprocess_ypred))
            else:
                raise ValueError(f"Invalid combination: emit_at = {metric._emit_at}, reset_at = {metric._reset_at}")
    return result


__all__ = [
    'PreprocessMetric',
    'PluginPreprocess',
    'MinibatchPreprocess',
    'EpochPreprocess',
    'RunningEpochPreprocess',
    'ExperiencePreprocess',
    'StreamPreprocess',
    'preprocessed_metrics',
]

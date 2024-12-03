from typing import Optional, Dict, Sequence

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import GenericExperienceForgetting, \
    GenericStreamForgetting, GenericExperienceForwardTransfer, GenericStreamForwardTransfer
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_to_bwt
from avalanche.evaluation.metrics.loss import LossPluginMetric

from .r2_score import ExperienceR2Score, StreamR2Score, PluginR2Score


class ExperienceWiseForgetting(GenericExperienceForgetting):

    def __init__(self, metric):
        super().__init__(metric)

    def result_key(self, k: int) -> Optional[float]:
        return self.forgetting.result_key(k=k)

    def result(self) -> Dict[int, float]:
        return self.forgetting.result()

    def metric_update(self, strategy):
        if any([isinstance(self._current_metric, pm) for pm in [LossPluginMetric, PluginR2Score]]):
            self._current_metric.update(strategy)
        else:
            self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        result = self._current_metric.result()
        return result[0] if isinstance(result, Sequence) else result

    def __str__(self):
        return "Forgetting_Exp"


class StreamWiseForgetting(GenericStreamForgetting):

    def __init__(self, metric):
        super().__init__(metric)

    def result_key(self, k: int) -> Optional[float]:
        return self.forgetting.result_key(k=k)

    def result(self) -> Dict[int, float]:
        return self.forgetting.result()

    def metric_update(self, strategy):
        if any([isinstance(self._current_metric, pm) for pm in [LossPluginMetric, PluginR2Score]]):
            self._current_metric.update(strategy)
        else:
            self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        result = self._current_metric.result()
        return result[0] if isinstance(result, Sequence) else result

    def __str__(self):
        return "Forgetting_Stream"


class ExperienceWiseForwardTransfer(GenericExperienceForwardTransfer):

    def __init__(self, metric):
        super().__init__(metric)

    def result_key(self, k: int) -> Optional[float]:
        return self.forward_transfer.result_key(k=k)

    def result(self) -> Dict[int, float]:
        return self.forward_transfer.result()

    def metric_update(self, strategy):
        if any([isinstance(self._current_metric, pm) for pm in [LossPluginMetric, PluginR2Score]]):
            self._current_metric.update(strategy)
        else:
            self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        result = self._current_metric.result()
        return result[0] if isinstance(result, Sequence) else result

    def __str__(self):
        return "ForwardTransfer_Exp"


class StreamWiseForwardTransfer(GenericStreamForwardTransfer):

    def __init__(self, metric):
        super().__init__(metric)

    def metric_update(self, strategy):
        if any([isinstance(self._current_metric, pm) for pm in [LossPluginMetric, PluginR2Score]]):
            self._current_metric.update(strategy)
        else:
            self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)

    def metric_result(self, strategy):
        result = self._current_metric.result()
        return result[0] if isinstance(result, Sequence) else result

    def __str__(self):
        return "ForwardTransfer_Stream"


class ExperienceWiseBWT(ExperienceWiseForgetting):

    def result_key(self, k=None) -> Optional[float]:
        forgetting = super().result_key(k)
        return forgetting_to_bwt(forgetting)

    def result(self) -> Dict[int, float]:
        forgetting = super().result()
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "BWT_Exp"


class StreamWiseBWT(StreamWiseForgetting):

    def exp_result(self, k: int) -> Optional[float]:
        forgetting = super().exp_result(k)
        return forgetting_to_bwt(forgetting)

    def __str__(self):
        return "BWT_Stream"


def renamed_forgetting_metrics(*, experience=False, stream=False, base_metrics=None):
    metrics: list[PluginMetric] = []
    base_metrics = base_metrics if base_metrics is not None else [ExperienceR2Score(), StreamR2Score()]
    if experience:
        for metric in base_metrics:
            if str(metric).endswith('_Exp'):
                metrics.append(ExperienceWiseForgetting(metric))
    if stream:
        for metric in base_metrics:
            if str(metric).endswith('_Stream'):
                metrics.append(StreamWiseForgetting(metric))
    return metrics


def renamed_fwt_metrics(*, experience=False, stream=False, base_metrics=None):
    metrics: list[PluginMetric] = []
    base_metrics = base_metrics if base_metrics is not None else [ExperienceR2Score(), StreamR2Score()]
    if experience:
        for metric in base_metrics:
            if str(metric).endswith('_Exp'):
                metrics.append(ExperienceWiseForwardTransfer(metric))
    if stream:
        for metric in base_metrics:
            if str(metric).endswith('_Stream'):
                metrics.append(StreamWiseForwardTransfer(metric))
    return metrics


def renamed_bwt_metrics(*, experience=False, stream=False, base_metrics=None):
    metrics: list[PluginMetric] = []
    base_metrics = base_metrics if base_metrics is not None else [ExperienceR2Score(), StreamR2Score()]
    if experience:
        for metric in base_metrics:
            if str(metric).endswith('_Exp'):
                metrics.append(ExperienceWiseBWT(metric))
    if stream:
        for metric in base_metrics:
            if str(metric).endswith('_Exp'):
                metrics.append(StreamWiseBWT(metric))
    return metrics


__all__ = [
    'ExperienceWiseForgetting',
    'StreamWiseForgetting',
    'ExperienceWiseForwardTransfer',
    'StreamWiseForwardTransfer',
    'ExperienceWiseBWT',
    'StreamWiseBWT',
    'renamed_forgetting_metrics',
    'renamed_fwt_metrics',
    'renamed_bwt_metrics',
]
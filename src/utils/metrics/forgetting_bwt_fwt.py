from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import ExperienceForgetting, \
    StreamForgetting, ExperienceForwardTransfer, StreamForwardTransfer, \
    ExperienceBWT, StreamBWT


class ExperienceWiseForgetting(ExperienceForgetting):

    def __str__(self):
        return "Forgetting_Exp"


class StreamWiseForgetting(StreamForgetting):

    def __str__(self):
        return "Forgetting_Stream"


class ExperienceWiseForwardTransfer(ExperienceForwardTransfer):

    def __str__(self):
        return "ForwardTransfer_Exp"


class StreamWiseForwardTransfer(StreamForwardTransfer):

    def __str__(self):
        return "ForwardTransfer_Stream"


class ExperienceWiseBWT(ExperienceBWT):

    def __str__(self):
        return "BWT_Exp"


class StreamWiseBWT(StreamBWT):

    def __str__(self):
        return "BWT_Stream"


def renamed_forgetting_metrics(*, experience=False, stream=False):
    metrics: list[PluginMetric] = []
    if experience:
        metrics.append(ExperienceWiseForgetting())
    if stream:
        metrics.append(StreamWiseForgetting())
    return metrics


def renamed_fwt_metrics(*, experience=False, stream=False):
    metrics: list[PluginMetric] = []
    if experience:
        metrics.append(ExperienceWiseForwardTransfer())
    if stream:
        metrics.append(StreamWiseForwardTransfer())
    return metrics


def renamed_bwt_metrics(*, experience=False, stream=False):
    metrics: list[PluginMetric] = []
    if experience:
        metrics.append(ExperienceWiseBWT())
    if stream:
        metrics.append(StreamWiseBWT())
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
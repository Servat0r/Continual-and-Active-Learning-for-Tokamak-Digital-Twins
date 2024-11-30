import json
from typing import Any

from avalanche.core import Template
from avalanche.logging import BaseLogger

from .. import extract_metric_info
from ..metrics import MeanStdMetric


class MeanStdPlugin(BaseLogger):

    def __init__(self, metric_names: list[str], num_experiences: int):
        super().__init__()
        self.num_experiences = num_experiences
        self.metric_names = metric_names
        self.results = []
        self.metrics = [MeanStdMetric(metric_names) for _ in range(num_experiences)]
        self.in_train_phase = False
        self.training_exp = -1

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.in_train_phase = True
        self.training_exp += 1

    def after_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.in_train_phase = False

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        current_eval_exp = strategy.experience.current_experience
        if current_eval_exp <= self.training_exp:
            if not self.in_train_phase:
                metrics = {extract_metric_info(val.name)['name']: val.value for val in metric_values}
                print(f"{type(self).__name__}:", metrics, sep='\n')
                self.metrics[current_eval_exp].update(metrics)
                self.results.append(self.metrics[current_eval_exp].result())

    def dump_results(self, file_path_or_buf):
        if isinstance(file_path_or_buf, str):
            fp = open(file_path_or_buf, "w")
        else:
            fp = file_path_or_buf
        json.dump(self.results, fp, indent=2)
        fp.close()


__all__ = ["MeanStdPlugin"]
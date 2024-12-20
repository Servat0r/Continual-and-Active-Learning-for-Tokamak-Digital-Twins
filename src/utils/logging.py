import os
import sys

import torch
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation import GenericPluginMetric
from avalanche.logging import BaseLogger
from avalanche.training.templates import SupervisedTemplate

from .misc import debug_print, extract_metric_info, extract_metric_type


########################### Helpers
def get_log_folder(
        pow_type: str, cluster_type: str, task: str, dataset_type: str, outputs: str | list[str],
        strategy: str, extra_log_folder: str, hour: int | str, minute: int | str, seconds: int | str,
        day: int | str, month: int | str, year: int = 2024, task_id: int = 0,
) -> str:
    """
    Retrieves the EXACT log folder path according to the given parameters.
    :param pow_type: One of {"highpow", "lowpow"}.
    :param cluster_type: One of {"Ip_Pin_based", "tau_based", "pca_based"}.
    :param task: One of {"classification", "regression"}.
    :param dataset_type: One of {"complete", "not_null"}.
    :param outputs: Either a string or a list of strings, each per output columns. If a string, it must
    be of the form of the output of "_".join(outputs_list).
    :param strategy: Strategy name, e.g. "Naive" or "Replay".
    :param extra_log_folder: Extra log folder path (see README).
    :param hour: Hour of the day when run started.
    :param minute: Minute of the day when run started.
    :param seconds: Seconds of the day when run started.
    :param day: Day of the month when run started.
    :param month: Month of the year when run started.
    :param year: Year when run started.
    :param task_id: Run id, in {0, ..., N-1}.
    :return: Log folder path.
    """
    params = {
        'hour': hour, 'minute': minute, 'seconds': seconds,
        'day': day, 'month': month, 'year': year
    }
    for name, value in params.items():
        if isinstance(value, int) and value < 10:
            params[name] = '0' + str(value)
    hour = params['hour']
    minute = params['minute']
    seconds = params['seconds']
    day = params['day']
    month = params['month']
    year = params['year']
    basepath = f"{year}-{month}-{day}_{hour}-{minute}-{seconds}"
    outputs_string = outputs if isinstance(outputs, str) else '_'.join(outputs)
    index_dir = os.path.join(
        'logs', pow_type, cluster_type, task, dataset_type, outputs_string, strategy, extra_log_folder
    )
    print(index_dir)
    for dirname in os.listdir(index_dir):
        if dirname.startswith(basepath) and dirname.endswith(f"task_{task_id}"):
            return os.path.join(index_dir, dirname)
    raise ValueError(f"Not found any directory starting with \"{basepath}\"")
############################à


class CustomCSVLogger(BaseLogger):
    """
    Custom CSV logger, used to log all needed metrics. For each <type> of train/eval/test, it creates
    three files: <type>_results_epoch.csv, <type>_results_experience.csv, <type>_results_stream.csv,
    collecting metric values at the end of each epoch or experience, or at the end of the whole stream.
    """

    VAL = 'val'
    TEST = 'test'

    def __init__(
            self, log_folder: str = None, metrics: list[GenericPluginMetric] = None,
            val_stream=None,
    ):
        """
        :param log_folder: Directory in which to create log files.
        :param metrics: Which metrics to track.
        :param val_stream: An object in CLScenario().streams representing validation stream.
        """
        super().__init__()
        self.log_folder = log_folder if log_folder is not None else "csvlogs"
        os.makedirs(self.log_folder, exist_ok=True)

        self.training_files = {
            'epoch': open(os.path.join(self.log_folder, "training_results_epoch.csv"), "w"), # Epoch
            'exp': open(os.path.join(self.log_folder, "training_results_experience.csv"), "w"), # Experience
            'stream': open(os.path.join(self.log_folder, "training_results_stream.csv"), "w"), # Stream
        }
        self.eval_files = {
            'epoch': open(os.path.join(self.log_folder, "eval_results_epoch.csv"), "w"), # Epoch
            'exp': open(os.path.join(self.log_folder, "eval_results_experience.csv"), "w"), # Experience
            'stream': open(os.path.join(self.log_folder, "eval_results_stream.csv"), "w"), # Stream
        }
        self.test_files = {
            'epoch': open(os.path.join(self.log_folder, "test_results_epoch.csv"), "w"), # Epoch
            'exp': open(os.path.join(self.log_folder, "test_results_experience.csv"), "w"), # Experience
            'stream': open(os.path.join(self.log_folder, "test_results_stream.csv"), "w") # Stream
        }

        # current training experience id
        self.training_exp_id = None

        # is open?
        self.is_open = True
        self.stream_type = self.VAL
        self.set_val_stream_type()

        # if we are currently training or evaluating
        # evaluation within training will not change this flag
        self.in_train_phase = None

        # validation metrics computed during training
        self.val_acc, self.val_loss = 0, 0

        self.metric_names = {
            'epoch': [],
            'exp': [],
            'stream': []
        }

        for metric in metrics:
            debug_print(metric, flush=True)
            metric_name, metric_type = extract_metric_type(str(metric))
            self.metric_names[metric_type].append(str(metric))

        # Debug print
        debug_print(self.metric_names, flush=True)

        # print csv headers
        for filetype, file in self.training_files.items():
            print(
                "training_exp",
                "epoch",
                "lr",
                *self.metric_names[filetype],
                sep=",",
                file=file,
                flush=True,
            )
        for filetype, file in self.eval_files.items():
            if filetype == 'epoch':
                print(
                    "training_exp",
                    "epoch",
                    "lr",
                    *self.metric_names[filetype],
                    sep=",",
                    file=file,
                    flush=True,
                )
            else:
                print(
                    "eval_exp",
                    "training_exp",
                    *self.metric_names[filetype],
                    sep=",",
                    file=file,
                    flush=True,
                )
        self.val_stream = None
        self.val_experiences = None
        if val_stream is not None:
            self.set_validation_stream(val_stream)

    def set_val_stream_type(self):
        self.stream_type = self.VAL

    def set_test_stream_type(self):
        self.stream_type = self.TEST

    def set_validation_stream(self, stream):
        self.val_stream = stream
        self.val_experiences = []
        for val_exp in self.val_stream:
            self.val_experiences.append(val_exp)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return "\n" + str(m_val)
        elif isinstance(m_val, float):
            return f"{m_val:.4f}"
        else:
            return str(m_val)

    def print_train_metrics(
        self, training_exp, epoch, lr, metric_values, type,
    ):
        file = self.training_files[type]
        metric_names = self.metric_names[type]
        metric_raw_values = [0.0 for _ in range(len(metric_names))]
        for val in metric_values:
            metric_info = extract_metric_info(val.name)
            if metric_info['type'] == type:
                metric_name = metric_info['name']
                metric_idx = metric_names.index(metric_name)
                metric_raw_values[metric_idx] = self._val_to_str(val.value)
        print(
            training_exp,
            epoch,
            lr,
            *metric_raw_values,
            sep=",",
            file=file,
            flush=True,
        )

    def print_eval_metrics(
        self, eval_exp, training_exp, metric_values, out_type, in_type, epoch=None, lr=None,
    ):
        if self.stream_type == self.VAL:
            file = self.eval_files[out_type]
        elif self.stream_type == self.TEST:
            file = self.test_files[out_type]
        else:
            raise RuntimeError(f"Invalid stream type {self.stream_type}")
        metric_names = self.metric_names[in_type]
        metric_raw_values = [0.0 for _ in range(len(metric_names))]
        for val in metric_values:
            metric_info = extract_metric_info(val.name)
            if metric_info['type'] == in_type:
                metric_name = metric_info['name']
                metric_idx = metric_names.index(metric_name)
                metric_raw_values[metric_idx] = self._val_to_str(val.value)
        if out_type == 'epoch':
            print(
                training_exp,
                epoch,
                lr,
                *metric_raw_values,
                sep=",",
                file=file,
                flush=True,
            )
        else:
            print(
                eval_exp,
                training_exp,
                *metric_raw_values,
                sep=",",
                file=file,
                flush=True,
            )

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        #super().after_training_epoch(strategy, metric_values, **kwargs)
        if not self.is_open:
            print(
                "Logger has been closed",
                file=sys.stderr,
                flush=True,
            )
            return
        self.print_train_metrics(
            self.training_exp_id,
            strategy.clock.train_exp_epochs,
            strategy.optimizer.param_groups[0]['lr'],
            metric_values,
            type='epoch',
        )

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        #super().after_eval_exp(strategy, metric_values, **kwargs)
        if not self.is_open:
            print(
                "Logger has been closed",
                file=sys.stderr,
                flush=True,
            )
            return
        if self.in_train_phase:
            self.print_eval_metrics(
                strategy.experience.current_experience,
                self.training_exp_id,
                metric_values,
                out_type='epoch',
                in_type='exp',
                epoch=strategy.clock.train_exp_epochs,
                lr=strategy.optimizer.param_groups[0]['lr'],
            )
        else:
            self.print_eval_metrics(
                strategy.experience.current_experience,
                self.training_exp_id,
                metric_values,
                out_type='exp',
                in_type='exp',
            )

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        #super().before_training(strategy, metric_values, **kwargs)
        self.training_exp_id = strategy.experience.current_experience

    def before_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        """
        Manage the case in which `eval` is first called before `train`
        """
        if self.in_train_phase is None:
            self.in_train_phase = False

    def before_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        self.in_train_phase = True

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: list["MetricValue"],
        **kwargs,
    ):
        self.in_train_phase = False

    def close(self):
        self.is_open = False
        for file in self.training_files.values():
            file.close()
        for file in self.eval_files.values():
            file.close()
        for file in self.test_files.values():
            file.close()


__all__ = [
    "get_log_folder", "CustomCSVLogger"
]
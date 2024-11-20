import os
import torch
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation import GenericPluginMetric
from avalanche.logging import BaseLogger
from avalanche.training.templates import SupervisedTemplate

from .misc import debug_print


########################### Helpers
def extract_metric_info(metric_name: str) -> dict[str, str]:
    values = metric_name.split("/")
    results = {
        'name': values[0],
        'phase': values[1],
        'stream': values[2] if len(values) > 2 else None,
        'exp': values[3] if len(values) > 3 else None,
    }
    if results['name'].endswith('Epoch'):
        results['type'] = 'epoch'
    elif results['name'].endswith('Exp'):
        results['type'] = 'exp'
    elif results['name'].endswith('Stream'):
        results['type'] = 'stream'
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")
    return results


def extract_metric_type(metric_name: str):
    if metric_name.endswith('Epoch'):
        return metric_name[:-6], 'epoch'
    elif metric_name.endswith('Exp'):
        return metric_name[:-4], 'exp'
    elif metric_name.endswith('Stream'):
        return metric_name[:-7], 'stream'
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")

def get_log_folder(
        pow_type, cluster_type, task, dataset_type, strategy, folder_name,
        hour, minute, seconds, day, month, year=2024
):
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
    index_dir = os.path.join('logs', pow_type, cluster_type, task, dataset_type, strategy)
    print(index_dir)
    for dirname in os.listdir(index_dir):
        if dirname.startswith(basepath):
            return os.path.join(index_dir, dirname)
    raise ValueError(f"Not found any directory starting with \"{basepath}\"")
############################à


class CustomCSVLogger(BaseLogger):
    def __init__(self, log_folder=None, metrics: list[GenericPluginMetric] = None):
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

        # current training experience id
        self.training_exp_id = None

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
                *self.metric_names[filetype],
                sep=",",
                file=file,
                flush=True,
            )
        for filetype, file in self.eval_files.items():
            print(
                "eval_exp",
                "training_exp",
                *self.metric_names[filetype],
                sep=",",
                file=file,
                flush=True,
            )

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return "\n" + str(m_val)
        elif isinstance(m_val, float):
            return f"{m_val:.4f}"
        else:
            return str(m_val)

    def print_train_metrics(
        self, training_exp, epoch, metric_values, type,
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
            *metric_raw_values,
            sep=",",
            file=file,
            flush=True,
        )

    def print_eval_metrics(
        self, eval_exp, training_exp, metric_values, type,
    ):
        file = self.eval_files[type]
        metric_names = self.metric_names[type]
        metric_raw_values = [0.0 for _ in range(len(metric_names))]
        for val in metric_values:
            metric_info = extract_metric_info(val.name)
            if metric_info['type'] == type:
                metric_name = metric_info['name']
                metric_idx = metric_names.index(metric_name)
                metric_raw_values[metric_idx] = self._val_to_str(val.value)
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
        self.print_train_metrics(
            self.training_exp_id,
            strategy.clock.train_exp_epochs,
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
        # Temporarily ignore the case of validation during training (refer to CSVLogger class for implementing)
        if not self.in_train_phase:
            self.print_eval_metrics(
                strategy.experience.current_experience,
                self.training_exp_id,
                metric_values,
                type='exp',
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
        for file in self.training_files.values():
            file.close()
        for file in self.eval_files.values():
            file.close()


__all__ = [
    "extract_metric_info", "extract_metric_type",
    "get_log_folder", "CustomCSVLogger"
]
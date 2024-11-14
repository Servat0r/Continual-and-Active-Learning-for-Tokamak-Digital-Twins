import os
import csv
import torch

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue
from avalanche.logging import BaseLogger


class CustomCSVLogger(SupervisedPlugin):
    def __init__(self, filename='metrics_log.csv', custom_metrics=None):
        """
        Custom CSV Logger for Avalanche.

        :param filename: Name of the CSV file to log the metrics.
        :param custom_metrics: Dictionary of custom metric names and Metric objects.
        """
        super().__init__()
        self.filename = filename
        self.custom_metrics = custom_metrics if custom_metrics else {}

        # Initialize CSV file with headers
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['epoch', 'iteration', 'phase'] + list(self.custom_metrics.keys())
            writer.writerow(headers)

    def log_metrics(self, strategy, phase):
        """
        Log the custom metrics to the CSV file.
        """
        metrics_values = {name: metric.result() for name, metric in self.custom_metrics.items()}
        log_data = [
            strategy.clock.train_exp_counter if phase == "train" else strategy.clock.eval_exp_counter,
            strategy.clock.train_iterations if phase == "train" else strategy.clock.eval_iterations,
            phase
        ]
        log_data.extend(metrics_values.values())

        # Write to CSV file
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_data)

    def before_training_epoch(self, strategy, **kwargs):
        # Reset metrics at the start of each epoch
        for metric in self.custom_metrics.values():
            metric.reset()

    def after_training_iteration(self, strategy, **kwargs):
        # Update and log metrics at each iteration
        for metric in self.custom_metrics.values():
            metric.update(strategy)
        self.log_metrics(strategy, "train")

    def before_eval(self, strategy, **kwargs):
        # Reset metrics before each evaluation
        for metric in self.custom_metrics.values():
            metric.reset()

    def after_eval_iteration(self, strategy, **kwargs):
        # Update and log metrics at each evaluation iteration
        for metric in self.custom_metrics.values():
            metric.update(strategy)
        self.log_metrics(strategy, "eval")


__all__ = ["CustomCSVLogger"]

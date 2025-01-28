from typing import Dict
import numpy as np


class MeanStdMetric:
    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names
        self.metric_values = {metric: [] for metric in metric_names}
        self.experience_results = []

    def update(self, metrics: Dict[str, float]):
        """
        Update the metric values with new results.

        Parameters:
        - metrics (Dict[str, float]): Dictionary of metric names and their values.
        """
        for metric_name in self.metric_names:
            if metric_name in metrics:
                self.metric_values[metric_name].append(metrics[metric_name])

    def result(self):
        """
        Compute the mean, standard deviation, and adjusted mean at the end of an experience.
        Returns:
        - result (dict): A dictionary with mean, std, and adjusted mean values.
        """
        results = {
            'mean': {},
            'std': {},
            'adjusted_mean': {},
        }
        for metric_name in self.metric_names:
            if len(self.metric_values[metric_name]) > 0:
                results['mean'][metric_name] = np.mean(self.metric_values[metric_name])
                results['std'][metric_name] = np.std(self.metric_values[metric_name])
                results['adjusted_mean'][metric_name] = np.mean(self.metric_values[metric_name]) \
                                                        - np.min(self.metric_values[metric_name])
        self.experience_results.append(results)
        return results

    def reset(self):
        """
        Reset the metric values for a new experience.
        """
        self.metric_values = {metric: [] for metric in self.metric_names}
        self.experience_results = []


__all__ = ['MeanStdMetric']

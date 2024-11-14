import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Mean


class RelativeDistance(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._mean_relative_distance = Mean()

    def reset(self) -> None:
        """Resets the metric to its initial state."""
        self._mean_relative_distance.reset()

    def update(self, predicted: torch.Tensor, actual: torch.Tensor) -> None:
        """
        Updates the metric with the predicted and actual vectors.

        :param predicted: Predicted vector (batch of predictions)
        :param actual: Actual (ground truth) vector (batch of targets)
        """
        # Calculate the relative distance for each sample in the batch
        relative_distances = torch.norm(predicted - actual, dim=1) / torch.norm(actual, dim=1)

        # Update the mean relative distance
        self._mean_relative_distance.update(relative_distances.mean().item())

    def result(self) -> float:
        """Returns the mean relative distance across all samples."""
        return self._mean_relative_distance.result()

    def __str__(self):
        return "RelativeDistance"

# Example of usage within an Avalanche evaluation plugin
# evaluator = EvaluationPlugin(RelativeDistance(), ...)

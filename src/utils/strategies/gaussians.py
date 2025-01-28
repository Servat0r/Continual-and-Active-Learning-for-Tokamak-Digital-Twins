import torch
from avalanche.training.supervised import Naive
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate


class GaussianNaive(SupervisedTemplate):

    def training_epoch(self, **kwargs):
        for batch in self.dataloader:
            self._before_training_iteration(**kwargs)
            x, y = batch[0], batch[1]
            mean, log_var = self.model(x)
            loss = self._criterion(mean, log_var, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._after_training_iteration(**kwargs)

    def criterion(self, outputs, targets):
        """
        Custom criterion using GaussianNLLLoss.
        :param outputs: Tuple of (mean, variance).
        :param targets: Ground truth labels.
        """
        # Expect outputs as (mean, variance)
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError(f"Model outputs must be a tuple (mean, variance), got: {outputs}.")
        mean, variance = outputs
        return self.loss_function(mean, targets, variance)


__all__ = ['GaussianNaive']